from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from .height_scan import FlatHeightScan
    from .low_level_controller import LowLevelCommand, LowLevelController, ZeroTorqueController
except ImportError:
    from height_scan import FlatHeightScan
    from low_level_controller import LowLevelCommand, LowLevelController, ZeroTorqueController


class MissingMuJoCoError(RuntimeError):
    pass


class MujocoG1Env:
    """Thin MuJoCo wrapper for G1 sim2sim smoke tests."""

    def __init__(
        self,
        xml_path: str | Path,
        controller: LowLevelController | None = None,
        height_scan: FlatHeightScan | None = None,
        physics_dt: float | None = None,
    ):
        try:
            import mujoco
        except ImportError as exc:
            raise MissingMuJoCoError("Python package `mujoco` is not installed in this environment.") from exc

        self.mujoco = mujoco
        self.xml_path = Path(xml_path)
        if not self.xml_path.exists():
            raise FileNotFoundError(f"G1 MuJoCo XML not found: {self.xml_path}")

        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        if physics_dt is not None:
            self.model.opt.timestep = physics_dt
        self.data = mujoco.MjData(self.model)
        self.controller = controller or ZeroTorqueController()
        self.height_scan = height_scan or FlatHeightScan()

    @property
    def physics_dt(self) -> float:
        return float(self.model.opt.timestep)

    def reset(self) -> None:
        self.mujoco.mj_resetData(self.model, self.data)
        self.controller.reset(self.model, self.data)
        self.mujoco.mj_forward(self.model, self.data)

    def step(self, command: LowLevelCommand, decimation: int = 1) -> None:
        for _ in range(decimation):
            ctrl = self.controller.compute_ctrl(self.model, self.data, command)
            if ctrl.shape[0] != self.model.nu:
                raise ValueError(f"Controller returned ctrl shape {ctrl.shape}, expected ({self.model.nu},).")
            self.data.ctrl[:] = ctrl
            self.mujoco.mj_step(self.model, self.data)

    def base_xy_yaw(self) -> np.ndarray:
        if self.model.nq < 7:
            return np.zeros(3, dtype=np.float32)
        x = float(self.data.qpos[0])
        y = float(self.data.qpos[1])
        quat = np.asarray(self.data.qpos[3:7], dtype=np.float64)
        yaw = self._yaw_from_wxyz(quat)
        return np.asarray([x, y, yaw], dtype=np.float32)

    def base_xyz_rpy(self) -> np.ndarray:
        if self.model.nq < 7:
            return np.zeros(6, dtype=np.float32)
        xyz = np.asarray(self.data.qpos[0:3], dtype=np.float64)
        quat = np.asarray(self.data.qpos[3:7], dtype=np.float64)
        rpy = self._rpy_from_wxyz(quat)
        return np.concatenate([xyz, rpy], axis=0).astype(np.float32)

    def observe_height_scan(self) -> np.ndarray:
        return self.height_scan.observe(self.model, self.data)

    def observe_fdm_state(self) -> np.ndarray:
        if self.model.nq < 7:
            return np.zeros(8, dtype=np.float32)
        pose = self.base_xyz_rpy()
        yaw = float(pose[5])
        energy = float(np.sum(np.square(self.data.ctrl)) * 0.001) if self.model.nu else 0.0
        state = np.asarray(
            [
                0.0,
                0.0,
                np.sin(yaw),
                np.cos(yaw),
                0.0,
                energy,
                1.0,
                1.0,
            ],
            dtype=np.float32,
        )
        return state

    def observe_fdm_proprioception(self, command: LowLevelCommand) -> np.ndarray:
        if hasattr(self.controller, "fdm_proprioception"):
            proprio = self.controller.fdm_proprioception(self.model, self.data, command)
            return np.asarray(proprio, dtype=np.float32)
        return np.zeros(157, dtype=np.float32)

    def illegal_contact(self) -> bool:
        """Return true for terrain contact on non-foot robot bodies."""
        return bool(self.illegal_contact_pairs())

    def illegal_contact_pairs(self) -> list[tuple[str, str]]:
        """Return non-support robot body contacts against terrain/obstacle geoms."""
        pairs: list[tuple[str, str]] = []
        for contact_idx in range(self.data.ncon):
            contact = self.data.contact[contact_idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            pair = self._illegal_robot_terrain_contact_pair(geom1, geom2)
            if pair is not None:
                pairs.append(pair)
                continue
            pair = self._illegal_robot_terrain_contact_pair(geom2, geom1)
            if pair is not None:
                pairs.append(pair)
        return pairs

    def _illegal_robot_terrain_contact_pair(
        self,
        robot_geom_id: int,
        terrain_geom_id: int,
    ) -> tuple[str, str] | None:
        robot_body = self._geom_body_name(robot_geom_id)
        terrain_geom = self._geom_name(terrain_geom_id).lower()
        if not robot_body or not terrain_geom:
            return None
        if not self._is_robot_body(robot_body):
            return None
        if not self._is_terrain_geom(terrain_geom):
            return None
        if self._is_allowed_support_body(robot_body):
            return None
        return robot_body, terrain_geom

    def _geom_name(self, geom_id: int) -> str:
        if geom_id < 0:
            return ""
        return self.model.geom(geom_id).name or ""

    def _geom_body_name(self, geom_id: int) -> str:
        if geom_id < 0:
            return ""
        body_id = int(self.model.geom_bodyid[geom_id])
        return self.model.body(body_id).name or ""

    @staticmethod
    def _is_robot_body(body_name: str) -> bool:
        if body_name in {"world", ""}:
            return False
        return body_name.startswith(
            (
                "pelvis",
                "torso",
                "waist",
                "left_",
                "right_",
            )
        )

    @staticmethod
    def _is_allowed_support_body(body_name: str) -> bool:
        lowered = body_name.lower()
        support_tokens = (
            "ankle",
            "foot",
            "sole",
            "toe",
            "heel",
        )
        return any(token in lowered for token in support_tokens)

    @staticmethod
    def _is_terrain_geom(geom_name: str) -> bool:
        return any(token in geom_name for token in ("floor", "ground", "terrain", "fdm_"))

    @staticmethod
    def _yaw_from_wxyz(quat: np.ndarray) -> float:
        return float(MujocoG1Env._rpy_from_wxyz(quat)[2])

    @staticmethod
    def _rpy_from_wxyz(quat: np.ndarray) -> np.ndarray:
        w, x, y, z = quat
        norm = max(float(np.linalg.norm(quat)), 1e-8)
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
        roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch_arg = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
        pitch = np.arcsin(pitch_arg)
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return np.asarray([roll, pitch, yaw], dtype=np.float64)
