from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


G1_29DOF_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]


G1_DEFAULT_JOINT_POS = {
    "left_hip_pitch_joint": -0.1,
    "right_hip_pitch_joint": -0.1,
    "left_knee_joint": 0.3,
    "right_knee_joint": 0.3,
    "left_ankle_pitch_joint": -0.2,
    "right_ankle_pitch_joint": -0.2,
    "left_shoulder_pitch_joint": 0.3,
    "right_shoulder_pitch_joint": 0.3,
    "left_shoulder_roll_joint": 0.25,
    "right_shoulder_roll_joint": -0.25,
    "left_elbow_joint": 0.97,
    "right_elbow_joint": 0.97,
    "left_wrist_roll_joint": 0.15,
    "right_wrist_roll_joint": -0.15,
}


MUJOCO_TO_POLICY_JOINT_MAP = {
    "torso_joint": "waist_yaw_joint",
    "left_elbow_pitch_joint": "left_elbow_joint",
    "right_elbow_pitch_joint": "right_elbow_joint",
}


G1_STIFFNESS = {
    "hip_pitch": 100.0,
    "hip_roll": 100.0,
    "hip_yaw": 100.0,
    "knee": 150.0,
    "ankle": 40.0,
    "shoulder": 40.0,
    "elbow": 40.0,
    "wrist": 40.0,
    "waist_yaw": 200.0,
    "waist": 40.0,
}


G1_DAMPING = {
    "hip": 2.0,
    "knee": 4.0,
    "ankle": 2.0,
    "shoulder": 1.0,
    "elbow": 1.0,
    "wrist": 1.0,
    "waist": 5.0,
}


@dataclass
class LowLevelCommand:
    """Base-frame velocity command expected from the high-level planner."""

    vx: float
    vy: float
    wz: float

    @classmethod
    def zeros(cls) -> "LowLevelCommand":
        return cls(vx=0.0, vy=0.0, wz=0.0)

    def as_array(self) -> np.ndarray:
        return np.asarray([self.vx, self.vy, self.wz], dtype=np.float32)


class LowLevelController:
    """Interface for a MuJoCo G1 controller that tracks `[vx, vy, wz]`."""

    def reset(self, model, data) -> None:
        pass

    def compute_ctrl(self, model, data, command: LowLevelCommand) -> np.ndarray:
        raise NotImplementedError


class ZeroTorqueController(LowLevelController):
    """Safe placeholder. It does not walk; it only lets MuJoCo step with zero ctrl."""

    def compute_ctrl(self, model, data, command: LowLevelCommand) -> np.ndarray:
        return np.zeros(model.nu, dtype=np.float64)


class MissingGaitController(LowLevelController):
    """Explicit placeholder for the missing G1 velocity-tracking policy."""

    def compute_ctrl(self, model, data, command: LowLevelCommand) -> np.ndarray:
        if np.linalg.norm(command.as_array()) > 1e-6:
            raise RuntimeError(
                "Missing G1 low-level gait/policy. Provide a controller that maps "
                "[vx, vy, wz] to MuJoCo ctrl before running non-zero commands."
            )
        return np.zeros(model.nu, dtype=np.float64)


class TorchPolicyPDController(LowLevelController):
    """TorchScript G1 policy wrapped with MuJoCo torque-motor PD control.

    The IsaacLab action term uses joint-position targets with `scale=0.25` and
    default joint offsets. The provided MuJoCo XML uses torque motors, so this
    controller converts policy actions into position targets and then applies PD
    torques to the matching actuators.
    """

    def __init__(
        self,
        policy_path: str | Path,
        device: str = "cpu",
        joint_names: list[str] | None = None,
        obs_history: int = 5,
        obs_dim: int = 495,
        action_dim: int = 29,
        action_scale: float = 0.25,
        action_clip: float = 10.0,
        obs_layout: str = "g1_policy_99",
        inference_decimation: int = 4,
        obs_axis_transform: str = "identity",
        strict_joints: bool = True,
    ):
        self.policy_path = Path(policy_path)
        self.device = device
        self.joint_names = joint_names or G1_29DOF_JOINT_NAMES
        self.obs_history = obs_history
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.action_clip = action_clip
        self.obs_layout = obs_layout
        self.inference_decimation = max(1, int(inference_decimation))
        self.obs_axis_transform = obs_axis_transform
        self.strict_joints = strict_joints
        self.policy = None
        self.joint_qpos_adr: np.ndarray | None = None
        self.joint_dof_adr: np.ndarray | None = None
        self.actuator_ids: np.ndarray | None = None
        self.default_pos: np.ndarray | None = None
        self.last_action = np.zeros(self.action_dim, dtype=np.float32)
        self.second_last_action = np.zeros(self.action_dim, dtype=np.float32)
        self.obs_history_buffer: list[np.ndarray] = []
        self.kp: np.ndarray | None = None
        self.kd: np.ndarray | None = None
        self.policy_to_mujoco: list[tuple[int, str]] = []
        self._inference_counter = 0
        self._target_pos: np.ndarray | None = None
        self._term_history_buffer: dict[str, list[np.ndarray]] = {}
        self._imu_gyro_sensor: tuple[int, int] | None = None
        self._imu_quat_sensor: tuple[int, int] | None = None
        self.last_full_ctrl = np.zeros(0, dtype=np.float32)

    def reset(self, model, data) -> None:
        self._load_policy()
        self._bind_model(model)
        self._write_initial_state(model, data)
        self.last_action = np.zeros(self.action_dim, dtype=np.float32)
        self.second_last_action = np.zeros(self.action_dim, dtype=np.float32)
        self.last_full_ctrl = np.zeros(model.nu, dtype=np.float32)
        self._target_pos = self.default_pos.copy()
        self._inference_counter = 0
        terms = self._build_terms(model, data, LowLevelCommand.zeros())
        self._term_history_buffer = {
            name: [value.astype(np.float32).copy() for _ in range(self.obs_history)] for name, value in terms
        }

    def compute_ctrl(self, model, data, command: LowLevelCommand) -> np.ndarray:
        if self.policy is None or self.joint_qpos_adr is None:
            self.reset(model, data)

        if self._inference_counter % self.inference_decimation == 0 or self._target_pos is None:
            obs = self._build_obs(model, data, command)
            action = self._policy_forward(obs)
            self.second_last_action = self.last_action.copy()
            self.last_action = action.astype(np.float32)
            active_action = np.asarray([action[policy_idx] for policy_idx, _ in self.policy_to_mujoco], dtype=np.float64)
            self._target_pos = self.default_pos + self.action_scale * active_action
            self._inference_counter = 0

        q = np.asarray(data.qpos[self.joint_qpos_adr], dtype=np.float64)
        qd = np.asarray(data.qvel[self.joint_dof_adr], dtype=np.float64)
        torque = self.kp * (self._target_pos - q) - self.kd * qd
        torque = np.nan_to_num(torque, nan=0.0, posinf=0.0, neginf=0.0)

        ctrl = np.zeros(model.nu, dtype=np.float64)
        ctrl[self.actuator_ids] = torque
        if getattr(model, "actuator_ctrlrange", None) is not None:
            lo = model.actuator_ctrlrange[:, 0]
            hi = model.actuator_ctrlrange[:, 1]
            ctrl = np.clip(ctrl, lo, hi)
        ctrl = np.nan_to_num(ctrl, nan=0.0, posinf=0.0, neginf=0.0)
        self.last_full_ctrl = ctrl.astype(np.float32)
        self._inference_counter += 1
        return ctrl

    def fdm_proprioception(self, model, data, command: LowLevelCommand) -> np.ndarray:
        full_q, full_qd, full_default = self._full_joint_state(data)
        torque = np.zeros(self.action_dim, dtype=np.float32)
        if self.actuator_ids is not None and self.last_full_ctrl.shape[0] == model.nu:
            for local_idx, (policy_idx, _) in enumerate(self.policy_to_mujoco):
                torque[policy_idx] = self.last_full_ctrl[self.actuator_ids[local_idx]]
        return np.concatenate(
            [
                command.as_array(),
                self._projected_gravity(data).astype(np.float32),
                self._base_lin_vel_body(data).astype(np.float32),
                self._base_ang_vel_body(data).astype(np.float32),
                torque,
                full_q,
                full_qd,
                self.last_action.astype(np.float32),
                self.second_last_action.astype(np.float32),
            ],
            axis=0,
        ).astype(np.float32)

    def _load_policy(self) -> None:
        if self.policy is not None:
            return
        if not self.policy_path.exists():
            raise FileNotFoundError(f"Low-level policy not found: {self.policy_path}")
        import torch

        self.torch = torch
        self.policy = torch.jit.load(str(self.policy_path), map_location=self.device)
        self.policy.eval()

    def _bind_model(self, model) -> None:
        policy_name_to_idx = {name: idx for idx, name in enumerate(self.joint_names)}
        model_joint_names = [model.joint(idx).name for idx in range(model.njnt)]
        self.policy_to_mujoco = []
        for model_name in model_joint_names:
            if model_name == "floating_base_joint":
                continue
            policy_name = model_name if model_name in policy_name_to_idx else MUJOCO_TO_POLICY_JOINT_MAP.get(model_name)
            if policy_name in policy_name_to_idx:
                self.policy_to_mujoco.append((policy_name_to_idx[policy_name], model_name))

        mapped_policy_names = {self.joint_names[policy_idx] for policy_idx, _ in self.policy_to_mujoco}
        missing = [name for name in self.joint_names if name not in mapped_policy_names]
        if missing and self.strict_joints:
            raise RuntimeError(
                "MuJoCo model does not match the IsaacLab G1 29DOF policy joint list. "
                f"Missing or unmapped policy joints: {missing}. Use a matching 29DOF G1 MJCF, "
                "add mappings, or run with `--allow-partial-policy-joints` for a smoke test."
            )

        if not self.policy_to_mujoco:
            raise RuntimeError("No MuJoCo joints could be mapped to policy outputs.")

        active_names = [model_name for _, model_name in self.policy_to_mujoco]
        self.active_joint_names = active_names
        self.joint_qpos_adr = np.asarray([model.jnt_qposadr[self._joint_id(model, name)] for name in active_names])
        self.joint_dof_adr = np.asarray([model.jnt_dofadr[self._joint_id(model, name)] for name in active_names])
        self.actuator_ids = np.asarray([self._actuator_id_for_joint(model, name) for name in active_names])
        if np.any(self.actuator_ids < 0):
            missing_act = [name for name, act_id in zip(active_names, self.actuator_ids) if act_id < 0]
            raise RuntimeError(f"Missing MuJoCo actuators for policy joints: {missing_act}")

        default = np.zeros(len(active_names), dtype=np.float64)
        for idx, name in enumerate(active_names):
            default[idx] = G1_DEFAULT_JOINT_POS.get(name, 0.0)
        self.default_pos = default
        self.kp = np.asarray([self._gain_for(name, G1_STIFFNESS, 40.0) for name in active_names], dtype=np.float64)
        self.kd = np.asarray([self._gain_for(name, G1_DAMPING, 1.0) for name in active_names], dtype=np.float64)
        self._imu_gyro_sensor = self._sensor_spec(model, "imu_gyro")
        self._imu_quat_sensor = self._sensor_spec(model, "imu_quat")

    def _write_initial_state(self, model, data) -> None:
        if model.nq >= 7:
            data.qpos[0:3] = np.asarray([0.0, 0.0, 0.8], dtype=np.float64)
            data.qpos[3:7] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        data.qpos[self.joint_qpos_adr] = self.default_pos
        data.qvel[:] = 0.0

    def _build_obs(self, model, data, command: LowLevelCommand) -> np.ndarray:
        terms = self._build_terms(model, data, command)
        if not self._term_history_buffer:
            self._term_history_buffer = {
                name: [value.astype(np.float32).copy() for _ in range(self.obs_history)] for name, value in terms
            }

        chunks: list[np.ndarray] = []
        for name, value in terms:
            history = self._term_history_buffer.setdefault(
                name, [value.astype(np.float32).copy() for _ in range(self.obs_history)]
            )
            history.append(value.astype(np.float32))
            del history[:-self.obs_history]
            chunks.append(np.concatenate(history, axis=0))

        obs = np.concatenate(chunks, axis=0).astype(np.float32)
        if obs.shape[0] < self.obs_dim:
            obs = np.pad(obs, (0, self.obs_dim - obs.shape[0]))
        elif obs.shape[0] > self.obs_dim:
            obs = obs[: self.obs_dim]
        return obs

    def _build_terms(self, model, data, command: LowLevelCommand) -> list[tuple[str, np.ndarray]]:
        full_q, full_qd, full_default = self._full_joint_state(data)
        q_rel = full_q - full_default
        lin_vel = self._transform_obs_vector(self._base_lin_vel_body(data)).astype(np.float32)
        ang_vel = self._transform_obs_vector(self._base_ang_vel_body(data)).astype(np.float32) * 0.2
        gravity = self._transform_obs_vector(self._projected_gravity(data)).astype(np.float32)
        cmd = command.as_array()

        frame_dim = self.obs_dim // self.obs_history
        layout = self._resolve_obs_layout(frame_dim)
        if layout == "g1_policy_99":
            return [
                ("base_lin_vel", lin_vel),
                ("base_ang_vel", ang_vel),
                ("projected_gravity", gravity),
                ("velocity_commands", cmd),
                ("joint_pos_rel", q_rel),
                ("joint_vel_rel", full_qd * 0.05),
                ("last_action", self.last_action.astype(np.float32)),
            ]
        return [
            ("base_ang_vel", ang_vel),
            ("projected_gravity", gravity),
            ("velocity_commands", cmd),
            ("joint_pos_rel", q_rel),
            ("joint_vel_rel", full_qd * 0.05),
            ("last_action", self.last_action.astype(np.float32)),
        ]

    def _full_joint_state(self, data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        full_q = np.zeros(self.action_dim, dtype=np.float32)
        full_qd = np.zeros(self.action_dim, dtype=np.float32)
        full_default = np.zeros(self.action_dim, dtype=np.float32)
        if self.joint_qpos_adr is None or self.joint_dof_adr is None:
            return full_q, full_qd, full_default
        active_q = np.asarray(data.qpos[self.joint_qpos_adr], dtype=np.float32)
        active_qd = np.asarray(data.qvel[self.joint_dof_adr], dtype=np.float32)
        for local_idx, (policy_idx, _model_name) in enumerate(self.policy_to_mujoco):
            full_q[policy_idx] = active_q[local_idx]
            full_qd[policy_idx] = active_qd[local_idx]
            full_default[policy_idx] = G1_DEFAULT_JOINT_POS.get(self.joint_names[policy_idx], 0.0)
        return full_q, full_qd, full_default

    def _resolve_obs_layout(self, frame_dim: int) -> str:
        if self.obs_layout != "auto":
            return self.obs_layout
        if frame_dim == 99:
            return "g1_policy_99"
        return "g1_nav_96"

    def _policy_forward(self, obs: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            obs_t = self.torch.from_numpy(obs[None]).to(self.device)
            out = self.policy(obs_t)
            if isinstance(out, (tuple, list)):
                out = out[0]
            action = out.detach().cpu().numpy().reshape(-1)
        if action.shape[0] != self.action_dim:
            if self.strict_joints:
                raise RuntimeError(f"Policy output dim {action.shape[0]} != expected {self.action_dim}.")
            padded = np.zeros(self.action_dim, dtype=np.float64)
            n = min(self.action_dim, action.shape[0])
            padded[:n] = action[:n]
            action = padded
        action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        if self.action_clip > 0.0:
            action = np.clip(action, -self.action_clip, self.action_clip)
        return action.astype(np.float64)

    @staticmethod
    def _joint_id(model, name: str) -> int:
        for idx in range(model.njnt):
            if model.joint(idx).name == name:
                return idx
        return -1

    @staticmethod
    def _actuator_id_for_joint(model, joint_name: str) -> int:
        joint_id = TorchPolicyPDController._joint_id(model, joint_name)
        if joint_id < 0:
            return -1
        actuator_trnid = getattr(model, "actuator_trnid", None)
        if actuator_trnid is not None:
            for idx in range(model.nu):
                if int(actuator_trnid[idx, 0]) == joint_id:
                    return idx

        short_name = joint_name.removesuffix("_joint")
        for idx in range(model.nu):
            actuator_name = model.actuator(idx).name
            if actuator_name in (joint_name, short_name):
                return idx
        return -1

    @staticmethod
    def _gain_for(name: str, table: dict[str, float], default: float) -> float:
        for key, value in table.items():
            if key in name:
                return value
        return default

    def _base_lin_vel_body(self, data) -> np.ndarray:
        if data.qvel.shape[0] < 3:
            return np.zeros(3, dtype=np.float64)
        lin_vel_w = np.asarray(data.qvel[0:3], dtype=np.float64)
        q = self._base_quat_wxyz(data)
        rot = TorchPolicyPDController._quat_wxyz_to_rot(q)
        return rot.T @ lin_vel_w

    def _base_ang_vel_body(self, data) -> np.ndarray:
        gyro = self._sensor_data(data, self._imu_gyro_sensor, 3)
        if gyro is not None:
            return gyro
        if data.qvel.shape[0] < 6:
            return np.zeros(3, dtype=np.float64)
        ang_vel_w = np.asarray(data.qvel[3:6], dtype=np.float64)
        q = self._base_quat_wxyz(data)
        rot = TorchPolicyPDController._quat_wxyz_to_rot(q)
        return rot.T @ ang_vel_w

    def _projected_gravity(self, data) -> np.ndarray:
        # MuJoCo free-joint quaternion is wxyz at qpos[3:7].
        q = self._base_quat_wxyz(data)
        rot = TorchPolicyPDController._quat_wxyz_to_rot(q)
        return rot.T @ np.asarray([0.0, 0.0, -1.0], dtype=np.float64)

    def _base_quat_wxyz(self, data) -> np.ndarray:
        quat = self._sensor_data(data, self._imu_quat_sensor, 4)
        if quat is not None:
            return quat
        if data.qpos.shape[0] < 7:
            return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return np.asarray(data.qpos[3:7], dtype=np.float64)

    def _transform_obs_vector(self, vec: np.ndarray) -> np.ndarray:
        x, y, z = np.asarray(vec, dtype=np.float64)
        if self.obs_axis_transform == "rot_x_pos_90":
            return np.asarray([x, -z, y], dtype=np.float64)
        if self.obs_axis_transform == "rot_x_neg_90":
            return np.asarray([x, z, -y], dtype=np.float64)
        if self.obs_axis_transform == "swap_yz":
            return np.asarray([x, z, y], dtype=np.float64)
        if self.obs_axis_transform == "swap_yz_neg":
            return np.asarray([x, -z, -y], dtype=np.float64)
        return np.asarray([x, y, z], dtype=np.float64)

    @staticmethod
    def _sensor_spec(model, name: str) -> tuple[int, int] | None:
        for idx in range(model.nsensor):
            if model.sensor(idx).name == name:
                return int(model.sensor_adr[idx]), int(model.sensor_dim[idx])
        return None

    @staticmethod
    def _sensor_data(data, sensor: tuple[int, int] | None, dim: int) -> np.ndarray | None:
        if sensor is None:
            return None
        adr, sensor_dim = sensor
        if sensor_dim < dim:
            return None
        return np.asarray(data.sensordata[adr : adr + dim], dtype=np.float64)

    @staticmethod
    def _quat_wxyz_to_rot(q: np.ndarray) -> np.ndarray:
        w, x, y, z = q / max(np.linalg.norm(q), 1e-8)
        return np.asarray(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
