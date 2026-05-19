from __future__ import annotations

from collections import Counter

import numpy as np


class FlatHeightScan:
    """Flat-terrain placeholder for the exteroceptive height scan."""

    def __init__(self, shape: tuple[int, int] = (60, 46), resolution: float = 0.1):
        self.shape = shape
        self.resolution = resolution
        self._last_debug: dict[str, float | int | str] = {
            "height_hit_count": 0,
            "height_fdm_hit_count": 0,
            "height_fdm_geom_count": 0,
            "height_fdm_x_min": float("nan"),
            "height_fdm_x_max": float("nan"),
            "height_fdm_y_min": float("nan"),
            "height_fdm_y_max": float("nan"),
            "height_top_geoms": "",
        }

    def observe(self, model, data) -> np.ndarray:
        return np.zeros(self.shape, dtype=np.float32)

    def debug_info(self) -> dict[str, float | int | str]:
        return self._last_debug.copy()


class RaycastHeightScan:
    """MuJoCo downward-ray height scan aligned with the Lab FDM env_sensor grid."""

    def __init__(
        self,
        shape: tuple[int, int] = (60, 46),
        resolution: float = 0.1,
        x_offset: float = 0.0,
        y_offset: float = 0.0,
        z_start: float = 0.5,
        max_distance: float = 5.0,
        terrain_geom_names: tuple[str, ...] = ("floor", "ground", "groundplane", "terrain", "fdm_"),
        exclude_body_name: str = "pelvis",
    ):
        self.shape = shape
        self.resolution = resolution
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.z_start = z_start
        self.max_distance = max_distance
        self.terrain_geom_names = terrain_geom_names
        self.exclude_body_name = exclude_body_name
        rows = np.arange(shape[0], dtype=np.float64)
        cols = np.arange(shape[1], dtype=np.float64)
        # Lab's GridPattern is centered on the pelvis sensor. height_scan_square_fdm
        # flips the row axis, so positive local y maps toward smaller row indices.
        local_y = y_offset - ((rows - shape[0] / 2.0) * resolution)
        local_x = x_offset + (cols - shape[1] / 2.0) * resolution
        self.grid_y, self.grid_x = np.meshgrid(local_y, local_x, indexing="ij")
        self._geom_id = np.zeros(1, dtype=np.int32)
        self._geom_group = np.ones(6, dtype=np.uint8)
        self._exclude_body_id: int | None = None
        self._last_debug: dict[str, float | int | str] = {}

    def observe(self, model, data) -> np.ndarray:
        mujoco = self._mujoco()
        if data.qpos.shape[0] < 7:
            return np.zeros(self.shape, dtype=np.float32)

        base_xy = np.asarray(data.qpos[0:2], dtype=np.float64)
        base_z = float(data.qpos[2])
        quat = np.asarray(data.qpos[3:7], dtype=np.float64)
        yaw = self._yaw_from_wxyz(quat)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        heights = np.full(self.shape, -base_z, dtype=np.float32)
        bodyexclude = self._body_id(model, self.exclude_body_name)
        hit_counts: Counter[str] = Counter()
        fdm_x: list[float] = []
        fdm_y: list[float] = []

        for idx in np.ndindex(self.shape):
            local_x = self.grid_x[idx]
            local_y = self.grid_y[idx]
            world_x = base_xy[0] + cos_yaw * local_x - sin_yaw * local_y
            world_y = base_xy[1] + sin_yaw * local_x + cos_yaw * local_y
            start = np.asarray([world_x, world_y, base_z + self.z_start], dtype=np.float64)
            direction = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
            self._geom_id[0] = -1
            dist = mujoco.mj_ray(
                model,
                data,
                start,
                direction,
                self._geom_group,
                1,
                bodyexclude,
                self._geom_id,
            )
            if dist >= 0.0 and dist <= self.max_distance:
                geom_name = self._geom_name(model, int(self._geom_id[0]))
                if self._is_terrain_geom(geom_name):
                    hit_counts[geom_name] += 1
                    if geom_name.lower().startswith("fdm_"):
                        fdm_x.append(float(local_x))
                        fdm_y.append(float(local_y))
                    hit_z = start[2] - dist
                    heights[idx] = np.float32(hit_z - base_z)
        self._last_debug = self._make_debug_info(hit_counts, fdm_x, fdm_y)
        return heights

    def debug_info(self) -> dict[str, float | int | str]:
        return self._last_debug.copy()

    @staticmethod
    def _mujoco():
        import mujoco

        return mujoco

    @staticmethod
    def _yaw_from_wxyz(quat: np.ndarray) -> float:
        w, x, y, z = quat
        norm = max(float(np.linalg.norm(quat)), 1e-8)
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
        return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    @staticmethod
    def _geom_name(model, geom_id: int) -> str:
        if geom_id < 0:
            return ""
        name = model.geom(geom_id).name
        return name or ""

    def _is_terrain_geom(self, geom_name: str) -> bool:
        lowered = geom_name.lower()
        return any(token in lowered for token in self.terrain_geom_names)

    @staticmethod
    def _make_debug_info(
        hit_counts: Counter[str],
        fdm_x: list[float],
        fdm_y: list[float],
    ) -> dict[str, float | int | str]:
        top_geoms = ";".join(f"{name}:{count}" for name, count in hit_counts.most_common(6))
        fdm_geom_count = sum(1 for name in hit_counts if name.lower().startswith("fdm_"))
        if fdm_x:
            x_min = float(np.min(fdm_x))
            x_max = float(np.max(fdm_x))
            y_min = float(np.min(fdm_y))
            y_max = float(np.max(fdm_y))
        else:
            x_min = x_max = y_min = y_max = float("nan")
        return {
            "height_hit_count": int(sum(hit_counts.values())),
            "height_fdm_hit_count": int(len(fdm_x)),
            "height_fdm_geom_count": int(fdm_geom_count),
            "height_fdm_x_min": x_min,
            "height_fdm_x_max": x_max,
            "height_fdm_y_min": y_min,
            "height_fdm_y_max": y_max,
            "height_top_geoms": top_geoms,
        }

    def _body_id(self, model, body_name: str) -> int:
        if self._exclude_body_id is not None:
            return self._exclude_body_id
        self._exclude_body_id = -1
        for idx in range(model.nbody):
            if model.body(idx).name == body_name:
                self._exclude_body_id = idx
                break
        return self._exclude_body_id
