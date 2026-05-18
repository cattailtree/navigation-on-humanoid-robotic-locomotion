from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_ROOT = Path(r"D:\fdm_data\mujoco_sim2sim")


@dataclass
class Sim2SimConfig:
    root: Path = DEFAULT_ROOT
    g1_xml: Path = DEFAULT_ROOT / "assets" / "g1" / "scene_29dof.xml"
    low_level_policy: Path | None = DEFAULT_ROOT / "policies" / "g1_policy.pt"
    fdm_run: str | None = None
    fdm_checkpoint: Path | None = DEFAULT_ROOT / "fdm_checkpoints" / "model_collection_round_14.pth"
    log_dir: Path = DEFAULT_ROOT / "logs"
    physics_dt: float | None = 0.005
    control_decimation: int = 4
    planner_frequency: float = 10.0
    goal_xy_yaw: tuple[float, float, float] = (5.0, 0.0, 0.0)
    # Match the IsaacLab G1/FDM height model: ObsExteroceptiveCfg shape=(60, 46)
    # with env_sensor pattern size=(4.5, 5.9), resolution=0.1.
    height_scan_shape: tuple[int, int] = (60, 46)
    height_scan_resolution: float = 0.1
    device: str = "cuda"

    @property
    def control_dt(self) -> float | None:
        if self.physics_dt is None:
            return None
        return self.physics_dt * self.control_decimation
