from __future__ import annotations

from math import cos, sin
from pathlib import Path
import sys

import numpy as np

from executors.robot_adapter import RobotNavAdapter, VelocityCommand
from maps.semantic_graph import Pose2D


SIM2SIM_ROOT = Path(__file__).resolve().parents[2] / "mujoco_sim2sim"
if str(SIM2SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM2SIM_ROOT))

from height_scan import FlatHeightScan  # noqa: E402
from low_level_controller import LowLevelCommand  # noqa: E402
from mujoco_g1_env import MujocoG1Env  # noqa: E402


class Sim2SimRobotAdapter(RobotNavAdapter):
    """RobotNavAdapter backed by the MuJoCo G1 sim2sim environment."""

    def __init__(
        self,
        xml_path: str | Path,
        controller,
        *,
        physics_dt: float | None,
        control_decimation: int,
    ) -> None:
        self.env = MujocoG1Env(
            xml_path=xml_path,
            controller=controller,
            height_scan=FlatHeightScan(),
            physics_dt=physics_dt,
        )
        self.control_decimation = control_decimation

    def reset(self, start_pose: Pose2D) -> None:
        self.env.reset()
        self.teleport(start_pose)

    def pose(self) -> Pose2D:
        x, y, yaw = self.env.base_xy_yaw()
        return Pose2D(float(x), float(y), float(yaw))

    def step_velocity(self, command: VelocityCommand) -> None:
        self.env.step(
            LowLevelCommand(vx=command.vx, vy=command.vy, wz=command.wz),
            decimation=self.control_decimation,
        )

    def teleport(self, pose: Pose2D) -> None:
        if self.env.model.nq < 7:
            return
        self.env.data.qpos[0] = pose.x
        self.env.data.qpos[1] = pose.y
        self.env.data.qpos[3] = cos(pose.yaw * 0.5)
        self.env.data.qpos[4] = 0.0
        self.env.data.qpos[5] = 0.0
        self.env.data.qpos[6] = sin(pose.yaw * 0.5)
        self.env.data.qvel[:] = 0.0
        self.env.mujoco.mj_forward(self.env.model, self.env.data)

    def teleport_xy(self, x: float, y: float) -> None:
        current_pose = self.pose()
        self.teleport(Pose2D(x=x, y=y, yaw=current_pose.yaw))

    def illegal_contact(self) -> bool:
        return self.env.illegal_contact()

    def base_xy_yaw_array(self) -> np.ndarray:
        return self.env.base_xy_yaw()
