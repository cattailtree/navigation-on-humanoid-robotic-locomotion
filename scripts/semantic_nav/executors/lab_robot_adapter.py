from __future__ import annotations

from math import atan2

import torch

from executors.robot_adapter import RobotNavAdapter, VelocityCommand
from maps.semantic_graph import Pose2D


class LabRobotAdapter(RobotNavAdapter):
    """RobotNavAdapter backed by an Isaac Lab ManagerBasedRLEnv."""

    def __init__(self, env, *, env_id: int = 0) -> None:
        self.env = env
        self.env_id = int(env_id)
        self.device = env.device
        self._env_ids = torch.tensor([self.env_id], device=self.device, dtype=torch.long)
        self._last_episode_length = 0
        self._last_reset_reason = ""

    def reset(self, start_pose: Pose2D) -> None:
        self.env.reset()
        self.teleport(start_pose)
        self.env.step(torch.zeros(self.env.num_envs, 3, device=self.device))
        self._last_episode_length = self.episode_step()

    def pose(self) -> Pose2D:
        robot = self.env.scene.articulations["robot"]
        pos = robot.data.root_pos_w[self.env_id]
        quat = robot.data.root_quat_w[self.env_id]
        yaw = _yaw_from_wxyz(quat)
        origin = self.env.scene.env_origins[self.env_id]
        return Pose2D(
            x=float((pos[0] - origin[0]).item()),
            y=float((pos[1] - origin[1]).item()),
            yaw=float(yaw),
        )

    def step_velocity(self, command: VelocityCommand) -> None:
        actions = torch.zeros(self.env.num_envs, 3, device=self.device)
        actions[self.env_id, 0] = command.vx
        actions[self.env_id, 1] = command.vy
        actions[self.env_id, 2] = command.wz
        with torch.inference_mode():
            self.env.step(actions)

    def episode_step(self) -> int:
        if not hasattr(self.env, "episode_length_buf"):
            return 0
        return int(self.env.episode_length_buf[self.env_id].item())

    def consume_reset_event(self) -> bool:
        current = self.episode_step()
        was_reset = current < self._last_episode_length
        self._last_reset_reason = self._termination_reason() if was_reset else ""
        self._last_episode_length = current
        return was_reset

    def last_reset_reason(self) -> str:
        return self._last_reset_reason

    def teleport(self, pose: Pose2D) -> None:
        robot = self.env.scene.articulations["robot"]
        root_states = robot.data.default_root_state[self._env_ids].detach().clone()
        origin = self.env.scene.env_origins[self._env_ids]
        root_states[:, 0] = origin[:, 0] + pose.x
        root_states[:, 1] = origin[:, 1] + pose.y
        root_states[:, 3:7] = _quat_wxyz_from_yaw(pose.yaw, device=self.device).unsqueeze(0)
        velocities = torch.zeros_like(root_states[:, 7:13])
        with torch.inference_mode():
            robot.write_root_pose_to_sim(root_states[:, :7], env_ids=self._env_ids)
            robot.write_root_velocity_to_sim(velocities, env_ids=self._env_ids)

    def teleport_xy(self, x: float, y: float) -> None:
        current_pose = self.pose()
        self.teleport(Pose2D(x=x, y=y, yaw=current_pose.yaw))

    def illegal_contact(self) -> bool:
        if "contact_forces" not in self.env.scene.sensors:
            return False
        # For the first Lab milestone, termination/done handles serious failures.
        return False

    def _termination_reason(self) -> str:
        manager = getattr(self.env, "termination_manager", None)
        if manager is None:
            return ""
        active = []
        for term_name in getattr(manager, "active_terms", []):
            try:
                term_value = bool(manager.get_term(term_name)[self.env_id].item())
            except Exception:
                term_value = False
            if term_value:
                active.append(term_name)
        if not active:
            return ""
        return "reset_reason=" + ",".join(active)


def _quat_wxyz_from_yaw(yaw: float, *, device: str) -> torch.Tensor:
    half = torch.tensor(yaw * 0.5, device=device)
    return torch.stack(
        [
            torch.cos(half),
            torch.tensor(0.0, device=device),
            torch.tensor(0.0, device=device),
            torch.sin(half),
        ]
    )


def _yaw_from_wxyz(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)
