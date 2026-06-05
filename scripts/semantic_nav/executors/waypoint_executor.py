from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, sin

from maps.semantic_graph import Pose2D
from executors.robot_adapter import VelocityCommand
from planners.execution_plan import ExecutionStep


@dataclass(frozen=True)
class WaypointExecutorConfig:
    xy_tolerance: float = 0.35
    yaw_tolerance: float = 0.35
    require_yaw_alignment: bool = False
    max_vx: float = 0.45
    max_vy: float = 0.08
    max_wz: float = 0.66
    k_vx: float = 0.8
    k_vy: float = 0.5
    k_wz: float = 1.2
    slow_radius: float = 1.0


@dataclass(frozen=True)
class ExecutorStatus:
    done: bool
    active_step_index: int
    active_step: ExecutionStep | None
    event: str | None = None


class WaypointExecutor:
    """Consume semantic execution steps and emit base-frame velocity commands."""

    def __init__(self, steps: list[ExecutionStep], cfg: WaypointExecutorConfig | None = None) -> None:
        self.steps = steps
        self.cfg = cfg or WaypointExecutorConfig()
        self.step_index = 0

    def reset(self) -> None:
        self.step_index = 0

    def current_step(self) -> ExecutionStep | None:
        if self.step_index >= len(self.steps):
            return None
        return self.steps[self.step_index]

    def update(self, robot_pose: Pose2D) -> tuple[VelocityCommand, ExecutorStatus]:
        step = self.current_step()
        if step is None:
            return VelocityCommand(0.0, 0.0, 0.0), ExecutorStatus(True, self.step_index, None)

        if step.kind == "floor_transition":
            self.step_index += 1
            return (
                VelocityCommand(0.0, 0.0, 0.0),
                ExecutorStatus(False, self.step_index, step, event=step.description),
            )

        if self._reached(robot_pose, step.pose):
            self.step_index += 1
            return (
                VelocityCommand(0.0, 0.0, 0.0),
                ExecutorStatus(self.step_index >= len(self.steps), self.step_index, step, event=f"reached {step.node_id}"),
            )

        command = self._tracking_command(robot_pose, step.pose)
        return command, ExecutorStatus(False, self.step_index, step)

    def apply_transition_if_needed(self, robot_pose: Pose2D) -> Pose2D:
        step = self.current_step()
        if step is None or step.kind != "floor_transition" or step.dst_node_id is None:
            return robot_pose
        return step.pose

    def _reached(self, robot_pose: Pose2D, target_pose: Pose2D) -> bool:
        dx = target_pose.x - robot_pose.x
        dy = target_pose.y - robot_pose.y
        if hypot(dx, dy) > self.cfg.xy_tolerance:
            return False
        if not self.cfg.require_yaw_alignment:
            return True
        yaw_err = abs(_wrap_to_pi(target_pose.yaw - robot_pose.yaw))
        return yaw_err <= self.cfg.yaw_tolerance

    def _tracking_command(self, robot_pose: Pose2D, target_pose: Pose2D) -> VelocityCommand:
        dx_world = target_pose.x - robot_pose.x
        dy_world = target_pose.y - robot_pose.y
        c = cos(robot_pose.yaw)
        s = sin(robot_pose.yaw)
        dx_body = c * dx_world + s * dy_world
        dy_body = -s * dx_world + c * dy_world
        distance = hypot(dx_world, dy_world)
        speed_scale = min(1.0, max(0.2, distance / max(self.cfg.slow_radius, 1e-6)))

        heading_to_target = atan2(dy_world, dx_world)
        heading_err = _wrap_to_pi(heading_to_target - robot_pose.yaw)

        vx = _clamp(self.cfg.k_vx * dx_body * speed_scale, -self.cfg.max_vx, self.cfg.max_vx)
        vy = _clamp(self.cfg.k_vy * dy_body * speed_scale, -self.cfg.max_vy, self.cfg.max_vy)
        wz = _clamp(self.cfg.k_wz * heading_err, -self.cfg.max_wz, self.cfg.max_wz)
        return VelocityCommand(vx=vx, vy=vy, wz=wz)


def advance_abstract_pose(pose: Pose2D, command: VelocityCommand, dt: float) -> Pose2D:
    """Simple unicycle-ish pose update for smoke testing the waypoint executor."""

    c = cos(pose.yaw)
    s = sin(pose.yaw)
    vx_world = c * command.vx - s * command.vy
    vy_world = s * command.vx + c * command.vy
    return Pose2D(
        x=pose.x + vx_world * dt,
        y=pose.y + vy_world * dt,
        yaw=_wrap_to_pi(pose.yaw + command.wz * dt),
    )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _wrap_to_pi(angle: float) -> float:
    while angle > 3.141592653589793:
        angle -= 6.283185307179586
    while angle < -3.141592653589793:
        angle += 6.283185307179586
    return angle
