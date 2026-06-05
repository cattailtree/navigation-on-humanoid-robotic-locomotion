from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from maps.semantic_graph import Pose2D
from planners.execution_plan import ExecutionStep


@dataclass(frozen=True)
class VelocityCommand:
    vx: float
    vy: float
    wz: float


class RobotNavAdapter(Protocol):
    """Backend interface consumed by semantic navigation executors."""

    def reset(self, start_pose: Pose2D) -> None:
        ...

    def pose(self) -> Pose2D:
        ...

    def step_velocity(self, command: VelocityCommand) -> None:
        ...

    def teleport(self, pose: Pose2D) -> None:
        ...

    def teleport_xy(self, x: float, y: float) -> None:
        ...

    def illegal_contact(self) -> bool:
        ...


@dataclass(frozen=True)
class ExecutionLoopResult:
    success: bool
    steps: int
    final_pose: Pose2D
    final_step: ExecutionStep | None
    reason: str
    perception_events: tuple[str, ...] = ()
    confirmed_nodes: tuple[str, ...] = ()
