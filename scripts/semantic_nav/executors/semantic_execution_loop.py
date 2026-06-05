from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from executors.robot_adapter import ExecutionLoopResult, RobotNavAdapter
from executors.waypoint_executor import WaypointExecutor
from maps.semantic_graph import SemanticGraph


@dataclass(frozen=True)
class PerceptionHookResult:
    event: str
    selected_node_ids: tuple[str, ...] = ()


PerceptionHook = Callable[[int, str], str | PerceptionHookResult | None]


def run_semantic_execution_loop(
    *,
    graph: SemanticGraph,
    robot: RobotNavAdapter,
    executor: WaypointExecutor | Any,
    max_steps: int,
    print_every: int = 50,
    active_floor: str,
    perception_hook: PerceptionHook | None = None,
    perception_every: int = 0,
    stop_on_detected_node: str | None = None,
    step_hook: Callable[[int, Any], None] | None = None,
) -> ExecutionLoopResult:
    perception_events: list[str] = []
    confirmed_nodes: set[str] = set()
    for step_idx in range(max_steps):
        command, status = executor.update(robot.pose())
        event = status.event or ""

        if status.event and status.event.startswith("floor transition") and status.active_step is not None:
            dst = status.active_step.dst_node_id
            if dst is not None:
                dst_node = graph.nodes[dst]
                robot.teleport_xy(dst_node.pose.x, dst_node.pose.y)
                active_floor = dst_node.floor
                event = f"{status.event}; teleported_to={dst}"
        else:
            robot.step_velocity(command)

        pose = robot.pose()
        illegal_contact = robot.illegal_contact()
        if perception_hook is not None and perception_every > 0 and step_idx > 0 and step_idx % perception_every == 0:
            hook_event = perception_hook(step_idx, active_floor)
            if hook_event:
                if isinstance(hook_event, PerceptionHookResult):
                    hook_event_text = hook_event.event
                    confirmed_nodes.update(hook_event.selected_node_ids)
                else:
                    hook_event_text = hook_event
                perception_events.append(f"step={step_idx} {hook_event_text}")
                event = f"{event}; {hook_event_text}" if event else hook_event_text
                if stop_on_detected_node is not None and stop_on_detected_node in confirmed_nodes:
                    return ExecutionLoopResult(
                        success=True,
                        steps=step_idx,
                        final_pose=pose,
                        final_step=executor.current_step(),
                        reason=f"detected semantic node {stop_on_detected_node}",
                        perception_events=tuple(perception_events),
                        confirmed_nodes=tuple(sorted(confirmed_nodes)),
                    )
        if print_every > 0 and (step_idx % print_every == 0 or event):
            active_step = status.active_step
            active_name = active_step.node_id if active_step is not None else "done"
            print(
                f"[semantic_nav:exec] step={step_idx} floor={active_floor} active={active_name} "
                f"pose=({pose.x:.3f}, {pose.y:.3f}, {pose.yaw:.3f}) "
                f"cmd=({command.vx:.3f}, {command.vy:.3f}, {command.wz:.3f}) "
                f"event={event or '-'} illegal_contact={illegal_contact}"
            )

        if step_hook is not None:
            step_hook(step_idx, pose)

        if status.done:
            return ExecutionLoopResult(
                success=True,
                steps=step_idx,
                final_pose=pose,
                final_step=status.active_step,
                reason="reached final target",
                perception_events=tuple(perception_events),
                confirmed_nodes=tuple(sorted(confirmed_nodes)),
            )

    return ExecutionLoopResult(
        success=False,
        steps=max_steps,
        final_pose=robot.pose(),
        final_step=executor.current_step(),
        reason="timeout",
        perception_events=tuple(perception_events),
        confirmed_nodes=tuple(sorted(confirmed_nodes)),
    )
