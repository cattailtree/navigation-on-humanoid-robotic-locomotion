from __future__ import annotations

from dataclasses import dataclass

from maps.semantic_graph import GraphPath, SemanticGraph
from planners.elevator_planner import ElevatorPlan, ElevatorPlanner
from planners.goal_parser import NavigationGoal


@dataclass(frozen=True)
class ElevatorTaskPlan:
    goal: NavigationGoal
    elevator_plan: ElevatorPlan
    post_transition_path: GraphPath
    full_node_sequence: list[str]

    @property
    def transition_event(self) -> tuple[str, str] | None:
        transition_target = self.elevator_plan.transition_node_id
        if transition_target is None:
            return None
        return self.elevator_plan.elevator_node_id, transition_target


class ElevatorTaskPlanner:
    """Plan a semantic floor-transition task followed by a target-floor local goal."""

    def __init__(self, graph: SemanticGraph) -> None:
        self.graph = graph
        self.elevator_planner = ElevatorPlanner(graph)

    def plan_downstairs_task(self, start_node_id: str, goal: NavigationGoal, target_node_id: str) -> ElevatorTaskPlan:
        elevator_plan = self.elevator_planner.plan_to_elevator(start_node_id, goal)
        if elevator_plan.transition_node_id is None:
            raise RuntimeError(
                f"Elevator {elevator_plan.elevator_node_id} has no transition to target floor {goal.target_floor}"
            )

        post_transition_path = self.graph.shortest_path(
            start=elevator_plan.transition_node_id,
            goal_fn=lambda node: node.node_id == target_node_id,
            edge_filter=lambda edge: edge.kind == "walk",
        )
        if post_transition_path.is_empty:
            raise RuntimeError(
                f"No target-floor walking path from {elevator_plan.transition_node_id} to {target_node_id}"
            )

        full_node_sequence = list(elevator_plan.approach_path.node_ids)
        if full_node_sequence[-1] != elevator_plan.transition_node_id:
            full_node_sequence.append(elevator_plan.transition_node_id)
        full_node_sequence.extend(post_transition_path.node_ids[1:])

        return ElevatorTaskPlan(
            goal=goal,
            elevator_plan=elevator_plan,
            post_transition_path=post_transition_path,
            full_node_sequence=full_node_sequence,
        )

