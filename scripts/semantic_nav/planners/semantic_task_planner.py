from __future__ import annotations

from dataclasses import dataclass

from llm.task_parser import NavigationTaskParser, RuleBasedTaskParser
from maps.semantic_graph import GraphPath, SemanticGraph
from perception.semantic_detector import SemanticDetector
from planners.elevator_planner import ElevatorPlan, ElevatorPlanner
from planners.goal_parser import NavigationGoal


@dataclass(frozen=True)
class SemanticTaskPlan:
    goal: NavigationGoal
    full_node_sequence: list[str]
    paths: list[GraphPath]
    elevator_plan: ElevatorPlan | None = None

    @property
    def transition_event(self) -> tuple[str, str] | None:
        if self.elevator_plan is None:
            return None
        transition_target = self.elevator_plan.transition_node_id
        if transition_target is None:
            return None
        return self.elevator_plan.elevator_node_id, transition_target


class SemanticTaskPlanner:
    """General semantic planner for local and floor-transition tasks."""

    def __init__(
        self,
        graph: SemanticGraph,
        detector: SemanticDetector | None = None,
        goal_parser: NavigationTaskParser | None = None,
    ) -> None:
        self.graph = graph
        self.goal_parser = goal_parser or RuleBasedTaskParser()
        self.elevator_planner = ElevatorPlanner(graph, detector=detector)

    def plan(self, start_node_id: str, goal_text: str, target_node_id: str | None = None) -> SemanticTaskPlan:
        start_node = self.graph.nodes[start_node_id]
        parsed = self.goal_parser.parse(
            goal_text,
            current_floor=start_node.floor,
            graph=self.graph,
            start_node_id=start_node_id,
        )
        if isinstance(parsed, NavigationGoal):
            goal = parsed
        else:
            goal = parsed.goal
            if target_node_id is None:
                target_node_id = parsed.target_node_id

        if goal.intent == "find_elevator":
            elevator_plan = self.elevator_planner.plan_to_elevator(start_node_id, goal)
            return SemanticTaskPlan(
                goal=goal,
                full_node_sequence=elevator_plan.approach_path.node_ids,
                paths=[elevator_plan.approach_path],
                elevator_plan=elevator_plan,
            )

        if target_node_id is None:
            if goal.intent == "floor_transition":
                elevator_plan = self.elevator_planner.plan_to_elevator(start_node_id, goal)
                return SemanticTaskPlan(
                    goal=goal,
                    full_node_sequence=elevator_plan.approach_path.node_ids,
                    paths=[elevator_plan.approach_path],
                    elevator_plan=elevator_plan,
                )
            raise ValueError("target_node_id is required for local goal tasks")

        target_node = self.graph.nodes[target_node_id]
        if start_node.floor == target_node.floor and goal.intent != "floor_transition":
            return self._plan_local_goal(start_node_id, target_node_id, goal)

        return self._plan_floor_transition(start_node_id, target_node_id, goal)

    def _plan_local_goal(self, start_node_id: str, target_node_id: str, goal: NavigationGoal) -> SemanticTaskPlan:
        path = self.graph.shortest_path(
            start=start_node_id,
            goal_fn=lambda node: node.node_id == target_node_id,
            edge_filter=lambda edge: edge.kind == "walk",
        )
        if path.is_empty:
            raise RuntimeError(f"No walking path from {start_node_id} to {target_node_id}")
        return SemanticTaskPlan(goal=goal, full_node_sequence=path.node_ids, paths=[path])

    def _plan_floor_transition(self, start_node_id: str, target_node_id: str, goal: NavigationGoal) -> SemanticTaskPlan:
        target_node = self.graph.nodes[target_node_id]
        if goal.target_floor is None:
            goal = NavigationGoal(
                raw_text=goal.raw_text,
                intent="floor_transition",
                target_floor=target_node.floor,
                target_label=goal.target_label,
            )

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

        return SemanticTaskPlan(
            goal=goal,
            full_node_sequence=full_node_sequence,
            paths=[elevator_plan.approach_path, post_transition_path],
            elevator_plan=elevator_plan,
        )
