from __future__ import annotations

from dataclasses import dataclass

from maps.semantic_graph import GraphPath, SemanticGraph, SemanticNode
from perception.semantic_detector import GraphSemanticDetector, SemanticDetector
from planners.astar_planner import SemanticAStarPlanner
from planners.goal_parser import NavigationGoal


@dataclass(frozen=True)
class ElevatorPlan:
    goal: NavigationGoal
    elevator_node_id: str
    approach_path: GraphPath
    transition_node_id: str | None
    subgoal_node_ids: list[str]


class ElevatorPlanner:
    """Find the best elevator lobby for a floor-transition task."""

    def __init__(self, graph: SemanticGraph, detector: SemanticDetector | None = None) -> None:
        self.graph = graph
        self.detector = detector or GraphSemanticDetector()
        self.astar = SemanticAStarPlanner(graph)

    def plan_to_elevator(self, start_node_id: str, goal: NavigationGoal) -> ElevatorPlan:
        if goal.intent not in ("floor_transition", "find_elevator"):
            raise ValueError(f"ElevatorPlanner cannot handle intent: {goal.intent}")

        detections = self.detector.detect(self.graph, current_node_id=start_node_id)
        candidate_node_ids = [
            detection.node_id
            for detection in detections
            if self._is_valid_elevator_for_goal(self.graph.nodes[detection.node_id], self.graph.nodes[start_node_id].floor, goal.target_floor)
        ]
        approach_path = self.astar.plan_to_any(
            start=start_node_id,
            goal_node_ids=candidate_node_ids,
            edge_filter=lambda edge: edge.kind == "walk",
        )
        if approach_path.is_empty:
            raise RuntimeError(f"No detected reachable elevator lobby from {start_node_id}")

        elevator_node_id = approach_path.node_ids[-1]
        transition_node_id = self._find_transition_target(elevator_node_id, goal.target_floor)
        subgoals = list(approach_path.node_ids[1:])
        return ElevatorPlan(
            goal=goal,
            elevator_node_id=elevator_node_id,
            approach_path=approach_path,
            transition_node_id=transition_node_id,
            subgoal_node_ids=subgoals,
        )

    def _is_current_floor_elevator(self, node: SemanticNode, current_floor: str) -> bool:
        if node.floor != current_floor:
            return False
        if node.kind == "elevator_lobby":
            return True
        return any(edge.kind == "elevator_transition" for edge in self.graph.outgoing_edges(node.node_id))

    def _is_valid_elevator_for_goal(self, node: SemanticNode, current_floor: str, target_floor: str | None) -> bool:
        if not self._is_current_floor_elevator(node, current_floor):
            return False
        if target_floor is None:
            return True
        return self._find_transition_target(node.node_id, target_floor) is not None

    def _find_transition_target(self, elevator_node_id: str, target_floor: str | None) -> str | None:
        if target_floor is None:
            return None
        for edge in self.graph.outgoing_edges(elevator_node_id):
            if edge.kind != "elevator_transition":
                continue
            if self.graph.nodes[edge.dst].floor == target_floor:
                return edge.dst
        return None
