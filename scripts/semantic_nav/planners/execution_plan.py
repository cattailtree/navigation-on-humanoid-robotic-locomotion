from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from maps.semantic_graph import Pose2D, SemanticGraph
from planners.semantic_task_planner import SemanticTaskPlan


@dataclass(frozen=True)
class ExecutionStep:
    kind: Literal["walk_to", "floor_transition"]
    node_id: str
    floor: str
    pose: Pose2D
    description: str
    dst_node_id: str | None = None


def build_execution_plan(graph: SemanticGraph, task_plan: SemanticTaskPlan) -> list[ExecutionStep]:
    """Convert a semantic node route into explicit executable high-level steps."""

    steps: list[ExecutionStep] = []
    transition_event = task_plan.transition_event

    for node_id in task_plan.full_node_sequence[1:]:
        node = graph.nodes[node_id]
        if transition_event is not None and node_id == transition_event[1]:
            src_node = graph.nodes[transition_event[0]]
            steps.append(
                ExecutionStep(
                    kind="floor_transition",
                    node_id=transition_event[0],
                    dst_node_id=transition_event[1],
                    floor=src_node.floor,
                    pose=src_node.pose,
                    description=f"floor transition {transition_event[0]} -> {transition_event[1]}",
                )
            )
            continue

        steps.append(
            ExecutionStep(
                kind="walk_to",
                node_id=node.node_id,
                floor=node.floor,
                pose=node.pose,
                description=f"walk to {node.node_id}",
            )
        )

    return steps
