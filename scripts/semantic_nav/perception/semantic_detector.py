from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from maps.semantic_graph import SemanticGraph, SemanticNode
from perception.detection_types import BoundingBox2D


@dataclass(frozen=True)
class SemanticDetection:
    node_id: str | None
    label: str
    score: float
    bbox: BoundingBox2D | None = None


class SemanticDetector(Protocol):
    def detect(self, graph: SemanticGraph, *, current_node_id: str | None = None) -> list[SemanticDetection]:
        ...


class GraphSemanticDetector:
    """Oracle-style semantic detector backed by graph node labels/kinds.

    This is the bridge used before real image observations are wired in. It lets
    the planner consume detector outputs instead of directly querying node kind.
    """

    def __init__(self, target_labels: tuple[str, ...] = ("elevator", "lift", "电梯")) -> None:
        self.target_labels = tuple(label.lower() for label in target_labels)

    def detect(self, graph: SemanticGraph, *, current_node_id: str | None = None) -> list[SemanticDetection]:
        detections: list[SemanticDetection] = []
        current_floor = graph.nodes[current_node_id].floor if current_node_id is not None else None
        for node in graph.nodes.values():
            if current_floor is not None and node.floor != current_floor:
                continue
            score = self._score_node(node)
            if score <= 0.0:
                continue
            detections.append(SemanticDetection(node_id=node.node_id, label=node.label or node.kind, score=score))
        detections.sort(key=lambda item: item.score, reverse=True)
        return detections

    def _score_node(self, node: SemanticNode) -> float:
        kind = node.kind.lower()
        label = node.label.lower()
        if kind == "elevator_lobby":
            return 1.0
        if any(target in label for target in self.target_labels):
            return 0.9
        return 0.0
