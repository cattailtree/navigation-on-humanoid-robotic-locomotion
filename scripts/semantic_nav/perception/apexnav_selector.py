from __future__ import annotations

from dataclasses import dataclass

from maps.semantic_graph import SemanticGraph, SemanticNode
from perception.semantic_detector import SemanticDetection, SemanticDetector


@dataclass(frozen=True)
class ApexNavSelectorConfig:
    target_labels: tuple[str, ...] = ("elevator", "lift")
    likely_target_score: float = 0.65
    frontier_score: float = 0.25


class ApexNavSemanticSelector(SemanticDetector):
    """ApexNav-style target-centric semantic candidate selector.

    This is not a direct source-code port. It mirrors the high-level decision
    order we want from ApexNav:

    1. known target semantic nodes
    2. likely target / object-region nodes
    3. frontier fallback
    """

    def __init__(self, cfg: ApexNavSelectorConfig | None = None) -> None:
        self.cfg = cfg or ApexNavSelectorConfig()
        self._target_labels = tuple(label.lower() for label in self.cfg.target_labels)

    def detect(self, graph: SemanticGraph, *, current_node_id: str | None = None) -> list[SemanticDetection]:
        current_floor = graph.nodes[current_node_id].floor if current_node_id is not None else None
        visible_nodes = [
            node
            for node in graph.nodes.values()
            if current_floor is None or node.floor == current_floor
        ]

        candidates = self._known_targets(visible_nodes)
        if candidates:
            return candidates

        candidates = self._likely_targets(visible_nodes)
        if candidates:
            return candidates

        return self._frontiers(visible_nodes)

    def _known_targets(self, nodes: list[SemanticNode]) -> list[SemanticDetection]:
        detections: list[SemanticDetection] = []
        for node in nodes:
            if node.kind == "elevator_lobby":
                detections.append(SemanticDetection(node_id=node.node_id, label=node.label or node.kind, score=1.0))
                continue
            label = node.label.lower()
            if any(target in label for target in self._target_labels):
                detections.append(SemanticDetection(node_id=node.node_id, label=node.label or node.kind, score=0.9))
        return sorted(detections, key=lambda item: item.score, reverse=True)

    def _likely_targets(self, nodes: list[SemanticNode]) -> list[SemanticDetection]:
        detections: list[SemanticDetection] = []
        for node in nodes:
            semantic_hint = str(node.attrs.get("semantic_hint", "")).lower()
            if not semantic_hint:
                continue
            if any(target in semantic_hint for target in self._target_labels):
                score = float(node.attrs.get("semantic_score", self.cfg.likely_target_score))
                detections.append(SemanticDetection(node_id=node.node_id, label=semantic_hint, score=score))
        return sorted(detections, key=lambda item: item.score, reverse=True)

    def _frontiers(self, nodes: list[SemanticNode]) -> list[SemanticDetection]:
        detections: list[SemanticDetection] = []
        for node in nodes:
            if node.kind != "frontier":
                continue
            score = float(node.attrs.get("frontier_score", self.cfg.frontier_score))
            detections.append(SemanticDetection(node_id=node.node_id, label=node.label or node.kind, score=score))
        return sorted(detections, key=lambda item: item.score, reverse=True)

