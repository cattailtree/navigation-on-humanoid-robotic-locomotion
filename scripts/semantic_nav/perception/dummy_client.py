from __future__ import annotations

from maps.semantic_graph import SemanticGraph
from perception.detection_types import PerceptionDetection, PerceptionRequest, PerceptionResponse
from perception.perception_client import PerceptionClient


class DummyGraphPerceptionClient(PerceptionClient):
    """Dependency-free perception client used before real camera detections.

    It behaves like a detector service by returning object-like detections, but
    it derives them from the known semantic graph. This keeps the planner path
    identical to the future YOLO/GroundingDINO service path.
    """

    def __init__(self, graph: SemanticGraph, target_labels: tuple[str, ...] = ("elevator", "lift")) -> None:
        self.graph = graph
        self.target_labels = tuple(label.lower() for label in target_labels)

    def detect(self, request: PerceptionRequest) -> PerceptionResponse:
        detections: list[PerceptionDetection] = []
        current_floor = request.floor
        if current_floor is None and request.current_node_id is not None:
            current_floor = self.graph.nodes[request.current_node_id].floor

        prompt_terms = tuple(prompt.lower() for prompt in request.prompts)
        target_terms = prompt_terms or self.target_labels
        for node in self.graph.nodes.values():
            if current_floor is not None and node.floor != current_floor:
                continue
            score = self._score_node(node.kind.lower(), node.label.lower(), target_terms)
            if score <= 0.0:
                continue
            detections.append(
                PerceptionDetection(
                    label=node.label or node.kind,
                    score=score,
                    node_id=node.node_id,
                    source="dummy_graph",
                )
            )
        detections.sort(key=lambda item: item.score, reverse=True)
        return PerceptionResponse(detections=tuple(detections))

    def _score_node(self, kind: str, label: str, target_terms: tuple[str, ...]) -> float:
        if kind == "elevator_lobby":
            return 1.0
        if any(term in label for term in target_terms):
            return 0.9
        if any(term in kind for term in target_terms):
            return 0.8
        return 0.0
