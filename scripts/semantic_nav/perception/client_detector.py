from __future__ import annotations

from maps.semantic_graph import SemanticGraph, SemanticNode
from perception.detection_types import PerceptionDetection, PerceptionRequest
from perception.perception_client import PerceptionClient
from perception.semantic_detector import SemanticDetection, SemanticDetector


class ClientBackedSemanticDetector(SemanticDetector):
    """Map external perception detections onto semantic graph candidates."""

    def __init__(
        self,
        client: PerceptionClient,
        prompts: tuple[str, ...] = ("elevator", "lift", "elevator door", "elevator sign"),
        min_score: float = 0.1,
        max_bbox_area: float = 0.72,
        min_bbox_area: float = 0.003,
        min_bbox_width: float = 0.18,
        edge_margin: float = 0.03,
        min_label_match_score: float = 0.55,
        image_jpeg_b64: str | None = None,
        log_detections: bool = False,
    ) -> None:
        self.client = client
        self.prompts = prompts
        self.min_score = min_score
        self.max_bbox_area = max_bbox_area
        self.min_bbox_area = min_bbox_area
        self.min_bbox_width = min_bbox_width
        self.edge_margin = edge_margin
        self.min_label_match_score = min_label_match_score
        self.image_jpeg_b64 = image_jpeg_b64
        self.log_detections = log_detections

    def detect(self, graph: SemanticGraph, *, current_node_id: str | None = None) -> list[SemanticDetection]:
        current_floor = graph.nodes[current_node_id].floor if current_node_id is not None else None
        response = self.client.detect(
            PerceptionRequest(
                prompts=self.prompts,
                current_node_id=current_node_id,
                floor=current_floor,
                image_jpeg_b64=self.image_jpeg_b64,
            )
        )

        detections_by_node: dict[str, SemanticDetection] = {}
        for perception_detection in response.detections:
            resolved_node_id = "-"
            resolved_score = 0.0
            if perception_detection.score < self.min_score:
                if self.log_detections:
                    self._log_detection(perception_detection, resolved_node_id, resolved_score, skipped="low_score")
                continue
            bbox_reject_reason = self._bbox_reject_reason(perception_detection)
            if bbox_reject_reason is not None:
                if self.log_detections:
                    self._log_detection(perception_detection, resolved_node_id, resolved_score, skipped=bbox_reject_reason)
                continue
            node = self._resolve_node(graph, perception_detection, current_floor)
            if node is not None:
                resolved_node_id = node.node_id
                resolved_score = self._label_match_score(perception_detection.label.lower(), node)
                if resolved_score < self.min_label_match_score:
                    if self.log_detections:
                        self._log_detection(perception_detection, resolved_node_id, resolved_score, skipped="weak_match")
                    node = None
                    resolved_node_id = "-"
                    resolved_score = 0.0
            if self.log_detections:
                self._log_detection(perception_detection, resolved_node_id, resolved_score)
            if node is None:
                candidate = SemanticDetection(
                    node_id=None,
                    label=perception_detection.label,
                    score=perception_detection.score,
                    bbox=perception_detection.bbox,
                )
                key = f"open:{candidate.label.lower()}"
                previous = detections_by_node.get(key)
                if previous is None or self._candidate_priority(candidate) > self._candidate_priority(previous):
                    detections_by_node[key] = candidate
                continue
            candidate = SemanticDetection(
                node_id=node.node_id,
                label=perception_detection.label or node.label or node.kind,
                score=perception_detection.score,
                bbox=perception_detection.bbox,
            )
            previous = detections_by_node.get(node.node_id)
            if previous is None or self._candidate_priority(candidate) > self._candidate_priority(previous):
                detections_by_node[node.node_id] = candidate
        detections = list(detections_by_node.values())
        detections.sort(key=lambda item: item.score, reverse=True)
        if self.log_detections:
            selected = ", ".join(f"{item.node_id}:{item.score:.3f}" for item in detections) or "none"
            print(f"[semantic_nav:perception] selected={selected}", flush=True)
        return detections

    def _log_detection(
        self,
        detection: PerceptionDetection,
        resolved_node_id: str,
        resolved_score: float,
        *,
        skipped: str | None = None,
    ) -> None:
        bbox = detection.bbox
        if bbox is None:
            bbox_text = "None"
        else:
            bbox_text = f"({bbox.x1:.3f},{bbox.y1:.3f},{bbox.x2:.3f},{bbox.y2:.3f})"
        suffix = f" skipped={skipped}" if skipped is not None else ""
        print(
            f"[semantic_nav:perception] raw label={detection.label} score={detection.score:.3f} "
            f"bbox={bbox_text} -> node={resolved_node_id} match={resolved_score:.2f}{suffix}",
            flush=True,
        )

    def _resolve_node(
        self,
        graph: SemanticGraph,
        detection: PerceptionDetection,
        current_floor: str | None,
    ) -> SemanticNode | None:
        if detection.node_id is not None and detection.node_id in graph.nodes:
            node = graph.nodes[detection.node_id]
            if current_floor is None or node.floor == current_floor:
                return node

        label = detection.label.lower()
        if not label:
            return None

        best_node: SemanticNode | None = None
        best_score = 0.0
        for node in graph.nodes.values():
            if current_floor is not None and node.floor != current_floor:
                continue
            score = self._label_match_score(label, node)
            if score > best_score:
                best_node = node
                best_score = score
        return best_node

    def _label_match_score(self, detection_label: str, node: SemanticNode) -> float:
        labels = [
            node.label.lower(),
            node.kind.lower(),
            str(node.attrs.get("detection_label", "")).lower(),
            str(node.attrs.get("semantic_hint", "")).lower(),
        ]
        labels = [label for label in labels if label]
        if any(detection_label == label for label in labels):
            return 1.0
        if any(detection_label in label or label in detection_label for label in labels):
            return 0.8
        detection_tokens = self._semantic_tokens(detection_label)
        for label in labels:
            if detection_tokens.intersection(self._semantic_tokens(label)):
                return 0.6
        return 0.0

    def _semantic_tokens(self, label: str) -> set[str]:
        tokens = label.replace("_", " ").replace("-", " ").split()
        return {token for token in tokens if len(token) > 2}

    def _candidate_priority(self, detection: SemanticDetection) -> tuple[int, float]:
        label = detection.label.lower()
        if "door" in label:
            return (3, detection.score)
        if label.strip() in {"elevator", "lift"}:
            return (2, detection.score)
        if "elevator" in label or "lift" in label:
            return (1, detection.score)
        return (0, detection.score)

    def _bbox_reject_reason(self, detection: PerceptionDetection) -> str | None:
        if detection.bbox is None:
            return None
        width = max(0.0, detection.bbox.x2 - detection.bbox.x1)
        height = max(0.0, detection.bbox.y2 - detection.bbox.y1)
        area = width * height
        if not (self.min_bbox_area <= area <= self.max_bbox_area):
            return "bbox_area"
        if width < self.min_bbox_width:
            return "bbox_width"
        if detection.bbox.x1 <= self.edge_margin or detection.bbox.x2 >= 1.0 - self.edge_margin:
            return "bbox_edge"
        return None
