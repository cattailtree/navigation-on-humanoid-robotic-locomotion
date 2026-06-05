from __future__ import annotations

import json
from urllib import request as url_request

from perception.detection_types import BoundingBox2D, PerceptionDetection, PerceptionRequest, PerceptionResponse
from perception.perception_client import PerceptionClient


class HttpPerceptionClient(PerceptionClient):
    """Small stdlib HTTP client for an external YOLO/GroundingDINO service."""

    def __init__(self, endpoint: str, timeout_s: float = 2.0) -> None:
        self.endpoint = endpoint
        self.timeout_s = timeout_s

    def detect(self, request: PerceptionRequest) -> PerceptionResponse:
        payload = {
            "prompts": list(request.prompts),
            "current_node_id": request.current_node_id,
            "floor": request.floor,
        }
        body = json.dumps(payload).encode("utf-8")
        http_request = url_request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with url_request.urlopen(http_request, timeout=self.timeout_s) as response:
            data = json.loads(response.read().decode("utf-8"))

        detections = tuple(self._parse_detection(item) for item in data.get("detections", []))
        return PerceptionResponse(detections=detections)

    def _parse_detection(self, item: dict) -> PerceptionDetection:
        bbox = item.get("bbox")
        parsed_bbox = None
        if bbox is not None:
            parsed_bbox = BoundingBox2D(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        return PerceptionDetection(
            label=str(item.get("label", "")),
            score=float(item.get("score", 0.0)),
            bbox=parsed_bbox,
            node_id=item.get("node_id"),
            source=str(item.get("source", "http")),
        )
