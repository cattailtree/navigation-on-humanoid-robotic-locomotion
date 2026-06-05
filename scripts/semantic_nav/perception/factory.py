from __future__ import annotations

from perception.apexnav_selector import ApexNavSemanticSelector
from perception.apexnav_vlm_client import ApexNavGroundingDINOClient, ApexNavYOLOv7Client
from perception.client_detector import ClientBackedSemanticDetector
from perception.dummy_client import DummyGraphPerceptionClient
from perception.http_client import HttpPerceptionClient
from perception.semantic_detector import GraphSemanticDetector, SemanticDetector
from maps.semantic_graph import SemanticGraph


def make_semantic_detector(
    kind: str = "apexnav",
    *,
    graph: SemanticGraph | None = None,
    perception_endpoint: str | None = None,
    image_jpeg_b64: str | None = None,
    log_detections: bool = False,
    min_score: float = 0.1,
) -> SemanticDetector:
    if kind == "apexnav":
        return ApexNavSemanticSelector()
    if kind == "graph":
        return GraphSemanticDetector()
    if kind == "dummy_client":
        if graph is None:
            raise ValueError("dummy_client detector requires graph")
        return ClientBackedSemanticDetector(DummyGraphPerceptionClient(graph))
    if kind == "http_client":
        if perception_endpoint is None:
            raise ValueError("http_client detector requires perception_endpoint")
        return ClientBackedSemanticDetector(
            HttpPerceptionClient(perception_endpoint),
            image_jpeg_b64=image_jpeg_b64,
            log_detections=log_detections,
            min_score=min_score,
        )
    if kind == "apexnav_gdino":
        endpoint = perception_endpoint or "http://127.0.0.1:12181/gdino"
        return ClientBackedSemanticDetector(
            ApexNavGroundingDINOClient(endpoint=endpoint),
            image_jpeg_b64=image_jpeg_b64,
            log_detections=log_detections,
            min_score=min_score,
        )
    if kind == "apexnav_yolov7":
        endpoint = perception_endpoint or "http://127.0.0.1:12184/yolov7"
        return ClientBackedSemanticDetector(
            ApexNavYOLOv7Client(endpoint=endpoint),
            image_jpeg_b64=image_jpeg_b64,
            log_detections=log_detections,
            min_score=min_score,
        )
    raise ValueError(f"Unknown semantic detector kind: {kind}")
