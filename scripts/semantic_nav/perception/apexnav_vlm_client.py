from __future__ import annotations

import json
from urllib import request as url_request

from perception.detection_types import BoundingBox2D, PerceptionDetection, PerceptionRequest, PerceptionResponse
from perception.perception_client import PerceptionClient


class ApexNavGroundingDINOClient(PerceptionClient):
    """Client compatible with ApexNav's `vlm.detector.grounding_dino` server."""

    def __init__(
        self,
        endpoint: str = "http://localhost:12181/gdino",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        timeout_s: float = 20.0,
    ) -> None:
        self.endpoint = endpoint
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.timeout_s = timeout_s

    def detect(self, request: PerceptionRequest) -> PerceptionResponse:
        if request.image_jpeg_b64 is None:
            raise ValueError("ApexNavGroundingDINOClient requires PerceptionRequest.image_jpeg_b64")
        payload = {
            "image": request.image_jpeg_b64,
            "caption": _apexnav_caption(request.prompts),
            "box_threshold": self.box_threshold,
            "text_threshold": self.text_threshold,
        }
        return _post_apexnav_detections(self.endpoint, payload, "grounding_dino", self.timeout_s)


class ApexNavYOLOv7Client(PerceptionClient):
    """Client compatible with ApexNav's `vlm.detector.yolov7` server."""

    def __init__(
        self,
        endpoint: str = "http://localhost:12184/yolov7",
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        agnostic_nms: bool = True,
        timeout_s: float = 20.0,
    ) -> None:
        self.endpoint = endpoint
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.agnostic_nms = agnostic_nms
        self.timeout_s = timeout_s

    def detect(self, request: PerceptionRequest) -> PerceptionResponse:
        if request.image_jpeg_b64 is None:
            raise ValueError("ApexNavYOLOv7Client requires PerceptionRequest.image_jpeg_b64")
        payload = {
            "image": request.image_jpeg_b64,
            "agnostic_nms": self.agnostic_nms,
            "conf_thres": self.conf_thres,
            "iou_thres": self.iou_thres,
        }
        response = _post_apexnav_detections(self.endpoint, payload, "yolov7", self.timeout_s)
        if not request.prompts:
            return response
        prompt_terms = tuple(prompt.lower() for prompt in request.prompts)
        filtered = tuple(
            detection
            for detection in response.detections
            if any(term in detection.label.lower() or detection.label.lower() in term for term in prompt_terms)
        )
        return PerceptionResponse(detections=filtered)


def _post_apexnav_detections(
    endpoint: str,
    payload: dict,
    source: str,
    timeout_s: float,
) -> PerceptionResponse:
    body = json.dumps(payload).encode("utf-8")
    http_request = url_request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with url_request.urlopen(http_request, timeout=timeout_s) as response:
        data = json.loads(response.read().decode("utf-8"))
    return _parse_object_detections_json(data, source)


def _parse_object_detections_json(data: dict, source: str) -> PerceptionResponse:
    boxes = data.get("boxes", [])
    logits = data.get("logits", [])
    phrases = data.get("phrases", [])
    detections: list[PerceptionDetection] = []
    for box, logit, phrase in zip(boxes, logits, phrases):
        detections.append(
            PerceptionDetection(
                label=str(phrase),
                score=float(logit),
                bbox=BoundingBox2D(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                source=source,
            )
        )
    return PerceptionResponse(detections=tuple(detections))


def _apexnav_caption(prompts: tuple[str, ...]) -> str:
    phrases = [prompt.strip(" .") for prompt in prompts if prompt.strip(" .")]
    if not phrases:
        return ""
    return " . ".join(phrases) + " ."
