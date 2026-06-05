from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox2D:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class PerceptionDetection:
    label: str
    score: float
    bbox: BoundingBox2D | None = None
    node_id: str | None = None
    source: str = "unknown"


@dataclass(frozen=True)
class PerceptionRequest:
    prompts: tuple[str, ...]
    current_node_id: str | None = None
    floor: str | None = None
    image_jpeg_b64: str | None = None


@dataclass(frozen=True)
class PerceptionResponse:
    detections: tuple[PerceptionDetection, ...]
