from __future__ import annotations

from typing import Protocol

from perception.detection_types import PerceptionRequest, PerceptionResponse


class PerceptionClient(Protocol):
    """External semantic perception client.

    Heavy vision models such as YOLO/GroundingDINO should live behind this
    interface instead of being imported into the Isaac Lab navigation process.
    """

    def detect(self, request: PerceptionRequest) -> PerceptionResponse:
        ...
