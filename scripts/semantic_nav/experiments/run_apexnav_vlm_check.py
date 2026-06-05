from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path


SEMANTIC_NAV_ROOT = Path(__file__).resolve().parents[1]
if str(SEMANTIC_NAV_ROOT) not in sys.path:
    sys.path.insert(0, str(SEMANTIC_NAV_ROOT))

from perception.apexnav_vlm_client import ApexNavGroundingDINOClient, ApexNavYOLOv7Client
from perception.detection_types import PerceptionRequest


def main() -> None:
    parser = argparse.ArgumentParser(description="Check an external ApexNav VLM server with one image.")
    parser.add_argument("--image", type=Path, required=True, help="RGB image path to send to the VLM server.")
    parser.add_argument("--backend", choices=("gdino", "yolov7"), default="gdino")
    parser.add_argument("--endpoint", default=None)
    parser.add_argument(
        "--prompt",
        default="elevator door . elevator sign . lift . doorway . corridor .",
        help="GroundingDINO caption-style prompt. Ignored by YOLOv7 except for label filtering.",
    )
    args = parser.parse_args()

    image_b64 = base64.b64encode(args.image.read_bytes()).decode("utf-8")
    prompts = tuple(part.strip() for part in args.prompt.split(".") if part.strip())
    if args.backend == "gdino":
        client = ApexNavGroundingDINOClient(endpoint=args.endpoint or "http://127.0.0.1:12181/gdino")
    else:
        client = ApexNavYOLOv7Client(endpoint=args.endpoint or "http://127.0.0.1:12184/yolov7")

    response = client.detect(PerceptionRequest(prompts=prompts, image_jpeg_b64=image_b64))
    print(f"[apexnav_vlm_check] backend={args.backend} detections={len(response.detections)}")
    for idx, detection in enumerate(response.detections):
        bbox = detection.bbox
        if bbox is None:
            bbox_text = "None"
        else:
            bbox_text = f"({bbox.x1:.3f}, {bbox.y1:.3f}, {bbox.x2:.3f}, {bbox.y2:.3f})"
        print(
            f"  {idx:02d}. label={detection.label} score={detection.score:.3f} "
            f"bbox={bbox_text} source={detection.source}"
        )


if __name__ == "__main__":
    main()
