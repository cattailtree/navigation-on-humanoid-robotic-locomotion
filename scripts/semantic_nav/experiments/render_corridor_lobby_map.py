from __future__ import annotations

import argparse
from math import cos, sin
from pathlib import Path
import sys

SEMANTIC_NAV_ROOT = Path(__file__).resolve().parents[1]
if str(SEMANTIC_NAV_ROOT) not in sys.path:
    sys.path.insert(0, str(SEMANTIC_NAV_ROOT))

import cv2
import numpy as np

from lab_scene.elevator_scene import corridor_lobby_wall_specs
from maps.semantic_graph import Pose2D


def _world_to_px(x: float, y: float, *, bounds: tuple[float, float, float, float], size: tuple[int, int]) -> tuple[int, int]:
    min_x, max_x, min_y, max_y = bounds
    width, height = size
    px = int(round((x - min_x) / (max_x - min_x) * (width - 1)))
    py = int(round((max_y - y) / (max_y - min_y) * (height - 1)))
    return px, py


def _rect_px(
    center: Pose2D,
    rect_size: tuple[float, float],
    *,
    bounds: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    sx, sy = rect_size
    p0 = _world_to_px(center.x - sx * 0.5, center.y - sy * 0.5, bounds=bounds, size=image_size)
    p1 = _world_to_px(center.x + sx * 0.5, center.y + sy * 0.5, bounds=bounds, size=image_size)
    return (min(p0[0], p1[0]), min(p0[1], p1[1])), (max(p0[0], p1[0]), max(p0[1], p1[1]))


def _draw_label(image: np.ndarray, text: str, xy: tuple[int, int], color: tuple[int, int, int] = (30, 30, 30)) -> None:
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a top-down sketch of the semantic corridor-lobby scene.")
    parser.add_argument("--out", type=Path, default=Path("D:/semantic_nav_corridor_lobby_map.png"))
    parser.add_argument("--arena-center", type=float, nargs=2, default=(5.5, 0.4))
    parser.add_argument("--arena-size", type=float, nargs=2, default=(13.0, 9.2))
    parser.add_argument("--elevator-pose", type=float, nargs=3, default=(8.4, 3.0, 3.14159))
    parser.add_argument("--start-pose", type=float, nargs=3, default=(0.0, 0.0, -1.5708))
    args = parser.parse_args()

    width, height = 1200, 820
    margin = 0.8
    cx, cy = args.arena_center
    sx, sy = args.arena_size
    bounds = (cx - sx * 0.5 - margin, cx + sx * 0.5 + margin, cy - sy * 0.5 - margin, cy + sy * 0.5 + margin)

    image = np.full((height, width, 3), (248, 247, 242), dtype=np.uint8)

    arena_min, arena_max = _rect_px(Pose2D(cx, cy, 0.0), (sx, sy), bounds=bounds, image_size=(width, height))
    cv2.rectangle(image, arena_min, arena_max, (180, 185, 190), 2)
    _draw_label(image, "bounded arena", (arena_min[0] + 14, arena_min[1] + 28), (95, 100, 105))

    for name, center, rect_size in corridor_lobby_wall_specs():
        p0, p1 = _rect_px(center, rect_size, bounds=bounds, image_size=(width, height))
        cv2.rectangle(image, p0, p1, (70, 82, 78), -1)
        label_xy = (max(8, p0[0] + 4), max(18, p0[1] - 8))
        _draw_label(image, name, label_xy, (70, 82, 78))

    start = Pose2D(args.start_pose[0], args.start_pose[1], args.start_pose[2])
    elevator = Pose2D(args.elevator_pose[0], args.elevator_pose[1], args.elevator_pose[2])
    start_px = _world_to_px(start.x, start.y, bounds=bounds, size=(width, height))
    elevator_px = _world_to_px(elevator.x, elevator.y, bounds=bounds, size=(width, height))

    cv2.circle(image, start_px, 14, (45, 120, 220), -1)
    arrow_tip = _world_to_px(
        start.x + 1.0 * cos(start.yaw),
        start.y + 1.0 * sin(start.yaw),
        bounds=bounds,
        size=(width, height),
    )
    cv2.arrowedLine(image, start_px, arrow_tip, (45, 120, 220), 3, tipLength=0.25)
    _draw_label(image, "start / robot", (start_px[0] + 18, start_px[1] - 12), (45, 120, 220))

    cv2.rectangle(image, (elevator_px[0] - 34, elevator_px[1] - 18), (elevator_px[0] + 34, elevator_px[1] + 18), (60, 60, 175), -1)
    cv2.line(image, (elevator_px[0], elevator_px[1] - 18), (elevator_px[0], elevator_px[1] + 18), (245, 245, 250), 2)
    _draw_label(image, "elevator target", (elevator_px[0] + 42, elevator_px[1] + 6), (60, 60, 175))

    path_points = [
        start_px,
        _world_to_px(3.0, 0.0, bounds=bounds, size=(width, height)),
        _world_to_px(6.2, 0.0, bounds=bounds, size=(width, height)),
        _world_to_px(7.7, 1.3, bounds=bounds, size=(width, height)),
        elevator_px,
    ]
    cv2.polylines(image, [np.array(path_points, dtype=np.int32)], False, (30, 150, 110), 3, cv2.LINE_AA)
    _draw_label(image, "rough expected route: turn from wall -> corridor -> open lobby -> elevator", (54, height - 46), (30, 120, 90))

    corridor_band_0 = _world_to_px(0.0, -2.30, bounds=bounds, size=(width, height))
    corridor_band_1 = _world_to_px(6.0, 2.30, bounds=bounds, size=(width, height))
    overlay = image.copy()
    cv2.rectangle(overlay, corridor_band_0, corridor_band_1, (220, 235, 245), -1)
    cv2.addWeighted(overlay, 0.18, image, 0.82, 0.0, image)
    _draw_label(image, "wide corridor, not wall-touching", _world_to_px(1.3, 1.25, bounds=bounds, size=(width, height)), (65, 95, 120))

    lobby_band_0 = _world_to_px(6.0, -3.0, bounds=bounds, size=(width, height))
    lobby_band_1 = _world_to_px(10.6, 3.8, bounds=bounds, size=(width, height))
    overlay = image.copy()
    cv2.rectangle(overlay, lobby_band_0, lobby_band_1, (230, 245, 226), -1)
    cv2.addWeighted(overlay, 0.16, image, 0.84, 0.0, image)
    _draw_label(image, "open lobby", _world_to_px(6.1, -2.05, bounds=bounds, size=(width, height)), (75, 120, 80))

    corner_px = _world_to_px(6.2, 0.0, bounds=bounds, size=(width, height))
    cv2.circle(image, corner_px, 10, (20, 120, 90), -1)
    _draw_label(image, "elevator is not in the initial viewing direction", (corner_px[0] + 14, corner_px[1] + 26), (20, 120, 90))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), image)
    print(f"[semantic_nav:map] saved={args.out}")


if __name__ == "__main__":
    main()
