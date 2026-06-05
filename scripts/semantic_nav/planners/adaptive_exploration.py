from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from maps.semantic_graph import Pose2D
from planners.grid_astar import GridBounds


@dataclass(frozen=True)
class ExplorationViewpoint:
    pose: Pose2D
    name: str


class AdaptiveExplorationStrategy:
    """Small frontier-like exploration policy for bounded Lab experiments.

    This mirrors ApexNav's separation of exploration and target navigation without
    pulling in the full ROS frontier stack: while the target is unseen, visit a
    sequence of viewpoints that sweep the known arena. Once perception detects the
    target, the caller switches to A* target navigation.
    """

    def __init__(
        self,
        bounds: GridBounds,
        *,
        floor: str,
        spacing: float = 1.6,
        wall_margin: float = 1.6,
        prefix_viewpoints: list[ExplorationViewpoint] | None = None,
    ) -> None:
        self.bounds = bounds
        self.floor = floor
        self.wall_margin = wall_margin
        self.viewpoints = (prefix_viewpoints or []) + self._make_lawnmower_viewpoints(spacing)
        self.index = 0

    def next_viewpoint(self, pose: Pose2D) -> ExplorationViewpoint | None:
        while self.index < len(self.viewpoints):
            viewpoint = self.viewpoints[self.index]
            if hypot(viewpoint.pose.x - pose.x, viewpoint.pose.y - pose.y) > 0.7:
                return viewpoint
            self.index += 1
        return None

    def mark_reached(self) -> None:
        self.index += 1

    def _make_lawnmower_viewpoints(self, spacing: float) -> list[ExplorationViewpoint]:
        margin = self.wall_margin
        min_x = self.bounds.min_x + margin
        max_x = self.bounds.max_x - margin
        min_y = self.bounds.min_y + margin
        max_y = self.bounds.max_y - margin
        rows: list[float] = []
        y = min_y
        while y <= max_y:
            rows.append(y)
            y += spacing
        if not rows or rows[-1] < max_y:
            rows.append(max_y)

        viewpoints: list[ExplorationViewpoint] = []
        for row_idx, row_y in enumerate(rows):
            x_values = [min_x, max_x] if row_idx % 2 == 0 else [max_x, min_x]
            for x in x_values:
                next_x = max_x if x == min_x else min_x
                yaw = 0.0 if next_x > x else 3.14159
                viewpoints.append(
                    ExplorationViewpoint(
                        pose=Pose2D(x=x, y=row_y, yaw=yaw),
                        name=f"explore_{len(viewpoints):03d}",
                    )
                )
        return viewpoints
