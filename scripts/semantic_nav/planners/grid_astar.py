from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from math import hypot

from maps.semantic_graph import Pose2D


@dataclass(frozen=True)
class GridBounds:
    center_x: float
    center_y: float
    size_x: float
    size_y: float

    @property
    def min_x(self) -> float:
        return self.center_x - self.size_x * 0.5

    @property
    def max_x(self) -> float:
        return self.center_x + self.size_x * 0.5

    @property
    def min_y(self) -> float:
        return self.center_y - self.size_y * 0.5

    @property
    def max_y(self) -> float:
        return self.center_y + self.size_y * 0.5


@dataclass(frozen=True)
class GridAStarConfig:
    bounds: GridBounds
    resolution: float = 0.2
    obstacle_margin: float = 0.8


class OccupancyGridAStar:
    def __init__(self, cfg: GridAStarConfig) -> None:
        self.cfg = cfg
        self.width = int(round(cfg.bounds.size_x / cfg.resolution)) + 1
        self.height = int(round(cfg.bounds.size_y / cfg.resolution)) + 1
        self.blocked: set[tuple[int, int]] = set()
        self._add_boundary_walls()

    def plan(self, start: Pose2D, goal: Pose2D) -> list[Pose2D]:
        start_cell = self._nearest_free(self.world_to_cell(start.x, start.y))
        goal_cell = self._nearest_free(self.world_to_cell(goal.x, goal.y))
        if start_cell is None or goal_cell is None:
            return []

        queue: list[tuple[float, float, tuple[int, int]]] = [(self._heuristic(start_cell, goal_cell), 0.0, start_cell)]
        best_cost: dict[tuple[int, int], float] = {start_cell: 0.0}
        prev: dict[tuple[int, int], tuple[int, int]] = {}
        visited: set[tuple[int, int]] = set()

        while queue:
            _, curr_cost, curr = heappop(queue)
            if curr in visited:
                continue
            visited.add(curr)
            if curr == goal_cell:
                cells = self._reconstruct(start_cell, goal_cell, prev)
                return self._simplify_path([self.cell_to_pose(cell) for cell in cells])

            for nxt, step_cost in self._neighbors(curr):
                next_cost = curr_cost + step_cost
                if next_cost >= best_cost.get(nxt, float("inf")):
                    continue
                best_cost[nxt] = next_cost
                prev[nxt] = curr
                heappush(queue, (next_cost + self._heuristic(nxt, goal_cell), next_cost, nxt))
        return []

    def add_rect_obstacle(self, *, center_x: float, center_y: float, size_x: float, size_y: float) -> None:
        min_x = center_x - size_x * 0.5 - self.cfg.obstacle_margin
        max_x = center_x + size_x * 0.5 + self.cfg.obstacle_margin
        min_y = center_y - size_y * 0.5 - self.cfg.obstacle_margin
        max_y = center_y + size_y * 0.5 + self.cfg.obstacle_margin
        min_cell = self.world_to_cell(min_x, min_y)
        max_cell = self.world_to_cell(max_x, max_y)
        for i in range(min(min_cell[0], max_cell[0]), max(min_cell[0], max_cell[0]) + 1):
            for j in range(min(min_cell[1], max_cell[1]), max(min_cell[1], max_cell[1]) + 1):
                self.blocked.add((i, j))

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        i = int(round((x - self.cfg.bounds.min_x) / self.cfg.resolution))
        j = int(round((y - self.cfg.bounds.min_y) / self.cfg.resolution))
        return self._clamp_cell((i, j))

    def cell_to_pose(self, cell: tuple[int, int]) -> Pose2D:
        i, j = cell
        return Pose2D(
            x=self.cfg.bounds.min_x + i * self.cfg.resolution,
            y=self.cfg.bounds.min_y + j * self.cfg.resolution,
            yaw=0.0,
        )

    def _add_boundary_walls(self) -> None:
        margin_cells = max(1, int(round(self.cfg.obstacle_margin / self.cfg.resolution)))
        for i in range(self.width):
            for j in range(self.height):
                if i < margin_cells or j < margin_cells or i >= self.width - margin_cells or j >= self.height - margin_cells:
                    self.blocked.add((i, j))

    def _nearest_free(self, cell: tuple[int, int]) -> tuple[int, int] | None:
        if self._is_free(cell):
            return cell
        max_radius = max(self.width, self.height)
        for radius in range(1, max_radius):
            candidates = []
            ci, cj = cell
            for di in range(-radius, radius + 1):
                candidates.append((ci + di, cj - radius))
                candidates.append((ci + di, cj + radius))
            for dj in range(-radius + 1, radius):
                candidates.append((ci - radius, cj + dj))
                candidates.append((ci + radius, cj + dj))
            free = [self._clamp_cell(candidate) for candidate in candidates if self._is_free(self._clamp_cell(candidate))]
            if free:
                return min(free, key=lambda item: self._heuristic(item, cell))
        return None

    def _neighbors(self, cell: tuple[int, int]) -> list[tuple[tuple[int, int], float]]:
        neighbors = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                nxt = (cell[0] + di, cell[1] + dj)
                if not self._is_free(nxt):
                    continue
                neighbors.append((nxt, 1.4142 if di and dj else 1.0))
        return neighbors

    def _is_free(self, cell: tuple[int, int]) -> bool:
        i, j = cell
        return 0 <= i < self.width and 0 <= j < self.height and cell not in self.blocked

    def _clamp_cell(self, cell: tuple[int, int]) -> tuple[int, int]:
        return min(max(cell[0], 0), self.width - 1), min(max(cell[1], 0), self.height - 1)

    def _heuristic(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        return hypot(a[0] - b[0], a[1] - b[1])

    def _reconstruct(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        prev: dict[tuple[int, int], tuple[int, int]],
    ) -> list[tuple[int, int]]:
        cells = [goal]
        curr = goal
        while curr != start:
            curr = prev[curr]
            cells.append(curr)
        cells.reverse()
        return cells

    def _simplify_path(self, poses: list[Pose2D], min_spacing: float = 0.7) -> list[Pose2D]:
        if len(poses) <= 2:
            return poses
        simplified = [poses[0]]
        last = poses[0]
        for pose in poses[1:-1]:
            if hypot(pose.x - last.x, pose.y - last.y) >= min_spacing:
                simplified.append(pose)
                last = pose
        simplified.append(poses[-1])
        return simplified
