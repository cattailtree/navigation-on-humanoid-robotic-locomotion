from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from math import atan2, cos, hypot, pi, sin

from maps.semantic_graph import Pose2D
from planners.grid_astar import GridBounds, OccupancyGridAStar


@dataclass(frozen=True)
class HumanoidSE2AStarConfig:
    grid: OccupancyGridAStar
    yaw_bins: int = 16
    step_distance: float = 0.6
    turn_step_bins: int = 1
    xy_tolerance: float = 0.45
    yaw_tolerance: float = 0.70
    max_expansions: int = 40000
    forward_cost: float = 1.0
    arc_cost: float = 1.25
    turn_cost: float = 0.55
    reverse_cost: float = 3.0
    goal_yaw_weight: float = 0.35
    output_min_spacing: float = 0.55
    output_yaw_threshold: float | None = None


class HumanoidSE2AStar:
    """SE(2) lattice planner with humanoid-friendly forward/turn primitives."""

    def __init__(self, cfg: HumanoidSE2AStarConfig) -> None:
        self.cfg = cfg
        self.grid = cfg.grid

    def plan(self, start: Pose2D, goal: Pose2D) -> list[Pose2D]:
        start_state = (*self.grid.world_to_cell(start.x, start.y), self._yaw_to_bin(start.yaw))
        goal_cell = self.grid.world_to_cell(goal.x, goal.y)
        goal_yaw = self._desired_goal_yaw(start, goal)
        goal_yaw_bin = self._yaw_to_bin(goal_yaw)

        start_state = self._nearest_free_state(start_state)
        if start_state is None:
            return []

        queue: list[tuple[float, float, tuple[int, int, int]]] = [
            (self._heuristic(start_state, goal_cell, goal_yaw_bin), 0.0, start_state)
        ]
        best_cost: dict[tuple[int, int, int], float] = {start_state: 0.0}
        prev: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        visited: set[tuple[int, int, int]] = set()
        best_goal: tuple[int, int, int] | None = None
        expansions = 0

        while queue and expansions < self.cfg.max_expansions:
            _, curr_cost, curr = heappop(queue)
            if curr in visited:
                continue
            visited.add(curr)
            expansions += 1
            if self._is_goal(curr, goal, goal_yaw):
                best_goal = curr
                break
            for nxt, step_cost in self._neighbors(curr):
                next_cost = curr_cost + step_cost
                if next_cost >= best_cost.get(nxt, float("inf")):
                    continue
                best_cost[nxt] = next_cost
                prev[nxt] = curr
                priority = next_cost + self._heuristic(nxt, goal_cell, goal_yaw_bin)
                heappush(queue, (priority, next_cost, nxt))

        if best_goal is None:
            return []
        states = self._reconstruct(start_state, best_goal, prev)
        poses = [self._state_to_pose(state) for state in states]
        poses[-1] = Pose2D(goal.x, goal.y, goal_yaw)
        return self._simplify_pose_path(poses)

    def _neighbors(self, state: tuple[int, int, int]) -> list[tuple[tuple[int, int, int], float]]:
        i, j, yaw_bin = state
        candidates: list[tuple[tuple[int, int, int], float]] = []
        for delta_bin, cost in (
            (0, self.cfg.forward_cost),
            (self.cfg.turn_step_bins, self.cfg.arc_cost),
            (-self.cfg.turn_step_bins, self.cfg.arc_cost),
        ):
            next_yaw_bin = (yaw_bin + delta_bin) % self.cfg.yaw_bins
            yaw = self._bin_to_yaw(next_yaw_bin)
            next_pose = Pose2D(
                self.grid.cell_to_pose((i, j)).x + self.cfg.step_distance * cos(yaw),
                self.grid.cell_to_pose((i, j)).y + self.cfg.step_distance * sin(yaw),
                yaw,
            )
            ni, nj = self.grid.world_to_cell(next_pose.x, next_pose.y)
            if self._is_free((ni, nj)):
                candidates.append(((ni, nj, next_yaw_bin), cost))
        for delta_bin in (self.cfg.turn_step_bins, -self.cfg.turn_step_bins):
            next_yaw_bin = (yaw_bin + delta_bin) % self.cfg.yaw_bins
            candidates.append(((i, j, next_yaw_bin), self.cfg.turn_cost))
        return candidates

    def _nearest_free_state(self, state: tuple[int, int, int]) -> tuple[int, int, int] | None:
        cell = self.grid._nearest_free((state[0], state[1]))
        if cell is None:
            return None
        return (cell[0], cell[1], state[2])

    def _is_free(self, cell: tuple[int, int]) -> bool:
        return self.grid._is_free(cell)

    def _is_goal(self, state: tuple[int, int, int], goal: Pose2D, goal_yaw: float) -> bool:
        pose = self._state_to_pose(state)
        if hypot(pose.x - goal.x, pose.y - goal.y) > self.cfg.xy_tolerance:
            return False
        return abs(_wrap_to_pi(goal_yaw - pose.yaw)) <= self.cfg.yaw_tolerance

    def _heuristic(self, state: tuple[int, int, int], goal_cell: tuple[int, int], goal_yaw_bin: int) -> float:
        cell_dist = hypot(state[0] - goal_cell[0], state[1] - goal_cell[1])
        yaw_dist = min((state[2] - goal_yaw_bin) % self.cfg.yaw_bins, (goal_yaw_bin - state[2]) % self.cfg.yaw_bins)
        return cell_dist * self.grid.cfg.resolution / max(self.cfg.step_distance, 1e-6) + yaw_dist * self.cfg.goal_yaw_weight

    def _state_to_pose(self, state: tuple[int, int, int]) -> Pose2D:
        pose = self.grid.cell_to_pose((state[0], state[1]))
        return Pose2D(pose.x, pose.y, self._bin_to_yaw(state[2]))

    def _yaw_to_bin(self, yaw: float) -> int:
        wrapped = _wrap_to_pi(yaw)
        return int(round((wrapped + pi) / (2.0 * pi) * self.cfg.yaw_bins)) % self.cfg.yaw_bins

    def _bin_to_yaw(self, yaw_bin: int) -> float:
        return _wrap_to_pi((yaw_bin / self.cfg.yaw_bins) * 2.0 * pi - pi)

    def _desired_goal_yaw(self, start: Pose2D, goal: Pose2D) -> float:
        return atan2(goal.y - start.y, goal.x - start.x)

    def _reconstruct(
        self,
        start: tuple[int, int, int],
        goal: tuple[int, int, int],
        prev: dict[tuple[int, int, int], tuple[int, int, int]],
    ) -> list[tuple[int, int, int]]:
        states = [goal]
        curr = goal
        while curr != start:
            curr = prev[curr]
            states.append(curr)
        states.reverse()
        return states

    def _simplify_pose_path(self, poses: list[Pose2D]) -> list[Pose2D]:
        if len(poses) <= 2:
            return poses
        simplified = [poses[0]]
        last = poses[0]
        yaw_threshold = self.cfg.output_yaw_threshold
        if yaw_threshold is None:
            yaw_threshold = pi / self.cfg.yaw_bins
        for pose in poses[1:-1]:
            yaw_changed = abs(_wrap_to_pi(pose.yaw - last.yaw)) >= yaw_threshold
            moved_enough = hypot(pose.x - last.x, pose.y - last.y) >= self.cfg.output_min_spacing
            if yaw_changed or moved_enough:
                simplified.append(pose)
                last = pose
        simplified.append(poses[-1])
        while len(simplified) > 2 and _distance(simplified[-2], simplified[-1]) < self.cfg.output_min_spacing:
            simplified.pop(-2)
        return simplified


def _distance(a: Pose2D, b: Pose2D) -> float:
    return hypot(a.x - b.x, a.y - b.y)


def _wrap_to_pi(angle: float) -> float:
    while angle > pi:
        angle -= 2.0 * pi
    while angle < -pi:
        angle += 2.0 * pi
    return angle
