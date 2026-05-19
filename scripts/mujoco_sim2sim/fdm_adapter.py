from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

torch = None


def _ensure_torch():
    global torch
    if torch is None:
        import torch as torch_module

        torch = torch_module
    return torch

try:
    from .config import Sim2SimConfig
    from .fdm_model_bridge import DEFAULT_RUN_DIR, load_fdm_model
    from .low_level_controller import LowLevelCommand
except ImportError:
    from config import Sim2SimConfig
    from fdm_model_bridge import DEFAULT_RUN_DIR, load_fdm_model
    from low_level_controller import LowLevelCommand


@dataclass
class PlannerObservation:
    start_xy_yaw: np.ndarray
    goal_xy_yaw: np.ndarray
    height_scan: np.ndarray
    fdm_state: np.ndarray | None = None
    fdm_proprioception: np.ndarray | None = None


class PlannerAdapter:
    """Boundary around the existing FDM/MPPI planner."""

    def reset(self) -> None:
        pass

    def command(self, obs: PlannerObservation) -> LowLevelCommand:
        raise NotImplementedError

    def debug_info(self) -> dict[str, float]:
        return {}


class ZeroPlannerAdapter(PlannerAdapter):
    """Smoke-test planner that keeps the robot still."""

    def command(self, obs: PlannerObservation) -> LowLevelCommand:
        return LowLevelCommand.zeros()


@dataclass
class ConstantCommandAdapter(PlannerAdapter):
    """Smoke-test planner that sends a fixed G1 base velocity command."""

    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0

    def command(self, obs: PlannerObservation) -> LowLevelCommand:
        return LowLevelCommand(vx=self.vx, vy=self.vy, wz=self.wz)


@dataclass
class GoalTrackingAdapter(PlannerAdapter):
    """Lightweight SE2 goal follower used until the FDM/MPPI adapter is plugged in."""

    max_vx: float = 1.0
    max_vy: float = 0.10
    max_wz: float = 0.66
    kp_xy: float = 0.6
    kp_yaw: float = 0.8
    final_yaw_distance: float = 0.8

    def command(self, obs: PlannerObservation) -> LowLevelCommand:
        x, y, yaw = obs.start_xy_yaw.astype(np.float64)
        gx, gy, gyaw = obs.goal_xy_yaw.astype(np.float64)
        dx_w = gx - x
        dy_w = gy - y
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        dx_b = cos_yaw * dx_w + sin_yaw * dy_w
        dy_b = -sin_yaw * dx_w + cos_yaw * dy_w
        distance = np.hypot(dx_b, dy_b)
        path_heading_err = np.arctan2(dy_b, dx_b)
        final_heading_err = np.arctan2(np.sin(gyaw - yaw), np.cos(gyaw - yaw))
        final_blend = np.clip((self.final_yaw_distance - distance) / max(self.final_yaw_distance, 1e-6), 0.0, 1.0)
        heading_err = self._wrap_to_pi((1.0 - final_blend) * path_heading_err + final_blend * final_heading_err)

        vx = np.clip(self.kp_xy * dx_b, -self.max_vx, self.max_vx)
        vy = np.clip(self.kp_xy * dy_b, -self.max_vy, self.max_vy)
        wz = np.clip(self.kp_yaw * heading_err, -self.max_wz, self.max_wz)
        return LowLevelCommand(vx=float(vx), vy=float(vy), wz=float(wz))

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        return float(np.arctan2(np.sin(angle), np.cos(angle)))


class LocalBatchedMPPIOptimizer:
    """Isaac-free batched MPPI optimizer matching the FDM planner shape contract."""

    def __init__(
        self,
        *,
        horizon: int,
        action_dim: int,
        population_size: int,
        num_iterations: int,
        gamma: float,
        sigma: float,
        beta: float,
        lower_bound: list[float],
        upper_bound: list[float],
        device: str,
        seed: int,
    ):
        torch_module = _ensure_torch()
        self.horizon = horizon
        self.action_dim = action_dim
        self.population_size = population_size
        self.num_iterations = num_iterations
        self.gamma = gamma
        self.sigma = sigma
        self.beta = beta
        self.device = device
        self.lower_bound = torch_module.as_tensor(
            lower_bound, dtype=torch_module.float32, device=device
        ).view(1, horizon, action_dim)
        self.upper_bound = torch_module.as_tensor(
            upper_bound, dtype=torch_module.float32, device=device
        ).view(1, horizon, action_dim)
        self.mean = ((self.lower_bound + self.upper_bound) * 0.5).clone()
        self.var = torch_module.full(
            (1, horizon, action_dim), sigma * sigma, dtype=torch_module.float32, device=device
        )
        self._generator = torch_module.Generator(device=device)
        self._generator.manual_seed(seed)

    def reset(self) -> None:
        self.mean = ((self.lower_bound + self.upper_bound) * 0.5).clone()

    def seed_mean(self, command: np.ndarray) -> None:
        torch_module = _ensure_torch()
        command_t = torch_module.as_tensor(
            command, dtype=torch_module.float32, device=self.device
        ).view(1, 1, self.action_dim)
        self.mean[:] = torch.clamp(command_t, self.lower_bound, self.upper_bound)

    def optimize(self, objective) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _ensure_torch()
        past_action = self.mean[:, 0].clone()
        self.mean[:, :-1] = self.mean[:, 1:].clone()

        last_population = None
        last_values = None
        for _ in range(self.num_iterations):
            lb_dist = self.mean - self.lower_bound
            ub_dist = self.upper_bound - self.mean
            constrained_var = torch.minimum(torch.minimum((lb_dist / 2.0) ** 2, (ub_dist / 2.0) ** 2), self.var)
            noise = torch.empty(
                (self.population_size, self.horizon, self.action_dim),
                dtype=torch.float32,
                device=self.device,
            )
            noise.normal_(generator=self._generator)
            noise = torch.clamp(noise, -2.0, 2.0)
            population = noise * torch.sqrt(constrained_var)
            population[:, 0, :] += self.beta * self.mean[0, 0, :] + (1.0 - self.beta) * past_action[0]
            for step_idx in range(max(self.horizon - 1, 0)):
                population[:, step_idx + 1, :] += (
                    self.beta * self.mean[0, step_idx + 1, :]
                    + (1.0 - self.beta) * population[:, step_idx, :]
                )
            population = torch.minimum(torch.maximum(population, self.lower_bound), self.upper_bound)
            values, _terms = objective(population)
            values = torch.nan_to_num(values, nan=-1e10, posinf=-1e10, neginf=-1e10)
            weights = torch.exp(self.gamma * (values - values.max())).view(-1, 1, 1)
            self.mean[:] = torch.sum(population * weights, dim=0, keepdim=True) / (torch.sum(weights) + 1e-10)
            last_population = population
            last_values = values

        assert last_population is not None and last_values is not None
        return self.mean[0].clone(), last_population, last_values


@dataclass
class FDMPlannerAdapter(PlannerAdapter):
    """Isaac-free FDM/MPPI planner for the MuJoCo bridge."""

    checkpoint: Path
    run_dir: Path = DEFAULT_RUN_DIR
    device: str = "cpu"
    population_size: int = 512
    mppi_iterations: int = 8
    mppi_gamma: float = 1.0
    mppi_sigma: float = 0.8
    mppi_beta: float = 0.6
    seed: int = 7
    replan_interval: int = 5
    max_vx: float = 1.0
    max_vy: float = 0.10
    max_wz: float = 0.66
    action_min_vx: float = -0.10
    action_max_vx: float = 1.50
    action_max_vy: float = 0.10
    action_max_wz: float = 1.00
    gait_min_vx: float = -0.10
    gait_max_vx: float = 1.00
    gait_max_vy: float = 0.10
    gait_max_wz: float = 0.66
    terminal_position_weight: float = 12.0
    terminal_rot_weight: float = 5.0
    terminal_heading_to_goal_weight: float = 2.0
    collision_traj_factor: float = 12.0
    collision_high_risk_factor: float = 1200.0
    collision_threshold: float = 0.5
    collision_safety_factor: float = 0.0
    collision_num_neighbors: int = 2
    collision_neighbor_spread_weight: float = 0.6
    velocity_tracking_weight: float = 0.55
    desired_velocity: float = 0.35
    action_cost_dt: float = 0.25
    heading_running_weight: float = 0.6
    energy_weight: float = 0.0
    smooth_vx_weight: float = 0.02
    smooth_vy_weight: float = 0.02
    smooth_wz_weight: float = 0.02
    yaw_rate_change_weight: float = 0.01
    tracking_prior_weight: float = 0.0
    command_progress_weight: float = 0.0
    terminal_progress_weight: float = 0.0
    lateral_command_weight: float = 1.0
    near_obstacle_soft_weight: float = 6.0
    near_obstacle_hard_weight: float = 30.0
    near_obstacle_soft_distance: float = 0.30
    near_obstacle_hard_distance: float = 0.15
    scan_obstacle_weight: float = 1.0
    scan_obstacle_clearance: float = 0.30
    scan_obstacle_height_threshold: float = 0.08
    scan_obstacle_relative_to_floor: bool = True
    scan_floor_percentile: float = 5.0
    scan_use_footprint: bool = True
    scan_footprint_front: float = 0.45
    scan_footprint_back: float = 0.15
    scan_footprint_half_width: float = 0.28
    near_obstacle_speed_weight: float = 0.0
    near_obstacle_slow_distance: float = 0.90
    near_obstacle_stop_distance: float = 0.35
    front_obstacle_width: float = 0.55
    front_obstacle_lookahead: float = 1.20
    front_obstacle_min_vx: float = 0.30
    height_scan_offset_x: float = 0.0
    height_scan_offset_y: float = 0.0
    scan_resolution: float = Sim2SimConfig.height_scan_resolution
    stabilize_command: bool = False
    yaw_command_limit: float = 0.45
    lateral_command_limit: float = 0.04
    yaw_drift_limit: float = 0.55
    goal_tolerance: float = 0.08
    progress_guard_ratio: float = 0.4
    progress_guard_max_risk: float = 0.25
    progress_guard_max_scan_cost: float = 3.0

    def __post_init__(self) -> None:
        self._model = None
        self._dims: dict | None = None
        self._mppi: LocalBatchedMPPIOptimizer | None = None
        self._last_command = np.zeros(3, dtype=np.float32)
        self._hold_steps_remaining = 0
        self._state_history: list[np.ndarray] = []
        self._proprio_history: list[np.ndarray] = []
        self._debug_info: dict[str, float] = {}
        self._last_progress_guard_applied = 0.0
        self._last_best_idx = 0
        self._last_cost_terms: dict[str, torch.Tensor] = {}
        self._last_state_traj: torch.Tensor | None = None
        self._last_collision_prob: torch.Tensor | None = None
        self._last_energy: torch.Tensor | None = None
        self._last_front_clearance = float("inf")
        self._last_front_vx_limit = self.gait_max_vx
        self._goal_tracker = GoalTrackingAdapter(max_vx=self.max_vx, max_vy=self.max_vy, max_wz=self.max_wz)

    def reset(self) -> None:
        self._last_command[:] = 0.0
        self._hold_steps_remaining = 0
        self._state_history = []
        self._proprio_history = []
        self._debug_info = {}
        self._last_progress_guard_applied = 0.0
        self._last_best_idx = 0
        self._last_front_clearance = float("inf")
        self._last_front_vx_limit = self.gait_max_vx
        if self._mppi is not None:
            self._mppi.reset()

    def debug_info(self) -> dict[str, float]:
        return self._debug_info.copy()

    def _ensure_loaded(self):
        if self._model is None:
            self._model, self._dims = load_fdm_model(
                checkpoint=self.checkpoint,
                run_dir=self.run_dir,
                device=self.device,
            )
            print(
                "[FDM] loaded planner model "
                f"history={self._dims['history']} horizon={self._dims['horizon']} "
                f"height_shape={self._dims['height_shape']}"
            )
            horizon = int(self._dims["horizon"])
            lower_bound = [[self.action_min_vx, -self.action_max_vy, -self.action_max_wz] for _ in range(horizon)]
            upper_bound = [[self.action_max_vx, self.action_max_vy, self.action_max_wz] for _ in range(horizon)]
            self._mppi = LocalBatchedMPPIOptimizer(
                horizon=horizon,
                action_dim=3,
                population_size=self.population_size,
                num_iterations=self.mppi_iterations,
                gamma=self.mppi_gamma,
                sigma=self.mppi_sigma,
                beta=self.mppi_beta,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                device=self.device,
                seed=self.seed,
            )
        return self._model, self._dims

    def command(self, obs: PlannerObservation) -> LowLevelCommand:
        if self._hold_steps_remaining > 0:
            self._hold_steps_remaining -= 1
            self._debug_info["fdm_replan"] = 0.0
            return LowLevelCommand(
                vx=float(self._last_command[0]),
                vy=float(self._last_command[1]),
                wz=float(self._last_command[2]),
            )

        model, dims = self._ensure_loaded()
        assert self._mppi is not None

        self._update_history(obs, int(dims["history"]), int(dims["state_dim"]), int(dims["proprio_dim"]))
        base_command = np.asarray(self._goal_tracker.command(obs).as_array(), dtype=np.float32)
        if not np.any(self._last_command):
            self._mppi.seed_mean(base_command)

        with torch.no_grad():
            mean_sequence, population, values = self._mppi.optimize(lambda actions: self._objective(obs, actions, model, dims))
            base_sequence = torch.as_tensor(base_command, dtype=torch.float32, device=self.device).view(1, 1, 3)
            base_sequence = base_sequence.repeat(1, int(dims["horizon"]), 1)
            candidates = torch.cat([base_sequence, mean_sequence.unsqueeze(0), population], dim=0)
            costs, cost_terms, state_traj, collision_prob, energy = self._evaluate_actions(obs, candidates, model, dims)
            best_idx = int(torch.argmin(costs).item())
            best_idx = self._apply_progress_guard(obs, candidates, base_command, best_idx, collision_prob, cost_terms)

        best = candidates[best_idx, 0].detach().cpu().numpy().astype(np.float32)
        best = self._stabilize_command(obs, best)
        self._record_front_obstacle_limit(obs)
        self._last_command = best
        self._hold_steps_remaining = max(0, self.replan_interval - 1)
        self._last_best_idx = best_idx
        self._last_cost_terms = cost_terms
        self._last_state_traj = state_traj
        self._last_collision_prob = collision_prob
        self._last_energy = energy
        self._update_debug_info(obs, best_idx, costs, cost_terms, state_traj, collision_prob, energy, values)
        return LowLevelCommand(vx=float(best[0]), vy=float(best[1]), wz=float(best[2]))

    def _objective(self, obs: PlannerObservation, actions: torch.Tensor, model, dims: dict) -> tuple[torch.Tensor, dict]:
        costs, terms, _state_traj, _collision_prob, _energy = self._evaluate_actions(obs, actions, model, dims)
        return -costs, terms

    def _evaluate_actions(
        self,
        obs: PlannerObservation,
        actions: torch.Tensor,
        model,
        dims: dict,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = actions.shape[0]
        state = self._state_history_tensor(batch, int(dims["history"]), int(dims["state_dim"]))
        proprio = self._proprio_tensor(batch, int(dims["history"]), int(dims["proprio_dim"]))
        extero = self._height_tensor(obs.height_scan, tuple(dims["height_shape"]), batch)
        add_extero = torch.zeros(batch, 1, dtype=torch.float32, device=self.device)
        model_out = model((state, proprio, extero, actions, add_extero))
        state_traj, collision_prob, energy = model_out[0], model_out[1], model_out[2]
        if len(model_out) > 3:
            collision_prob = torch.maximum(collision_prob, model_out[3])
        costs, terms = self._score_actions(obs, actions, state_traj, collision_prob, energy)
        return costs, terms, state_traj, collision_prob, energy

    def _update_history(self, obs: PlannerObservation, history: int, state_dim: int, proprio_dim: int) -> None:
        if obs.fdm_state is not None:
            state = np.asarray(obs.fdm_state, dtype=np.float32).reshape(-1)
            if state.shape[0] == state_dim:
                self._state_history.append(state.copy())
                del self._state_history[:-history]
        if obs.fdm_proprioception is not None:
            proprio = np.asarray(obs.fdm_proprioception, dtype=np.float32).reshape(-1)
            if proprio.shape[0] == proprio_dim:
                self._proprio_history.append(proprio.copy())
                del self._proprio_history[:-history]

    def _state_history_tensor(self, batch: int, history: int, state_dim: int):
        if self._state_history:
            pad = [self._state_history[0]] * max(0, history - len(self._state_history))
            history_np = np.stack((pad + self._state_history)[-history:], axis=0).astype(np.float32)
        else:
            history_np = np.zeros((history, state_dim), dtype=np.float32)
            if state_dim >= 4:
                history_np[:, 3] = 1.0
            if state_dim >= 8:
                history_np[:, 6:8] = 1.0
        tensor = torch.as_tensor(history_np, dtype=torch.float32, device=self.device)
        return tensor.unsqueeze(0).repeat(batch, 1, 1)

    def _proprio_tensor(self, batch: int, history: int, proprio_dim: int):
        if self._proprio_history:
            pad = [self._proprio_history[0]] * max(0, history - len(self._proprio_history))
            history_np = np.stack((pad + self._proprio_history)[-history:], axis=0).astype(np.float32)
        else:
            history_np = np.zeros((history, proprio_dim), dtype=np.float32)
            if proprio_dim >= 6:
                history_np[:, 5] = -1.0
            if proprio_dim >= 3:
                history_np[:, :3] = self._last_command
        tensor = torch.as_tensor(history_np, dtype=torch.float32, device=self.device)
        return tensor.unsqueeze(0).repeat(batch, 1, 1)

    def _height_tensor(self, height_scan: np.ndarray, shape: tuple[int, int], batch: int):
        height = np.asarray(height_scan, dtype=np.float32)
        if height.shape != shape:
            raise ValueError(f"FDM height scan shape mismatch: got {height.shape}, expected {shape}.")
        height = np.clip(height, -1.0, 1.5)
        tensor = torch.as_tensor(height, dtype=torch.float32, device=self.device)
        return tensor.unsqueeze(0).unsqueeze(0).repeat(batch, 1, 1, 1)

    def _score_actions(self, obs, actions, state_traj, collision_prob, energy):
        x, y, yaw = obs.start_xy_yaw.astype(np.float64)
        gx, gy, gyaw = obs.goal_xy_yaw.astype(np.float64)
        dx_w = gx - x
        dy_w = gy - y
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        goal_body = np.asarray(
            [cos_yaw * dx_w + sin_yaw * dy_w, -sin_yaw * dx_w + cos_yaw * dy_w],
            dtype=np.float32,
        )
        heading_goal = np.float32(np.arctan2(np.sin(gyaw - yaw), np.cos(gyaw - yaw)))
        goal_body_t = torch.as_tensor(goal_body, dtype=torch.float32, device=self.device)
        heading_goal_t = torch.as_tensor(heading_goal, dtype=torch.float32, device=self.device)

        pred_xy = state_traj[:, -1, :2]
        pred_heading = torch.atan2(state_traj[:, -1, 2], state_traj[:, -1, 3])
        pos_offset = torch.linalg.norm(pred_xy - goal_body_t, dim=1)
        rot_error = torch.abs(self._wrap_to_pi_tensor(pred_heading - heading_goal_t))
        goal_vec_from_pred = goal_body_t.unsqueeze(0) - pred_xy
        heading_to_goal = torch.atan2(goal_vec_from_pred[:, 1], goal_vec_from_pred[:, 0])
        heading_to_goal_error = torch.abs(self._wrap_to_pi_tensor(heading_to_goal - pred_heading))
        terminal_cost = (
            self.terminal_position_weight * pos_offset
            + self.terminal_rot_weight * rot_error
            + self.terminal_heading_to_goal_weight * heading_to_goal_error
        )
        if collision_prob.ndim == 1:
            collision_for_cost = collision_prob[:, None]
        else:
            collision_for_cost = collision_prob
        collision_mean = torch.mean(collision_for_cost, dim=1)
        collision_max = torch.max(collision_for_cost, dim=1).values
        threshold = self._collision_threshold()
        high_risk = (collision_for_cost > threshold).any(dim=1).to(torch.float32)
        if self._uses_unified_failure_prediction():
            collision_cost = self.collision_traj_factor * collision_mean
        else:
            collision_cost = self.collision_traj_factor * torch.sum(collision_for_cost, dim=1)
        collision_cost = collision_cost + self.collision_high_risk_factor * high_risk
        collision_cost = self._spread_collision_cost(state_traj, collision_cost)
        energy_cost = torch.mean(torch.relu(energy.squeeze(-1)), dim=1)
        actions_for_cost = actions * float(self.action_cost_dt)
        velocity_tracking_cost = (
            torch.abs(torch.linalg.norm(actions_for_cost[..., :2], dim=2) - self.desired_velocity).mean(dim=1)
        )
        action_diff = actions_for_cost[:, 1:] - actions_for_cost[:, :-1]
        smooth_cost = (
            self.smooth_vx_weight * torch.mean(torch.abs(action_diff[..., 0]), dim=1)
            + self.smooth_vy_weight * torch.mean(torch.abs(action_diff[..., 1]), dim=1)
            + self.smooth_wz_weight * torch.mean(torch.abs(action_diff[..., 2]), dim=1)
        )
        yaw_rate_change_cost = self.yaw_rate_change_weight * torch.mean(torch.abs(action_diff[..., 2]), dim=1)
        base_command_t = torch.as_tensor(
            self._goal_tracker.command(obs).as_array(), dtype=torch.float32, device=self.device
        )
        tracking_prior = torch.mean((actions - base_command_t) ** 2, dim=(1, 2))
        distance_to_goal = torch.linalg.norm(goal_body_t)
        goal_dir = goal_body_t / torch.clamp(distance_to_goal, min=1e-4)
        first_progress = torch.sum(actions[:, 0, :2] * goal_dir, dim=1)
        base_progress = torch.clamp(torch.sum(base_command_t[:2] * goal_dir), min=0.0)
        min_command_progress = 0.5 * base_progress
        command_progress_cost = torch.relu(min_command_progress - first_progress) ** 2
        predicted_progress = torch.sum(pred_xy * goal_dir, dim=1)
        min_terminal_progress = torch.minimum(distance_to_goal, torch.tensor(0.35, device=self.device))
        terminal_progress_cost = torch.relu(min_terminal_progress - predicted_progress) ** 2
        active_goal = (distance_to_goal > self.goal_tolerance).to(torch.float32)
        lateral_command_cost = torch.mean(torch.abs(actions_for_cost[..., 1]), dim=1)
        running_heading_cost = self._running_heading_cost(goal_body_t, state_traj)
        footprint_clearance = self._scan_footprint_clearance(obs.height_scan, state_traj)
        scan_obstacle_cost = self._scan_obstacle_cost_from_clearance(footprint_clearance)
        obstacle_speed_cost = self._obstacle_speed_cost(footprint_clearance, actions)
        total = (
            terminal_cost
            + collision_cost
            + self.energy_weight * energy_cost
            + self.velocity_tracking_weight * velocity_tracking_cost
            + smooth_cost
            + yaw_rate_change_cost
            + self.tracking_prior_weight * tracking_prior
            + active_goal * self.command_progress_weight * command_progress_cost
            + active_goal * self.terminal_progress_weight * terminal_progress_cost
            + self.lateral_command_weight * lateral_command_cost
            + self.heading_running_weight * running_heading_cost
            + scan_obstacle_cost
            + obstacle_speed_cost
        )
        terms = {
            "terminal": terminal_cost,
            "position_offset": pos_offset,
            "rot_error": rot_error,
            "heading_to_goal_error": heading_to_goal_error,
            "collision": collision_cost,
            "collision_mean": collision_mean,
            "collision_max": collision_max,
            "high_risk": high_risk,
            "energy": energy_cost,
            "velocity_tracking": velocity_tracking_cost,
            "smooth": smooth_cost,
            "yaw_rate_change": yaw_rate_change_cost,
            "tracking_prior": tracking_prior,
            "command_progress": command_progress_cost,
            "terminal_progress": terminal_progress_cost,
            "lateral_command": lateral_command_cost,
            "heading_running": running_heading_cost,
            "scan_obstacle": scan_obstacle_cost,
            "obstacle_speed": obstacle_speed_cost,
            "goal_distance": distance_to_goal.expand_as(total),
        }
        return total, terms

    def _collision_threshold(self) -> float:
        base = float(self.collision_threshold)
        return base - self.collision_safety_factor

    def _uses_unified_failure_prediction(self) -> bool:
        model_cfg = getattr(getattr(self, "_model", None), "cfg", None)
        return bool(getattr(model_cfg, "unified_failure_prediction", False))

    def _spread_collision_cost(self, state_traj, base_cost: torch.Tensor) -> torch.Tensor:
        if self.collision_num_neighbors <= 0 or self.collision_neighbor_spread_weight == 0.0:
            return base_cost
        if state_traj.shape[0] <= 1:
            return base_cost

        flattened = state_traj[..., :2].reshape(state_traj.shape[0], -1)
        dist = torch.cdist(flattened, flattened)
        dist.fill_diagonal_(float("inf"))
        k = min(self.collision_num_neighbors, state_traj.shape[0] - 1)
        neighbor_dist, neighbor_idx = torch.topk(dist, k=k, largest=False, dim=1)
        neighbor_cost = base_cost[neighbor_idx]
        propagated = torch.sum(neighbor_cost / (neighbor_dist + 1e-2), dim=1)
        return base_cost + self.collision_neighbor_spread_weight * propagated

    def _scan_obstacle_cost(self, height_scan: np.ndarray, state_traj) -> torch.Tensor:
        footprint_clearance = self._scan_footprint_clearance(height_scan, state_traj)
        return self._scan_obstacle_cost_from_clearance(footprint_clearance)

    def _scan_footprint_clearance(self, height_scan: np.ndarray, state_traj) -> torch.Tensor:
        if self.scan_obstacle_weight <= 0.0:
            return torch.full(
                state_traj.shape[:2],
                float("inf"),
                dtype=state_traj.dtype,
                device=self.device,
            )

        height_np = np.asarray(height_scan, dtype=np.float32)
        obstacle_np = self._height_obstacle_mask(height_np)
        if not np.any(obstacle_np):
            return torch.full(
                state_traj.shape[:2],
                float("inf"),
                dtype=state_traj.dtype,
                device=self.device,
            )

        try:
            from scipy.ndimage import distance_transform_edt

            dist_np = distance_transform_edt(~obstacle_np) * self.scan_resolution
            dist_map = torch.as_tensor(dist_np, dtype=state_traj.dtype, device=self.device)
        except ImportError:
            dist_map = self._torch_distance_to_obstacle(obstacle_np, state_traj.dtype)

        height_h, height_w = height_np.shape
        center = torch.tensor(
            [
                height_h / 2.0 - self.height_scan_offset_y / self.scan_resolution,
                height_w / 2.0 - self.height_scan_offset_x / self.scan_resolution,
            ],
            dtype=state_traj.dtype,
            device=self.device,
        )
        if self.scan_use_footprint:
            query_xy = self._footprint_query_points(state_traj)
        else:
            query_xy = state_traj[..., :2].unsqueeze(-2)
        path_row = (center[0] - query_xy[..., 1] / self.scan_resolution).long()
        path_col = (center[1] + query_xy[..., 0] / self.scan_resolution).long()
        path_row = torch.clamp(path_row, 0, height_h - 1)
        path_col = torch.clamp(path_col, 0, height_w - 1)

        clearance = dist_map[path_row, path_col]
        return torch.min(clearance, dim=-1).values

    def _scan_obstacle_cost_from_clearance(self, footprint_clearance: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(footprint_clearance).any():
            return torch.zeros(footprint_clearance.shape[0], dtype=footprint_clearance.dtype, device=self.device)
        cost = torch.zeros_like(footprint_clearance)
        cost[footprint_clearance < self.near_obstacle_soft_distance] += self.near_obstacle_soft_weight
        cost[footprint_clearance < self.near_obstacle_hard_distance] += self.near_obstacle_hard_weight
        return cost.mean(dim=1) * self.scan_obstacle_weight

    def _height_obstacle_mask(self, height_np: np.ndarray) -> np.ndarray:
        if not self.scan_obstacle_relative_to_floor:
            return height_np > self.scan_obstacle_height_threshold
        finite_height = height_np[np.isfinite(height_np)]
        if finite_height.size == 0:
            return np.zeros_like(height_np, dtype=bool)
        floor_height = float(np.percentile(finite_height, self.scan_floor_percentile))
        return (height_np - floor_height) > self.scan_obstacle_height_threshold

    def _obstacle_speed_cost(self, footprint_clearance: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if self.near_obstacle_speed_weight <= 0.0:
            return torch.zeros(actions.shape[0], dtype=actions.dtype, device=self.device)
        close = torch.relu(
            (self.near_obstacle_slow_distance - footprint_clearance) / self.near_obstacle_slow_distance
        )
        forward_speed = torch.relu(actions[..., 0])
        return self.near_obstacle_speed_weight * torch.mean(close**2 * forward_speed**2, dim=1)

    def _footprint_query_points(self, state_traj) -> torch.Tensor:
        x_offsets = torch.tensor(
            [-self.scan_footprint_back, 0.0, self.scan_footprint_front],
            dtype=state_traj.dtype,
            device=self.device,
        )
        y_offsets = torch.tensor(
            [-self.scan_footprint_half_width, 0.0, self.scan_footprint_half_width],
            dtype=state_traj.dtype,
            device=self.device,
        )
        xx, yy = torch.meshgrid(x_offsets, y_offsets, indexing="ij")
        offsets = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)

        heading_sin = state_traj[..., 2]
        heading_cos = state_traj[..., 3]
        offset_x = offsets[:, 0].view(1, 1, -1)
        offset_y = offsets[:, 1].view(1, 1, -1)
        rot_x = heading_cos.unsqueeze(-1) * offset_x - heading_sin.unsqueeze(-1) * offset_y
        rot_y = heading_sin.unsqueeze(-1) * offset_x + heading_cos.unsqueeze(-1) * offset_y
        center_xy = state_traj[..., :2].unsqueeze(-2)
        return center_xy + torch.stack((rot_x, rot_y), dim=-1)

    def _torch_distance_to_obstacle(self, obstacle_np: np.ndarray, dtype) -> torch.Tensor:
        rows_np, cols_np = np.nonzero(obstacle_np)
        rows = torch.as_tensor(rows_np, dtype=torch.float32, device=self.device)
        cols = torch.as_tensor(cols_np, dtype=torch.float32, device=self.device)
        grid_rows = torch.arange(obstacle_np.shape[0], dtype=torch.float32, device=self.device)
        grid_cols = torch.arange(obstacle_np.shape[1], dtype=torch.float32, device=self.device)
        rr, cc = torch.meshgrid(grid_rows, grid_cols, indexing="ij")
        grid = torch.stack((rr.reshape(-1), cc.reshape(-1)), dim=1)
        obstacles = torch.stack((rows, cols), dim=1)
        dist = torch.cdist(grid, obstacles).min(dim=1).values * self.scan_resolution
        return dist.view(obstacle_np.shape).to(dtype=dtype)

    def _apply_progress_guard(self, obs, candidates, base_command, best_idx, collision_prob, cost_terms) -> int:
        self._last_progress_guard_applied = 0.0
        x, y, yaw = obs.start_xy_yaw.astype(np.float64)
        gx, gy, _gyaw = obs.goal_xy_yaw.astype(np.float64)
        dx_w = gx - x
        dy_w = gy - y
        distance = float(np.hypot(dx_w, dy_w))
        if distance <= self.goal_tolerance:
            return best_idx

        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        goal_body = np.asarray([cos_yaw * dx_w + sin_yaw * dy_w, -sin_yaw * dx_w + cos_yaw * dy_w], dtype=np.float32)
        norm = max(float(np.linalg.norm(goal_body)), 1e-4)
        goal_dir = goal_body / norm
        base_progress = float(np.dot(base_command[:2], goal_dir))
        best_action = candidates[best_idx, 0, :2].detach().cpu().numpy()
        best_progress = float(np.dot(best_action, goal_dir))
        if base_progress <= 0.02:
            return best_idx
        if best_progress >= self.progress_guard_ratio * base_progress:
            return best_idx

        base_risk = float(collision_prob[0].max().item())
        base_scan_cost = float(cost_terms.get("scan_obstacle", torch.zeros_like(collision_prob[:, 0]))[0].item())
        guard_risk_limit = min(self.progress_guard_max_risk, self._collision_threshold())
        if base_risk <= guard_risk_limit and base_scan_cost <= self.progress_guard_max_scan_cost:
            self._last_progress_guard_applied = 1.0
            return 0
        self._last_progress_guard_applied = 0.0
        return best_idx

    @staticmethod
    def _wrap_to_pi_tensor(angle):
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def _running_heading_cost(self, goal_body_t, state_traj):
        pred_xy = state_traj[..., :2]
        pred_heading = torch.atan2(state_traj[..., 2], state_traj[..., 3])
        goal_vec = goal_body_t.view(1, 1, 2) - pred_xy
        goal_heading = torch.atan2(goal_vec[..., 1], goal_vec[..., 0])
        yaw_err = torch.abs(self._wrap_to_pi_tensor(goal_heading - pred_heading))
        return torch.mean(yaw_err, dim=1)

    def _update_debug_info(self, obs, best_idx, costs, terms, state_traj, collision_prob, energy, values) -> None:
        pred_xy = state_traj[best_idx, -1, :2].detach().cpu().numpy()
        pred_heading = torch.atan2(state_traj[best_idx, -1, 2], state_traj[best_idx, -1, 3]).item()
        debug = {
            "fdm_replan": 1.0,
            "fdm_progress_guard": float(self._last_progress_guard_applied),
            "fdm_best_cost": float(costs[best_idx].item()),
            "fdm_mppi_best_value": float(values.max().item()),
            "fdm_mppi_mean_value": float(values.mean().item()),
            "fdm_pred_x": float(pred_xy[0]),
            "fdm_pred_y": float(pred_xy[1]),
            "fdm_pred_yaw": float(pred_heading),
            "fdm_best_risk_max": float(collision_prob[best_idx].max().item()),
            "fdm_best_risk_mean": float(collision_prob[best_idx].mean().item()),
            "fdm_collision_threshold": float(self._collision_threshold()),
            "fdm_best_energy": float(torch.relu(energy[best_idx].squeeze(-1)).mean().item()),
            "fdm_front_clearance": float(self._last_front_clearance),
            "fdm_front_vx_limit": float(self._last_front_vx_limit),
        }
        for name, values in terms.items():
            debug[f"fdm_cost_{name}"] = float(values[best_idx].item())
        self._debug_info = debug

    def _stabilize_command(self, obs: PlannerObservation, command: np.ndarray) -> np.ndarray:
        stabilized = np.asarray(command, dtype=np.float32).copy()
        stabilized[0] = np.clip(stabilized[0], self.gait_min_vx, self.gait_max_vx)
        stabilized[1] = np.clip(stabilized[1], -self.gait_max_vy, self.gait_max_vy)
        stabilized[2] = np.clip(stabilized[2], -self.gait_max_wz, self.gait_max_wz)
        if not self.stabilize_command:
            return stabilized

        x, y, yaw = obs.start_xy_yaw.astype(np.float64)
        gx, gy, _gyaw = obs.goal_xy_yaw.astype(np.float64)
        goal_heading = np.arctan2(gy - y, gx - x)
        heading_error = np.arctan2(np.sin(goal_heading - yaw), np.cos(goal_heading - yaw))

        stabilized[1] = np.clip(stabilized[1], -self.lateral_command_limit, self.lateral_command_limit)
        stabilized[2] = np.clip(stabilized[2], -self.yaw_command_limit, self.yaw_command_limit)
        if abs(heading_error) > self.yaw_drift_limit:
            stabilized[1] *= 0.35
            stabilized[2] = np.clip(0.6 * heading_error, -self.yaw_command_limit, self.yaw_command_limit)
            stabilized[0] = max(stabilized[0], 0.25)
        stabilized[0] = np.clip(stabilized[0], self.gait_min_vx, self.gait_max_vx)
        stabilized[1] = np.clip(stabilized[1], -self.gait_max_vy, self.gait_max_vy)
        stabilized[2] = np.clip(stabilized[2], -self.gait_max_wz, self.gait_max_wz)
        return stabilized

    def _record_front_obstacle_limit(self, obs: PlannerObservation) -> None:
        clearance = self._front_obstacle_clearance(obs.height_scan)
        self._last_front_clearance = float(clearance)
        self._last_front_vx_limit = self.gait_max_vx
        if not np.isfinite(clearance):
            return

        if clearance <= self.near_obstacle_stop_distance:
            max_vx = self.front_obstacle_min_vx
        elif clearance >= self.near_obstacle_slow_distance:
            max_vx = self.gait_max_vx
        else:
            ratio = (clearance - self.near_obstacle_stop_distance) / max(
                self.near_obstacle_slow_distance - self.near_obstacle_stop_distance, 1e-6
            )
            max_vx = self.front_obstacle_min_vx + ratio * (self.gait_max_vx - self.front_obstacle_min_vx)
        self._last_front_vx_limit = float(max_vx)

    def _front_obstacle_clearance(self, height_scan: np.ndarray) -> float:
        height_np = np.asarray(height_scan, dtype=np.float32)
        obstacle_np = self._height_obstacle_mask(height_np)
        if not np.any(obstacle_np):
            return float("inf")

        height_h, height_w = height_np.shape
        center_row = height_h / 2.0 - self.height_scan_offset_y / self.scan_resolution
        center_col = height_w / 2.0 - self.height_scan_offset_x / self.scan_resolution
        rows_np, cols_np = np.nonzero(obstacle_np)
        local_x = (cols_np.astype(np.float32) - center_col) * self.scan_resolution
        local_y = -(rows_np.astype(np.float32) - center_row) * self.scan_resolution
        in_front = (
            (local_x > 0.0)
            & (local_x <= self.front_obstacle_lookahead)
            & (np.abs(local_y) <= self.front_obstacle_width)
        )
        if not np.any(in_front):
            return float("inf")
        return float(np.min(local_x[in_front]))


class MissingFDMPlannerAdapter(PlannerAdapter):
    """Explicit placeholder for the missing FDM checkpoint/Isaac-free adapter."""

    def command(self, obs: PlannerObservation) -> LowLevelCommand:
        raise RuntimeError(
            "FDM/MPPI adapter is not wired yet. Provide the FDM model checkpoint and "
            "an Isaac-free observation adapter before running closed-loop planning."
        )
