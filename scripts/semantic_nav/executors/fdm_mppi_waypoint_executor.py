from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin
from pathlib import Path
from typing import Any

import hydra
import omegaconf
import torch
import yaml

from executors.robot_adapter import VelocityCommand
from executors.waypoint_executor import ExecutorStatus, WaypointExecutorConfig
from maps.semantic_graph import Pose2D
from planners.execution_plan import ExecutionStep


@dataclass(frozen=True)
class FdmMppiExecutorConfig:
    run_dir: Path
    checkpoint: Path
    use_fdm: bool = True
    population_size: int = 512
    replan_every: int = 5
    warmup_steps: int | None = None
    lookahead_distance: float = 2.0
    pass_tolerance: float = 0.75
    progress_margin: float = 0.25
    final_tolerance: float = 1.15
    face_subgoal: bool = True
    min_forward_carrot: float = 1.0
    final_approach_distance: float = 2.0
    final_approach_tolerance: float = 0.55
    final_approach_yaw_tolerance: float = 0.35
    final_turn_wz: float = 0.45
    final_waypoint_handoff_distance: float = 1.0
    disable_collision_cost_goal_radius: float = 0.0
    disable_mppi_risk_cost_goal_radius: float = 0.0
    subgoal_yaw_gate: float = 0.75
    subgoal_yaw_tolerance: float = 0.25
    subgoal_turn_max_steps: int = 60
    min_vx: float = -0.1
    max_vx: float = 1.0
    max_vy: float = 0.3
    max_wz: float = 0.2


class FdmMppiWaypointExecutor:
    """Waypoint executor backed by the repository FDM + MPPI planner."""

    def __init__(
        self,
        *,
        env,
        steps: list[ExecutionStep],
        waypoint_cfg: WaypointExecutorConfig,
        planner_cfg: Any,
        fdm_cfg: FdmMppiExecutorConfig,
    ) -> None:
        self.env = env
        self.steps = steps
        self.waypoint_cfg = waypoint_cfg
        self.fdm_cfg = fdm_cfg
        self.step_index = 0
        self.device = env.device
        self._env_ids = torch.tensor([0], device=self.device, dtype=torch.long)
        self._step_counter = 0
        self._horizon = int(planner_cfg.model_cfg.prediction_horizon)
        self._last_plan = torch.zeros((1, self._horizon, 3), device=self.device)

        self.model = self._load_model(planner_cfg) if self.fdm_cfg.use_fdm else None
        self.planner = self._build_planner(planner_cfg)
        if self.fdm_cfg.use_fdm:
            self._attach_fdm_model_without_terrain_analysis()
            self._init_obs_buffers()
        else:
            self._init_empty_obs_buffers()
        self._initial_random_actions = torch.randn(
            self.env.num_envs,
            self._horizon,
            3,
            device=self.device,
        ) * 0.05
        self._last_plan[:] = self._initial_random_actions[:1]
        self._resample_next = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)
        self._subgoal_turning = False
        self._subgoal_turn_steps = 0
        self._subgoal_turn_target_name: str | None = None
        self._subgoal_turn_exhausted_target_name: str | None = None
        self._debug_info: dict[str, Any] = {}
        self._last_fdm_snapshot: dict[str, torch.Tensor] | None = None

    def reset(self) -> None:
        self.step_index = 0
        self._step_counter = 0
        self._last_plan[:] = self._initial_random_actions[:1]
        self._resample_next[:] = True
        self._subgoal_turning = False
        self._subgoal_turn_steps = 0
        self._subgoal_turn_target_name = None
        self._subgoal_turn_exhausted_target_name = None
        self._debug_info = {}
        self._last_fdm_snapshot = None
        self._reset_obs_buffers(self._env_ids)

    def current_step(self) -> ExecutionStep | None:
        if self.step_index >= len(self.steps):
            return None
        return self.steps[self.step_index]

    def update(self, robot_pose: Pose2D) -> tuple[VelocityCommand, ExecutorStatus]:
        step = self.current_step()
        if step is None:
            return VelocityCommand(0.0, 0.0, 0.0), ExecutorStatus(True, self.step_index, None)

        if step.kind == "floor_transition":
            self.step_index += 1
            self._resample_next[:] = True
            self._subgoal_turning = False
            self._subgoal_turn_steps = 0
            self._subgoal_turn_target_name = None
            self._subgoal_turn_exhausted_target_name = None
            return (
                VelocityCommand(0.0, 0.0, 0.0),
                ExecutorStatus(False, self.step_index, step, event=step.description),
            )

        advanced, event = self._advance_reached_waypoints(robot_pose)
        if advanced:
            step = self.current_step()
            if step is None:
                return VelocityCommand(0.0, 0.0, 0.0), ExecutorStatus(True, self.step_index, None, event=event)
            if step.kind == "floor_transition":
                return VelocityCommand(0.0, 0.0, 0.0), ExecutorStatus(False, self.step_index, step, event=event)

        if self._final_region_reached(robot_pose):
            final_step = self.current_step()
            self.step_index = len(self.steps)
            self._resample_next[:] = True
            final_name = final_step.node_id if final_step is not None else "final"
            final_event = f"{event}; reached final_region {final_name}" if event else f"reached final_region {final_name}"
            return VelocityCommand(0.0, 0.0, 0.0), ExecutorStatus(True, self.step_index, final_step, event=final_event)

        final_handoff = self._final_waypoint_handoff_command(robot_pose)
        if final_handoff is not None:
            command, handoff_event = final_handoff
            active_event = f"{event}; {handoff_event}" if event else handoff_event
            return command, ExecutorStatus(False, self.step_index, step, event=active_event)

        target = self._directional_subgoal(robot_pose)
        if target is None:
            self._resample_next[:] = True
            return (
                VelocityCommand(0.0, 0.0, 0.0),
                ExecutorStatus(True, self.step_index, None, event=event),
            )

        target_pose, target_name = target
        self._debug_info = {
            "target_name": target_name,
            "target_x": target_pose.x,
            "target_y": target_pose.y,
            "target_yaw": target_pose.yaw,
            "target_dist": _distance(robot_pose, target_pose),
            "target_yaw_err": _wrap_to_pi(target_pose.yaw - robot_pose.yaw),
            "subgoal_turning": self._subgoal_turning,
            "subgoal_turn_steps": self._subgoal_turn_steps,
        }
        command = self._plan_command(robot_pose, target_pose)
        active_event = event if target_name == step.node_id else f"{event}; subgoal={target_name}" if event else f"subgoal={target_name}"
        return command, ExecutorStatus(False, self.step_index, step, event=active_event)

    def notify_env_reset(self) -> None:
        self._resample_next[:] = True
        self._subgoal_turning = False
        self._subgoal_turn_steps = 0
        self._subgoal_turn_target_name = None
        self._subgoal_turn_exhausted_target_name = None
        self._debug_info = {}
        self._last_fdm_snapshot = None
        self._reset_obs_buffers(self._env_ids)

    def debug_info(self) -> dict[str, Any]:
        return dict(self._debug_info)

    def fdm_snapshot(self) -> dict[str, torch.Tensor] | None:
        if self._last_fdm_snapshot is None:
            return None
        return {key: value.clone() for key, value in self._last_fdm_snapshot.items()}

    def _load_model(self, planner_cfg: Any):
        saved_cfg = self._load_saved_model_cfg()
        if saved_cfg is not None:
            self._apply_cfg_dict(planner_cfg.model_cfg, saved_cfg)
        state_dict = torch.load(self.fdm_cfg.checkpoint, map_location=self.device, weights_only=True)
        if hasattr(planner_cfg.model_cfg, "use_geometric_collision_head"):
            planner_cfg.model_cfg.use_geometric_collision_head = any(
                str(key).startswith("geometric_collision_") for key in state_dict
            )
        if hasattr(planner_cfg.model_cfg, "proprioceptive_dim"):
            planner_cfg.model_cfg.proprioceptive_dim = self._proprio_dim()
        model = planner_cfg.model_cfg.class_type(cfg=planner_cfg.model_cfg, device=self.device)
        model.to(self.device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        print(f"[semantic_nav:fdm_mppi] loaded FDM checkpoint={self.fdm_cfg.checkpoint}", flush=True)
        return model

    def _load_saved_model_cfg(self) -> dict | None:
        cfg_path = self.fdm_cfg.run_dir / "params" / "config.yaml"
        if not cfg_path.exists():
            print(f"[semantic_nav:fdm_mppi] missing saved config={cfg_path}; using current planner model_cfg", flush=True)
            return None
        with cfg_path.open("r", encoding="utf-8") as cfg_file:
            cfg = yaml.load(cfg_file, Loader=yaml.UnsafeLoader)
        return cfg.get("model_cfg") if isinstance(cfg, dict) else None

    def _apply_cfg_dict(self, target, source: dict) -> None:
        for key, value in source.items():
            if key == "class_type":
                continue
            if not hasattr(target, key):
                continue
            current = getattr(target, key)
            if isinstance(value, dict) and current is not None:
                self._apply_cfg_dict(current, value)
            else:
                setattr(target, key, value)

    def _build_planner(self, planner_cfg: Any):
        from fdm.planner import get_planner_cfg

        planner_dict = get_planner_cfg(
            self.env.num_envs,
            traj_dim=planner_cfg.model_cfg.prediction_horizon,
            device=self.device,
            population_size=self.fdm_cfg.population_size,
        )
        planner_dict["to_cfg"]["control"] = "fdm" if self.fdm_cfg.use_fdm else "velocity_control"
        planner_dict["to_cfg"]["debug"] = False
        planner_dict["to_cfg"]["init_debug"] = False
        planner_dict["to_cfg"]["batch_size"] = max(256, self.fdm_cfg.population_size)
        planner_dict["to_cfg"]["terminal_cost_w_rot_error"] = 0.0
        planner_dict["to_cfg"]["collision_cost_disable_goal_radius"] = self.fdm_cfg.disable_collision_cost_goal_radius
        planner_dict["to_cfg"]["mppi_risk_cost_disable_goal_radius"] = self.fdm_cfg.disable_mppi_risk_cost_goal_radius
        planner_dict["action_cfg"]["lower_bound"] = [
            self.fdm_cfg.min_vx,
            -self.fdm_cfg.max_vy,
            -self.fdm_cfg.max_wz,
        ]
        planner_dict["action_cfg"]["upper_bound"] = [
            self.fdm_cfg.max_vx,
            self.fdm_cfg.max_vy,
            self.fdm_cfg.max_wz,
        ]
        planner_dict["optim"]["lower_bound"] = ["${action_cfg.lower_bound}" for _ in range(self._horizon)]
        planner_dict["optim"]["upper_bound"] = ["${action_cfg.upper_bound}" for _ in range(self._horizon)]
        planner_dict["optim"]["batch_size"] = self.env.num_envs
        cfg = omegaconf.OmegaConf.create(planner_dict)
        return hydra.utils.instantiate(cfg.to)

    def _attach_fdm_model_without_terrain_analysis(self) -> None:
        self.planner.fdm_model = self.model
        self.planner.terrain_analysis = None
        sensor = self.env.scene.sensors.get("env_sensor")
        if sensor is not None:
            self.planner.height_scan_resolution = getattr(sensor.cfg.pattern_cfg, "resolution", 1.0)
            self.planner.height_scan_size = getattr(sensor.cfg.pattern_cfg, "size", (10.0, 10.0))
            self.planner.height_scan_offset = sensor.cfg.offset.pos
        else:
            self.planner.height_scan_resolution = 1.0
            self.planner.height_scan_size = (10.0, 10.0)
            self.planner.height_scan_offset = (0.0, 0.0, 0.0)
        print("[semantic_nav:fdm_mppi] attached FDM model without TerrainAnalysis", flush=True)

    def _plan_command(self, robot_pose: Pose2D, target_pose: Pose2D) -> VelocityCommand:
        obs = self._collect_observations()
        if self.fdm_cfg.use_fdm:
            self._update_obs_buffers(
                state=obs["fdm_state"].clone(),
                proprio=obs["fdm_obs_proprioception"].clone(),
            )
        else:
            self._obs_env_step_counter += 1

        warmup_steps = self.fdm_cfg.warmup_steps
        if warmup_steps is None:
            warmup_steps = self._horizon if self.fdm_cfg.use_fdm else 0
        should_replan = (
            self._step_counter % max(1, self.fdm_cfg.replan_every) == 0
            and int(self._obs_env_step_counter[0].item()) >= warmup_steps
        )
        if should_replan:
            planner_obs = obs["planner_obs"]
            planner_obs["goal"] = torch.tensor(
                [[target_pose.x, target_pose.y, target_pose.yaw]],
                device=self.device,
                dtype=torch.float32,
            )
            planner_obs["start"] = torch.tensor(
                [[robot_pose.x, robot_pose.y, robot_pose.yaw]],
                device=self.device,
                dtype=torch.float32,
            )
            planner_obs["resample_population"] = self._resample_next.clone()
            if self.fdm_cfg.use_fdm:
                planner_obs["states"] = self._state_history.clone()
                planner_obs["proprio_obs"] = self._proprio_obs_history.clone()
                planner_obs["extero_obs"] = obs["fdm_obs_exteroceptive"].clone()
            with torch.inference_mode():
                _, planned_actions = self.planner.plan(obs=planner_obs, env_ids=self._env_ids, return_states=True)
            self._last_plan[0] = planned_actions[0]
            self._resample_next[:] = False

        # Execute only the first planned command for this env step. The
        # remaining sequence is used only for FDM rollout and MPPI costs.
        action = self._last_plan[0, 0].detach()
        if self.fdm_cfg.use_fdm:
            self._last_fdm_snapshot = {
                "state_history": self._state_history.detach().clone(),
                "proprio_history": self._proprio_obs_history.detach().clone(),
                "extero_obs": obs["fdm_obs_exteroceptive"].detach().clone(),
                "last_plan": self._last_plan.detach().clone(),
            }
        self._step_counter += 1
        return VelocityCommand(float(action[0].item()), float(action[1].item()), float(action[2].item()))

    def _collect_observations(self) -> dict:
        with torch.inference_mode():
            return self.env.observation_manager.compute()

    def _init_obs_buffers(self) -> None:
        self._state_history = torch.zeros(
            (
                self.env.num_envs,
                self.model.cfg.history_length,
                *(self.env.observation_manager.group_obs_dim["fdm_state"]),
            ),
            device=self.device,
        )
        self._proprio_obs_history = torch.zeros(
            (
                self.env.num_envs,
                self.model.cfg.history_length,
                *(self.env.observation_manager.group_obs_dim["fdm_obs_proprioception"]),
            ),
            device=self.device,
        )
        self._obs_env_step_counter = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.int)
        self._history_collection_interval = max(
            1,
            int(round(self.model.cfg.command_timestep / self.env.step_dt / self.model.cfg.history_length)),
        )

    def _init_empty_obs_buffers(self) -> None:
        self._state_history = torch.empty((self.env.num_envs, 0), device=self.device)
        self._proprio_obs_history = torch.empty((self.env.num_envs, 0), device=self.device)
        self._obs_env_step_counter = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.int)
        self._history_collection_interval = 1

    def _update_obs_buffers(self, *, state: torch.Tensor, proprio: torch.Tensor) -> None:
        updatable_envs = (self._obs_env_step_counter % self._history_collection_interval).type(torch.int) == 0
        self._state_history[updatable_envs] = torch.roll(self._state_history[updatable_envs], 1, dims=1)
        self._state_history[updatable_envs, 0] = state[updatable_envs]
        self._proprio_obs_history[updatable_envs] = torch.roll(self._proprio_obs_history[updatable_envs], 1, dims=1)
        self._proprio_obs_history[updatable_envs, 0] = proprio[updatable_envs]
        self._obs_env_step_counter += 1

    def _reset_obs_buffers(self, env_ids: torch.Tensor) -> None:
        self._obs_env_step_counter[env_ids] *= 0
        self._state_history[env_ids] *= 0
        self._proprio_obs_history[env_ids] *= 0

    def _proprio_dim(self) -> int:
        dims = self.env.observation_manager.group_obs_term_dim["fdm_obs_proprioception"]

        def numel(value) -> int:
            if isinstance(value, (list, tuple)):
                result = 1
                for item in value:
                    result *= int(item)
                return result
            return int(value)

        if isinstance(dims, (list, tuple)):
            return sum(numel(dim) for dim in dims)
        return numel(dims)

    def _reached(self, robot_pose: Pose2D, target_pose: Pose2D, *, tolerance: float | None = None) -> bool:
        dx = target_pose.x - robot_pose.x
        dy = target_pose.y - robot_pose.y
        threshold = self.waypoint_cfg.xy_tolerance if tolerance is None else tolerance
        return (dx * dx + dy * dy) ** 0.5 <= threshold

    def _target_tolerance(self, idx: int) -> float:
        if idx == len(self.steps) - 1 and 0 <= idx < len(self.steps):
            return max(self.waypoint_cfg.xy_tolerance, self.fdm_cfg.final_tolerance)
        return self.waypoint_cfg.xy_tolerance

    def _final_region_reached(self, robot_pose: Pose2D) -> bool:
        if self.step_index != len(self.steps) - 1 or self.step_index < 0:
            return False
        step = self.steps[self.step_index]
        if step.kind != "walk_to":
            return False
        distance = _distance(robot_pose, step.pose)
        if distance <= self.fdm_cfg.final_approach_distance:
            return distance <= self.fdm_cfg.final_approach_tolerance
        return distance <= self._target_tolerance(self.step_index)

    def _final_approach_turn_command(self, robot_pose: Pose2D) -> tuple[VelocityCommand, str] | None:
        if self.step_index != len(self.steps) - 1 or self.step_index < 0:
            return None
        step = self.steps[self.step_index]
        if step.kind != "walk_to":
            return None
        distance = _distance(robot_pose, step.pose)
        if distance > self.fdm_cfg.final_approach_distance:
            return None
        target_yaw = _heading(robot_pose, step.pose)
        yaw_error = _wrap_to_pi(target_yaw - robot_pose.yaw)
        if abs(yaw_error) <= self.fdm_cfg.final_approach_yaw_tolerance:
            return None
        wz = max(-self.fdm_cfg.final_turn_wz, min(self.fdm_cfg.final_turn_wz, 1.2 * yaw_error))
        return VelocityCommand(0.0, 0.0, wz), f"final_approach_turn yaw_err={yaw_error:.2f}"

    def _final_waypoint_handoff_command(self, robot_pose: Pose2D) -> tuple[VelocityCommand, str] | None:
        if self.step_index != len(self.steps) - 1 or self.step_index < 0:
            return None
        step = self.steps[self.step_index]
        if step.kind != "walk_to":
            return None
        distance = _distance(robot_pose, step.pose)
        if distance > self.fdm_cfg.final_waypoint_handoff_distance:
            return None
        command = _tracking_command(robot_pose, step.pose, self.waypoint_cfg)
        self._resample_next[:] = True
        self._last_plan.zero_()
        self._subgoal_turning = False
        self._subgoal_turn_steps = 0
        self._subgoal_turn_target_name = None
        self._subgoal_turn_exhausted_target_name = None
        self._debug_info = {
            "target_name": step.node_id,
            "target_x": step.pose.x,
            "target_y": step.pose.y,
            "target_yaw": step.pose.yaw,
            "target_dist": distance,
            "target_yaw_err": _wrap_to_pi(_heading(robot_pose, step.pose) - robot_pose.yaw),
            "subgoal_turning": False,
            "subgoal_turn_steps": 0,
        }
        return command, f"final_waypoint_handoff dist={distance:.2f}"

    def _subgoal_turn_command(
        self,
        robot_pose: Pose2D,
        target_pose: Pose2D,
        target_name: str,
    ) -> tuple[VelocityCommand, str] | None:
        yaw_error = _wrap_to_pi(target_pose.yaw - robot_pose.yaw)
        if target_name != self._subgoal_turn_target_name:
            self._subgoal_turning = False
            self._subgoal_turn_steps = 0
            self._subgoal_turn_target_name = target_name
            self._subgoal_turn_exhausted_target_name = None
        if self._subgoal_turn_exhausted_target_name == target_name:
            return None
        threshold = self.fdm_cfg.subgoal_yaw_tolerance if self._subgoal_turning else self.fdm_cfg.subgoal_yaw_gate
        if abs(yaw_error) <= threshold:
            self._subgoal_turning = False
            self._subgoal_turn_steps = 0
            self._subgoal_turn_exhausted_target_name = None
            return None
        if self._subgoal_turning and self._subgoal_turn_steps >= self.fdm_cfg.subgoal_turn_max_steps:
            self._subgoal_turning = False
            self._subgoal_turn_steps = 0
            self._subgoal_turn_exhausted_target_name = target_name
            return None
        self._subgoal_turning = True
        self._subgoal_turn_steps += 1
        self._debug_info.update(
            {
                "target_yaw_err": yaw_error,
                "subgoal_turning": self._subgoal_turning,
                "subgoal_turn_steps": self._subgoal_turn_steps,
            }
        )
        wz_limit = min(self.waypoint_cfg.max_wz, 0.66)
        wz = max(-wz_limit, min(wz_limit, 1.2 * yaw_error))
        self._resample_next[:] = True
        self._last_plan.zero_()
        return VelocityCommand(0.0, 0.0, wz), f"subgoal_turn={target_name} yaw_err={yaw_error:.2f}"

    def _advance_reached_waypoints(self, robot_pose: Pose2D) -> tuple[bool, str | None]:
        skip_to_idx = self._path_progress_index(robot_pose)
        advanced_names: list[str] = []
        while self.step_index <= skip_to_idx and self.step_index < len(self.steps):
            if self.step_index == len(self.steps) - 1:
                break
            step = self.steps[self.step_index]
            if step.kind != "walk_to":
                break
            advanced_names.append(step.node_id)
            self.step_index += 1
            self._resample_next[:] = True
            self._subgoal_turning = False
            self._subgoal_turn_steps = 0
            self._subgoal_turn_target_name = None
            self._subgoal_turn_exhausted_target_name = None
        if not advanced_names:
            return False, None
        return True, "reached " + ",".join(advanced_names)

    def _path_progress_index(self, robot_pose: Pose2D) -> int:
        """Return the last non-final path index that the robot has effectively passed."""
        if self.step_index >= len(self.steps) - 1:
            return self.step_index - 1

        best_idx = self.step_index - 1
        best_progress = -1.0
        for idx in range(self.step_index, len(self.steps) - 1):
            step = self.steps[idx]
            next_step = self.steps[idx + 1]
            if step.kind != "walk_to" or next_step.kind != "walk_to":
                break
            along, lateral = _project_on_segment(robot_pose, step.pose, next_step.pose)
            lateral_score = lateral / max(self.fdm_cfg.pass_tolerance, 1.0e-6)
            progress_score = idx + max(0.0, min(1.0, along))
            if lateral_score <= 1.0 and progress_score > best_progress:
                best_progress = progress_score
            else:
                continue
            if along >= self.fdm_cfg.progress_margin:
                best_idx = max(best_idx, idx)
            if along >= 1.0 - self.fdm_cfg.progress_margin:
                best_idx = max(best_idx, min(idx + 1, len(self.steps) - 2))
        return best_idx

    def _has_progressed_past(self, robot_pose: Pose2D, idx: int) -> bool:
        step = self.steps[idx]
        if self._reached(robot_pose, step.pose, tolerance=self._target_tolerance(idx)):
            return True
        prev_pose = self.steps[idx - 1].pose if idx > 0 and self.steps[idx - 1].kind == "walk_to" else None
        next_pose = self.steps[idx + 1].pose if idx + 1 < len(self.steps) and self.steps[idx + 1].kind == "walk_to" else None
        if prev_pose is not None:
            along, lateral = _project_on_segment(robot_pose, prev_pose, step.pose)
            if along >= 1.0 - self.fdm_cfg.progress_margin and lateral <= self.fdm_cfg.pass_tolerance:
                return True
        if next_pose is not None:
            along, lateral = _project_on_segment(robot_pose, step.pose, next_pose)
            if along >= self.fdm_cfg.progress_margin and lateral <= self.fdm_cfg.pass_tolerance:
                return True
        return False

    def _directional_subgoal(self, robot_pose: Pose2D) -> tuple[Pose2D, str] | None:
        if self.step_index >= len(self.steps):
            return None
        target_idx = self.step_index
        accumulated = 0.0
        prev = robot_pose
        for idx in range(self.step_index, len(self.steps)):
            step = self.steps[idx]
            if step.kind != "walk_to":
                break
            accumulated += _distance(prev, step.pose)
            target_idx = idx
            prev = step.pose
            if accumulated >= self.fdm_cfg.lookahead_distance:
                break
        target_step = self.steps[target_idx]
        if self.fdm_cfg.face_subgoal:
            yaw = _heading(robot_pose, target_step.pose)
            if target_idx == len(self.steps) - 1:
                target_pose = Pose2D(target_step.pose.x, target_step.pose.y, yaw)
            else:
                target_pose = self._forward_carrot(robot_pose, target_step.pose, yaw)
            return target_pose, target_step.node_id
        yaw = target_step.pose.yaw
        return Pose2D(target_step.pose.x, target_step.pose.y, yaw), target_step.node_id

    def _forward_carrot(self, robot_pose: Pose2D, raw_target: Pose2D, yaw: float) -> Pose2D:
        distance = _distance(robot_pose, raw_target)
        if distance >= self.fdm_cfg.min_forward_carrot:
            return Pose2D(raw_target.x, raw_target.y, yaw)
        forward = self.fdm_cfg.min_forward_carrot
        return Pose2D(
            robot_pose.x + forward * torch.cos(torch.tensor(yaw, device=self.device)).item(),
            robot_pose.y + forward * torch.sin(torch.tensor(yaw, device=self.device)).item(),
            yaw,
        )

    def _path_direction_yaw(self, robot_pose: Pose2D, target_idx: int) -> float:
        anchor = robot_pose
        if target_idx > self.step_index and self.steps[target_idx - 1].kind == "walk_to":
            anchor = self.steps[target_idx - 1].pose
        target = self.steps[target_idx].pose
        return torch.atan2(
            torch.tensor(target.y - anchor.y, device=self.device),
            torch.tensor(target.x - anchor.x, device=self.device),
        ).item()


def _distance(a: Pose2D, b: Pose2D) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    return float((dx * dx + dy * dy) ** 0.5)


def _heading(a: Pose2D, b: Pose2D) -> float:
    return float(torch.atan2(torch.tensor(b.y - a.y), torch.tensor(b.x - a.x)).item())


def _wrap_to_pi(angle: float) -> float:
    while angle > 3.141592653589793:
        angle -= 6.283185307179586
    while angle < -3.141592653589793:
        angle += 6.283185307179586
    return angle


def _tracking_command(robot_pose: Pose2D, target_pose: Pose2D, cfg: WaypointExecutorConfig) -> VelocityCommand:
    dx_world = target_pose.x - robot_pose.x
    dy_world = target_pose.y - robot_pose.y
    c = cos(robot_pose.yaw)
    s = sin(robot_pose.yaw)
    dx_body = c * dx_world + s * dy_world
    dy_body = -s * dx_world + c * dy_world
    distance = (dx_world * dx_world + dy_world * dy_world) ** 0.5
    speed_scale = min(1.0, max(0.2, distance / max(cfg.slow_radius, 1e-6)))

    heading_to_target = torch.atan2(torch.tensor(dy_world), torch.tensor(dx_world)).item()
    heading_err = _wrap_to_pi(heading_to_target - robot_pose.yaw)

    vx = _clamp(cfg.k_vx * dx_body * speed_scale, -cfg.max_vx, cfg.max_vx)
    vy = _clamp(cfg.k_vy * dy_body * speed_scale, -cfg.max_vy, cfg.max_vy)
    wz = _clamp(cfg.k_wz * heading_err, -cfg.max_wz, cfg.max_wz)
    return VelocityCommand(vx=vx, vy=vy, wz=wz)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _project_on_segment(point: Pose2D, start: Pose2D, end: Pose2D) -> tuple[float, float]:
    sx, sy = start.x, start.y
    ex, ey = end.x, end.y
    px, py = point.x, point.y
    vx, vy = ex - sx, ey - sy
    wx, wy = px - sx, py - sy
    denom = vx * vx + vy * vy
    if denom <= 1e-8:
        return 0.0, _distance(point, start)
    t_raw = (wx * vx + wy * vy) / denom
    t = max(0.0, min(1.0, t_raw))
    proj = Pose2D(sx + t * vx, sy + t * vy, 0.0)
    return float(t_raw), _distance(point, proj)
