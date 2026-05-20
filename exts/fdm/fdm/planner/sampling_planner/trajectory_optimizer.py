# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import numpy as np
import os
import pickle
import subprocess
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from typing import TYPE_CHECKING

from isaaclab.envs import ManagerBasedRLEnv

from nav_suite.terrain_analysis import TerrainAnalysis

from fdm import FDM_DATA_DIR
from fdm.env_cfg import TERRAIN_ANALYSIS_CFG
from .robot_shape import get_robot_shape
from .trajectory_optimizer_cfg import ActionCfg, RobotCfg, TrajectoryOptimizerCfg
from .trajectory_optimizer_mbrl import BatchedICEMOptimizer, BatchedMPPIOptimizer
from .utils import (
    TraversabilityFilter,
    cosine_distance,
    get_non_zero_action_length,
    get_se2,
    get_x_y_yaw,
    smallest_angle,
    state_history_transformer,
)

if TYPE_CHECKING:
    from fdm.model import FDMModel

class SimpleSE2TrajectoryOptimizer:
    def __init__(
        self,
        action_cfg: ActionCfg,
        robot_cfg: RobotCfg,
        optim: BatchedICEMOptimizer | BatchedMPPIOptimizer,
        to_cfg: TrajectoryOptimizerCfg,
        device: torch.device,
    ):
        """
        Initializes the SimpleSE2TrajectoryOptimizer with the given configurations and device.

        Args:
            action_cfg (ActionCfg): Configuration for the action space.
            robot_cfg (RobotCfg): Configuration for the robot footprint.
            to_cfg (TrajectoryOptimizerCfg): Configuration for the trajectory optimizer.
            optim (TODO): Underlying Black box optimizer
            device (torch.device): The device (CPU/GPU) on which to perform the computations.
        """

        self.action_cfg  = ActionCfg(
        action_dim=3,
        traj_dim=10,   # 按你的 horizon 改
        lower_bound=[-0.1, -0.1, -1.0],
        upper_bound=[1.5,  0.1,  1.0],
        )
        self.to_cfg = to_cfg
        self.device = device
        self.frame_id = "odom"
        self.fatal_xy, self.risky_xy, self.cautious_xy = get_robot_shape(robot_cfg, device)

        # fdm parameters
        self.fdm_model: FDMModel | None = None
        self.terrain_analysis: TerrainAnalysis | None = None

        # Initialize Optimizer
        self.optim = optim
        self.previous_solution = torch.zeros(
            self.optim.batch_size, self.optim.planning_horizon, self.optim.action_dimension, device=self.device
        )

        if self.to_cfg.states_cost_w_cost_map:
            weight_file = subprocess.getoutput(
                'echo "' + os.path.join(FDM_DATA_DIR, "Traversability-Model", "weights.dat") + '"'
            )
            with open(weight_file, "rb") as file:
                weights = pickle.load(file)
            self.traversability_filter = TraversabilityFilter(
                weights["conv1.weight"], weights["conv2.weight"], weights["conv3.weight"], weights["conv_final.weight"]
            )
            self.traversability_filter.to(self.device).eval()

        # Set objective function
        if self.to_cfg.n_step_fwd:
            self.func = self.b_obj_func_N_step
        else:
            self.func = self.b_obj_func

        self.debug_info = {}
        self._cvae_mean_buffer: list[torch.Tensor] = []
        self._cvae_target_buffer: list[torch.Tensor] = []
        self._cvae_context_buffer: list[torch.Tensor] = []
        self._cvae_goal_state_buffer: list[torch.Tensor] = []
        self._cvae_outcome_success_buffer: list[torch.Tensor] = []
        self._cvae_risk_max_buffer: list[torch.Tensor] = []
        self._cvae_risk_sum_buffer: list[torch.Tensor] = []
        self._cvae_success_mask_buffer: list[torch.Tensor] = []
        self._cvae_sample_weight_buffer: list[torch.Tensor] = []
        self._last_cvae_context: torch.Tensor | None = None
        self._cvae_samples_since_flush = 0
        # latest rollout in local/base frame for running terrain-aware costs
        self.latest_local_states: torch.Tensor | None = None

    def _build_cvae_context(self, env_ids: torch.Tensor | list[int] | slice | None = None) -> torch.Tensor | None:
        """Build flattened conditioning context for the CVAE sampler from planner observations."""
        def select_context_tensor(tensor: torch.Tensor) -> torch.Tensor:
            if env_ids is None or isinstance(env_ids, slice):
                return tensor
            env_count = len(env_ids)
            if tensor.shape[0] == env_count:
                return tensor
            return tensor[env_ids]

        if "cvae_context" in self.obs and torch.is_tensor(self.obs["cvae_context"]):
            context = select_context_tensor(self.obs["cvae_context"])
        else:
            context_keys = [
                "goal",
                "proprio",
                "proprioception",
                "proprio_obs",
                "history",
                "state",
                "states",
            ]
            context_parts = []
            for key in context_keys:
                if key not in self.obs or not torch.is_tensor(self.obs[key]):
                    continue
                tensor = self.obs[key]
                if tensor.dtype not in (torch.float16, torch.float32, torch.float64):
                    tensor = tensor.float()
                tensor = select_context_tensor(tensor)
                context_parts.append(tensor.reshape(tensor.shape[0], -1))
            if len(context_parts) == 0:
                return None
            context = torch.cat(context_parts, dim=-1)

        return context.to(self.device)

    def _select_planner_batch_tensor(
        self,
        tensor: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Select planner observations whether they are stored as full-env or local-batch tensors."""
        if tensor.shape[0] == batch_size:
            return tensor.to(device)
        if self.env_ids is None or isinstance(self.env_ids, slice):
            return tensor[:batch_size].to(device)
        if isinstance(self.env_ids, torch.Tensor):
            env_idx = self.env_ids.to(device=tensor.device, dtype=torch.long)
        else:
            env_idx = torch.tensor(self.env_ids, device=tensor.device, dtype=torch.long)
        return tensor[env_idx].to(device)

    def set_fdm_classes(self, fdm_model: FDMModel, env: ManagerBasedRLEnv):
        self.fdm_model = fdm_model
        self.terrain_analysis = TerrainAnalysis(cfg=TERRAIN_ANALYSIS_CFG, scene=env.scene)
        self.terrain_analysis.analyse()
        self.height_scan_resolution = getattr(env.scene.sensors["env_sensor"].cfg.pattern_cfg, "resolution", 1.0)
        self.height_scan_size = getattr(env.scene.sensors["env_sensor"].cfg.pattern_cfg, "size", (10.0, 10.0))
        self.height_scan_offset = env.scene.sensors["env_sensor"].cfg.offset.pos

    ###
    # Operations
    ###

    def plan(
        self, obs: dict, env_ids: torch.Tensor | None = None, return_states: bool = True
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """
        Initializes the observation dictionary with default values for planning.

        Args:
            obs (dict): A dictionary containing the following key-value pairs:
                - "goal": (torch.tensor, shape:=(BS, 3)): representing the goal with (x,y,yaw) in the odom frame.
                - "resample_population": bool: If the population should be resampled
                - "start": (torch.tensor, shape:=(BS, 3)): representing the start with (x,y,yaw) in the odom frame.
            env_ids: The environment ids for which to plan.
        Returns:
            torch.Tensor: The planned trajectory with shape (BS, TRAJ_LENGTH, STATE_DIM).
            torch.Tensor: The planned velocity with shape (BS, TRAJ_LENGTH, CONTROL_DIM).

        """
        # save obs for later use
        self.obs = obs

        if env_ids is None:
            BS = self.obs["start"].shape[0]
            self.env_ids = slice(None)
        else:
            BS = len(env_ids)
            self.env_ids = env_ids
            assert BS != 0, "No environments to plan for. This case should be handled by the planner."

        # MPPI
        population = None
        resample_env_ids = None

        # Reset - Only needed for MPPI
        if torch.any(self.obs["resample_population"]):
            # get resample environments
            resample_env_ids = torch.where(self.obs["resample_population"])[0].tolist()
            self.optim.reset(resample_env_ids)
        self._last_cvae_context = self._build_cvae_context(self.env_ids)

        best_population, self.var = self.optim.optimize(
            obj_fun=self.func,
            env_ids=self.env_ids,
            x0=population,
            x0_env_ids=resample_env_ids,
            var0=None,
            callback=self.logging_callback,
            cvae_context=self._last_cvae_context,
        )

        if return_states:
            states = self.func(best_population[None], only_rollout=True)
        else:
            states = None

        return states, best_population

    ###
    # FDM functions
    ###

    def b_obj_func_N_step(
        self,
        population: torch.Tensor,
        only_rollout: bool = False,
        control_mode: str | None = None,
        env_ids: list[int] | None = None,
    ) -> torch.Tensor:
        """
        Objective function called by optimizer.
        We dynamicially allocate everything given that the population can grow or shrink
        """
        NR_TRAJ = population.shape[0]
        BS = population.shape[1]  # equal to the number of environments that have to be replanned
        TRAJ_LENGTH = population.shape[2]

        # override env_ids when given
        if env_ids is not None:
            self.env_ids = env_ids

        start_state = self.get_start_state(BS, NR_TRAJ).clone()

        if control_mode is None:
            control_mode = self.to_cfg.control

        if control_mode == "velocity_control":
            # Each population / action is given in the base frame of the robot
            population = population.permute(1, 0, 2, 3).contiguous()  # BS, NR_TRAJ, TRAJ_LENGTH, CONTROL_DIM

            # If the actions is small make the robot stand
            if self.to_cfg.set_actions_below_threshold_to_0:
                m_vel_lin = torch.norm(population[:, :, :, :2], p=2, dim=3) < self.to_cfg.vel_limit_lin
                m_vel_ang = torch.abs(population[:, :, :, 2]) < self.to_cfg.vel_limit_ang
                m_vel_lin = m_vel_lin[:, :, :, None].repeat(1, 1, 1, 3)
                m_vel_lin[:, :, :, 2] = False
                m_vel_ang = m_vel_ang[:, :, :, None].repeat(1, 1, 1, 3)
                m_vel_ang[:, :, :, :2] = False

                if m_vel_lin.sum() > 0:
                    population[m_vel_lin] = 0
                if m_vel_ang.sum() > 0:
                    population[m_vel_ang] = 0

            # Integrate the velocity actions to positions
            actions = population * self.to_cfg.dt

            # Cumsum is an inplace operation therefore the clone is necesasry
            cummulative_yaw = actions.clone()[:, :, :, -1].cumsum(2)

            # We need to take the non-linearity by the rotation into account
            r_vec1 = torch.stack([torch.cos(cummulative_yaw), torch.sin(cummulative_yaw)], dim=3)
            r_vec2 = torch.stack([-torch.sin(cummulative_yaw), torch.cos(cummulative_yaw)], dim=3)

            so2 = torch.stack([r_vec1, r_vec2], dim=4)

            # Move the rotation in time and fill first timestep with identity
            so2 = torch.roll(so2, shifts=1, dims=2)
            so2[:, :, 0, :, :] = torch.eye(2, device=so2.device)[None, None].repeat(BS, NR_TRAJ, 1, 1)

            actions_local_frame = so2.contiguous().reshape(-1, 2, 2) @ actions[:, :, :, :2].contiguous().reshape(
                -1, 2, 1
            )
            actions_local_frame = actions_local_frame.contiguous().reshape(BS, NR_TRAJ, TRAJ_LENGTH, 2)
            cumulative_position = actions_local_frame.cumsum(dim=2)
            local_states = torch.cat([cumulative_position, cummulative_yaw[:, :, :, None]], dim=3)

            # Transform the states from the current base frame to the odom frame
            se2_odom_base = get_se2(start_state[:, :, None, :].repeat(1, 1, TRAJ_LENGTH, 1))
            se2_base_points = get_se2(local_states)
            se2_odom_points = se2_odom_base @ se2_base_points
            states = get_x_y_yaw(se2_odom_points)

            # save local rollout for terrain-aware running cost
            self.latest_local_states = local_states.clone()

        elif control_mode == "fdm" or control_mode == "fdm_baseline":

            # check if FDM model is provided
            assert self.fdm_model is not None, "FDM model is not set"

            # Each population / action is given in the base frame of the robot
            population = population.permute(1, 0, 2, 3).contiguous()  # BS, NR_TRAJ, TRAJ_LENGTH, CONTROL_DIM
            if getattr(self.to_cfg, "debug", False):
                pop_dbg = population.detach()

                # 看 env0 的所有轨迹在第1个控制步的动作分布
                u0 = pop_dbg[0, :, 0, :]  # (NR_TRAJ, act_dim)

            #    print("\n[SAMPLE-DBG] =====================================", flush=True)
            #    print(f"[SAMPLE-DBG] population shape={tuple(pop_dbg.shape)}", flush=True)
            #    print(f"[SAMPLE-DBG] first-step action mean={u0.mean(dim=0).cpu().tolist()}", flush=True)
            #    print(f"[SAMPLE-DBG] first-step action std ={u0.std(dim=0).cpu().tolist()}", flush=True)
            #    print(f"[SAMPLE-DBG] first-step action min ={u0.min(dim=0).values.cpu().tolist()}", flush=True)
            #    print(f"[SAMPLE-DBG] first-step action max ={u0.max(dim=0).values.cpu().tolist()}", flush=True)

                q = torch.quantile(
                    u0,
                    torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99], device=u0.device),
                    dim=0,
                )
            #    print(f"[SAMPLE-DBG] first-step action quantiles(1,10,50,90,99%)=\n{q.cpu()}", flush=True)

            # get initial states
            if isinstance(self.env_ids, slice):
                env_idx = torch.arange(BS)
            elif isinstance(self.env_ids, torch.Tensor):
                env_idx = torch.arange(self.env_ids.shape[0])
            else:
                env_idx = torch.arange(len(self.env_ids))

            # init final output buffers
            num_envs = len(env_idx)
            if control_mode == "fdm":
                local_states = torch.zeros((num_envs, NR_TRAJ, TRAJ_LENGTH, 4), device=self.fdm_model.device)
                energy_traj = torch.zeros((num_envs, NR_TRAJ, TRAJ_LENGTH, 1), device=self.fdm_model.device)
            else:
                local_states = torch.zeros((num_envs, NR_TRAJ, TRAJ_LENGTH, 2), device=self.fdm_model.device)
            if self.fdm_model.cfg.unified_failure_prediction:
                collision_prob_traj = torch.zeros((num_envs, NR_TRAJ), device=self.fdm_model.device)
            else:
                collision_prob_traj = torch.zeros((num_envs, NR_TRAJ, TRAJ_LENGTH), device=self.fdm_model.device)

            planner_states = self._select_planner_batch_tensor(self.obs["states"], num_envs, self.device)
            planner_proprio = self._select_planner_batch_tensor(self.obs["proprio_obs"], num_envs, self.device)
            planner_extero = (
                self._select_planner_batch_tensor(self.obs["extero_obs"], num_envs, self.device)
                if "extero_obs" in self.obs
                else None
            )
            planner_add_extero = (
                self._select_planner_batch_tensor(self.obs["add_extero_obs"], num_envs, self.device)
                if "add_extero_obs" in self.obs
                else None
            )

            # process in mini-batches due to high memory requirements
            num_env_per_batch = math.ceil(max(self.to_cfg.batch_size / NR_TRAJ, 1))
            for mini_batch_idx in range(math.ceil(num_envs / num_env_per_batch)):
                curr_idx_range = [
                    num_env_per_batch * mini_batch_idx,
                    min(num_env_per_batch * (mini_batch_idx + 1), num_envs),
                ]
                curr_env_idx = env_idx[curr_idx_range[0] : curr_idx_range[1]]
                curr_batch_size = len(curr_env_idx)

                # get state history transformed into local frame
                state_history = state_history_transformer(
                    planner_states,
                    curr_env_idx,
                    self.fdm_model.cfg.history_length,
                    self.fdm_model.cfg.exclude_state_idx_from_input,
                ).to(self.device)

                if control_mode == "fdm":
                    pass

                # make predictions
                model_in = (
                    state_history.unsqueeze(1)
                    .repeat(1, NR_TRAJ, 1, 1)
                    .view(curr_batch_size * NR_TRAJ, state_history.shape[1], state_history.shape[2]),
                    (
                        planner_proprio[curr_env_idx]
                        .to(self.device)
                        .unsqueeze(1)
                        .repeat(1, NR_TRAJ, 1, 1)
                        .view(curr_batch_size * NR_TRAJ, *(planner_proprio.shape[1:]))
                    ),
                    (
                        planner_extero[curr_env_idx]
                        .type(torch.float32)
                        .to(self.device)
                        .unsqueeze(1)
                        .repeat(1, NR_TRAJ, *([1] * (planner_extero.dim() - 1)))
                        .view(curr_batch_size * NR_TRAJ, *(planner_extero.shape[1:]))
                        if planner_extero is not None
                        else torch.zeros(1)
                    ),
                    population[curr_idx_range[0] : curr_idx_range[1]].view(curr_batch_size * NR_TRAJ, TRAJ_LENGTH, -1),
                    (
                        planner_add_extero[curr_env_idx]
                        .type(torch.float32)
                        .to(self.device)
                        .unsqueeze(1)
                        .repeat(1, NR_TRAJ, *([1] * (planner_add_extero.dim() - 1)))
                        .view(curr_batch_size * NR_TRAJ, *(planner_add_extero.shape[1:]))
                        if planner_add_extero is not None
                        else torch.zeros(1)
                    ),
                )
                if mini_batch_idx == 0:
                    sh, prop, ext, act, add_ext = model_in
                # make prediction
                with torch.no_grad():
                    if control_mode == "fdm":
                        model_out = self.fdm_model.forward(model_in)
                        curr_states = model_out[0]
                        curr_dynamic_collision_prob_traj = model_out[1]
                        curr_energy_traj = model_out[2]
                        if len(model_out) > 3:
                            curr_geometric_collision_prob_traj = model_out[3]
                            curr_collision_prob_traj = torch.maximum(
                                curr_dynamic_collision_prob_traj,
                                curr_geometric_collision_prob_traj,
                            )
                        else:
                            curr_collision_prob_traj = curr_dynamic_collision_prob_traj
                    else:
                        model_out = self.fdm_model.forward(model_in)
                        curr_states = model_out[0]
                        curr_collision_prob_traj = model_out[1]
                        if self.fdm_model.cfg.unified_failure_prediction:
                            curr_collision_prob_traj = torch.max(curr_collision_prob_traj, dim=-1)[0]
                if getattr(self.to_cfg, "debug", False) and mini_batch_idx == 0:
                    x = curr_collision_prob_traj.detach()

                #    print("\n[RISK-DBG] =====================================", flush=True)
                #    print(f"[RISK-DBG] control_mode={control_mode}", flush=True)
                #    print(f"[RISK-DBG] unified_failure_prediction={self.fdm_model.cfg.unified_failure_prediction}",
                 #         flush=True)
                 #   print(f"[RISK-DBG] curr_collision_prob_traj.shape={tuple(x.shape)}", flush=True)
                 #   print(
                 #       f"[RISK-DBG] min={x.min().item():.6f} max={x.max().item():.6f} mean={x.mean().item():.6f}",
                 #       flush=True,
                 #   )

                    flat = x.reshape(-1)
                    q = torch.quantile(flat, torch.tensor([0.5, 0.9, 0.99], device=flat.device))

                # reshape states back to BS, NR_TRAJ
                local_states[curr_idx_range[0] : curr_idx_range[1]] = curr_states.view(
                    curr_batch_size, NR_TRAJ, *(curr_states.shape[1:])
                )
                collision_prob_traj[curr_idx_range[0] : curr_idx_range[1]] = curr_collision_prob_traj.view(
                    curr_batch_size, NR_TRAJ, *(curr_collision_prob_traj.shape[1:])
                )
                if control_mode == "fdm":
                    energy_traj[curr_idx_range[0] : curr_idx_range[1]] = curr_energy_traj.view(
                        curr_batch_size, NR_TRAJ, *(curr_energy_traj.shape[1:])
                    )

            if control_mode == "fdm":
                # transform the orientation encoding to a yaw angle
                local_states[:, :, :, 2] = torch.atan2(local_states[..., 2], local_states[..., 3])
                local_states = local_states[..., :3]
            else:
                # append a zero yaw angle to the states
                local_states = torch.cat([local_states, torch.zeros_like(local_states[..., 0])[..., None]], dim=-1)

            # transform states into odom frame
            se2_odom_base = get_se2(start_state[:, :, None, :].repeat(1, 1, TRAJ_LENGTH, 1))
            se2_base_points = get_se2(local_states)
            se2_odom_points = se2_odom_base @ se2_base_points
            states = get_x_y_yaw(se2_odom_points)

            # save local rollout for terrain-aware running cost
            self.latest_local_states = local_states.clone()

            # Integrate the velocity actions to positions for loss calculation
            actions = population * self.to_cfg.dt

        elif control_mode == "position_control":  # noqa: R506
            raise ValueError(
                "Not correctly implemented in the cost function handling the yaw actions forward, sidward motion"
                " correctly."
            )

        else:
            raise ValueError(f"Control mode {control_mode} not supported")

        if only_rollout:
            return states

        # calculate the running cost
        running_cost = self.states_cost(states.clone(), actions)
        if self.to_cfg.debug:
            self.debug_info["states_running_cost"] = running_cost.clone()

        running_cost = running_cost.mean(dim=2)
        terminal_cost = self.terminal_cost(states[:, :, -1])

        total_cost = running_cost + terminal_cost

        if self.to_cfg.control == "fdm" or self.to_cfg.control == "fdm_baseline":
            collision_cost = self.collision_cost(states, collision_prob_traj)
            self.debug_info["collision_cost"] = collision_cost.clone()

            total_cost += collision_cost

        elif self.to_cfg.states_cost_w_cost_map:
            assert self.to_cfg.control != "fdm_baseline", "The height scan is not available for the baseline model"
            self.curr_cost_map_cost = self.cost_map_cost(local_states)
            self.debug_info["cost_map_cost"] = self.curr_cost_map_cost.clone()

            total_cost += self.curr_cost_map_cost

        if self.to_cfg.debug:
            self.debug_info["terminal_cost"] = terminal_cost.clone()
            self.debug_callback(states, total_cost)

        # make states and cost accessible for visualization
        self.states = states
        self.total_cost = total_cost
        self.population = population.permute(1, 0, 2, 3)
        if self.to_cfg.control == "fdm" or self.to_cfg.control == "fdm_baseline":
            if self.fdm_model.cfg.unified_failure_prediction:
                peak_risk = collision_prob_traj
            else:
                peak_risk = collision_prob_traj.max(dim=-1).values  # (BS, NR_TRAJ)

            # env0 先看
            b = 0
            best_idx = torch.argmin(total_cost[b])

            q = torch.quantile(
                peak_risk[b],
                torch.tensor([0.5, 0.9, 0.99], device=peak_risk.device)
            )

            #print(
            #    f"[BEST-RISK-DBG] env={b} "
            #    f"best_idx={best_idx.item()} "
            #    f"best_peak_risk={peak_risk[b, best_idx].item():.6f} "
            #    f"best_total_cost={total_cost[b, best_idx].item():.6f} "
            #    f"best_terminal={terminal_cost[b, best_idx].item():.6f}",
            #    flush=True,
            #)

        return -total_cost.T  # N_traj, BS

    def b_obj_func(
        self, population: torch.Tensor, only_rollout: bool = False, iteration: int | None = None
    ) -> torch.Tensor:
        """
        Objective function called by optimizer.
        """
        NR_TRAJ = population.shape[0]
        BS = population.shape[1]
        TRAJ_LENGTH = population.shape[2]

        state = self.get_start_state(BS, NR_TRAJ)
        STATE_DIM = state.shape[-1]

        if not only_rollout:
            running_cost = torch.zeros((BS, NR_TRAJ), device=self.device)

        if self.to_cfg or only_rollout:
            states = torch.zeros((BS, NR_TRAJ, TRAJ_LENGTH, STATE_DIM), device=self.device)

        action = population.permute(1, 0, 2, 3).contiguous()  # BS, NR_TRAJ, TRAJ_LENGTH, CONTROL_DIM

        for i in range(TRAJ_LENGTH):
            if not only_rollout:
                running_cost += self.states_cost(state[:, :, None], action[:, :, i, :][:, :, None])[:, :, 0]

            if self.to_cfg or only_rollout:
                states[:, :, i, :] = state.clone()

            state = self.forward_dynamics(state, action[:, :, i, :])

        if self.to_cfg.debug:
            self.debug_callback(states, running_cost)

        if only_rollout:
            return states

        return -((running_cost + self.terminal_cost(state)).T)

    ###
    # Cost functions
    ###

    def states_cost(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Evaluates state cost of a sequence of states.

        Args:
            states (torch.Tensor, dtype=torch.float32, shape=(BS, NR_TRAJ, TRAJ_LENGTH, STATE_DIM)): Sequence of states
            actions (torch.Tensor, dtype=torch.float32, shape=(BS, NR_TRAJ, TRAJ_LENGTH, ACTION_DIM)): Sequence of actions

        Returns:
            (torch.Tensor, dtype=torch.float32, shape=(BS, NR_TRAJ, TRAJ_LENGTH)): Sequence of costs per state
        """
        # ------------------------------------------------------------------
        # basic control effort
        # ------------------------------------------------------------------
        control_effort_trans_forward = torch.abs(actions[:, :, :, 0]) * self.to_cfg.state_cost_w_action_trans_forward

        side_weight = getattr(
            self.to_cfg,
            "state_cost_w_action_trans_side_biped",
            self.to_cfg.state_cost_w_action_trans_side,
        )
        control_effort_trans_side = torch.abs(actions[:, :, :, 1]) * side_weight

        control_effort_rot = torch.abs(actions[:, :, :, 2]) * self.to_cfg.state_cost_w_action_rot

        # ------------------------------------------------------------------
        # early goal reaching mask
        # ------------------------------------------------------------------
        position_offset = torch.norm(states[:, :, :, :2] - self.obs["goal"][self.env_ids, None, None, :2], dim=3)
        goal_yaw = self.obs["goal"][self.env_ids, None, None, 2].repeat(1, states.shape[1], states.shape[2])
        heading = cosine_distance(states[:, :, :, 2], goal_yaw) / 2

        reached_mask = (position_offset < self.to_cfg.state_cost_early_goal_distance_offset) * (
            heading < self.to_cfg.state_cost_early_goal_heading_offset
        )

        # ------------------------------------------------------------------
        # velocity tracking
        # ------------------------------------------------------------------
        velocity_tracking_cost = (
            torch.abs(torch.norm(actions[:, :, :, :2], p=2, dim=3) - self.to_cfg.state_cost_desired_velocity)
            * self.to_cfg.state_cost_velocity_tracking
        )

        # ------------------------------------------------------------------
        # running heading-to-goal alignment
        # ------------------------------------------------------------------
        heading_running_cost = self.heading_running_cost(states)

        # ------------------------------------------------------------------
        # smoothness / curvature suppression
        # ------------------------------------------------------------------
        smoothness_cost = self.smoothness_cost(actions)

        # ------------------------------------------------------------------
        # stair alignment cost (only active when local rollout + extero exists)
        # ------------------------------------------------------------------
        stair_alignment_cost = torch.zeros_like(control_effort_trans_forward)
        if self.latest_local_states is not None and "extero_obs" in self.obs:
            if (
                self.latest_local_states.shape[0] == states.shape[0]
                and self.latest_local_states.shape[1] == states.shape[1]
                and self.latest_local_states.shape[2] == states.shape[2]
            ):
                stair_alignment_cost = self.stair_alignment_cost(self.latest_local_states, actions)

        # ------------------------------------------------------------------
        # optional yaw-rate change penalty (extra zig-zag suppression)
        # ------------------------------------------------------------------
        yaw_rate_change_cost = self.yaw_rate_change_cost(actions)
        near_obstacle_cost = torch.zeros_like(control_effort_trans_forward)
        near_obstacle_enabled = (
            float(getattr(self.to_cfg, "state_cost_w_near_obstacle_soft", 0.0)) > 0.0
            or float(getattr(self.to_cfg, "state_cost_w_near_obstacle_hard", 0.0)) > 0.0
        )
        if near_obstacle_enabled and self.latest_local_states is not None and "extero_obs" in self.obs:
            if (
                    self.latest_local_states.shape[0] == states.shape[0]
                    and self.latest_local_states.shape[1] == states.shape[1]
                    and self.latest_local_states.shape[2] == states.shape[2]
            ):
                near_obstacle_cost = self.near_obstacle_cost(self.latest_local_states)

        # ------------------------------------------------------------------
        # zero action-related costs if goal already reached
        # ------------------------------------------------------------------
        control_effort_trans_forward[reached_mask] = 0
        control_effort_trans_side[reached_mask] = 0
        control_effort_rot[reached_mask] = 0
        velocity_tracking_cost[reached_mask] = 0
        heading_running_cost[reached_mask] = 0
        smoothness_cost[reached_mask] = 0
        stair_alignment_cost[reached_mask] = 0
        yaw_rate_change_cost[reached_mask] = 0
        near_obstacle_cost[reached_mask] = 0
        # ------------------------------------------------------------------
        # early goal reward
        # ------------------------------------------------------------------
        early_goal_cost = -torch.ones_like(control_effort_trans_forward) * self.to_cfg.state_cost_w_early_goal_reaching
        early_goal_cost[~reached_mask] = 0

        # ------------------------------------------------------------------
        # early stopping reward
        # ------------------------------------------------------------------
        res = get_non_zero_action_length(actions)
        precentage_early_stopping = ((actions.shape[2] - (res + 1)) / actions.shape[2])[:, :, None].repeat(
            1, 1, actions.shape[2]
        )
        early_stopping_cost = -precentage_early_stopping * self.to_cfg.state_cost_w_early_stopping

        total_cost = (
            control_effort_trans_forward
            + control_effort_trans_side
            + control_effort_rot
            + early_goal_cost
            + early_stopping_cost
            + velocity_tracking_cost
            + heading_running_cost
            + smoothness_cost
            + yaw_rate_change_cost
            + near_obstacle_cost
        )


        return total_cost

    def terminal_cost(self, state: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the terminal state cost for a given state.

        Args:
            state (torch.Tensor, dtype=torch.float32, shape=(BS, NR_TRAJ, STATE_DIM)): The terminal state to evaluate.

        Returns:
            torch.Tensor: The calculated terminal cost for the given state, with shape (BS, NR_TRAJ).
        """
        # compute and save position offset for visualization
        if self.to_cfg.pos_error_3d and self.terrain_analysis is not None:
            goal_pos_idx = (
                (
                    self.obs["goal"][self.env_ids, :2]
                    - torch.tensor(
                        [self.terrain_analysis.mesh_dimensions[2], self.terrain_analysis.mesh_dimensions[3]],
                        device=self.obs["goal"].device,
                    )
                )
                / self.terrain_analysis.cfg.grid_resolution
            ).int()
            goal_pos_idx[:, 0] = torch.clamp(goal_pos_idx[:, 0], 0, self.terrain_analysis.height_grid.shape[0] - 1)
            goal_pos_idx[:, 1] = torch.clamp(goal_pos_idx[:, 1], 0, self.terrain_analysis.height_grid.shape[1] - 1)

            final_path_pos_idx = (
                (
                    state[:, :, :2]
                    - torch.tensor(
                        [self.terrain_analysis.mesh_dimensions[2], self.terrain_analysis.mesh_dimensions[3]],
                        device=self.obs["goal"].device,
                    )
                )
                / self.terrain_analysis.cfg.grid_resolution
            ).int()
            final_path_pos_idx[..., 0] = torch.clamp(
                final_path_pos_idx[..., 0], 0, self.terrain_analysis.height_grid.shape[0] - 1
            )
            final_path_pos_idx[..., 1] = torch.clamp(
                final_path_pos_idx[..., 1], 0, self.terrain_analysis.height_grid.shape[1] - 1
            )
            # flatten final path pos idx
            final_path_pos_idx = final_path_pos_idx.view(-1, 2)

            z_height_goal = self.terrain_analysis.height_grid[goal_pos_idx[:, 0], goal_pos_idx[:, 1]]
            z_height_final = self.terrain_analysis.height_grid[final_path_pos_idx[:, 0], final_path_pos_idx[:, 1]]
            z_height_final = z_height_final.view(state.shape[0], state.shape[1])

            self.position_offset = torch.norm(
                torch.concatenate([state[:, :, :2], z_height_final[:, :, None]], dim=2)
                - torch.concatenate([self.obs["goal"][self.env_ids, None, :2], z_height_goal[:, None, None]], dim=2),
                dim=2,
            )

        else:
            self.position_offset = torch.norm(state[:, :, :2] - self.obs["goal"][self.env_ids, None, :2], dim=2)

        # Original goal yaw error
        goal_yaw_error = smallest_angle(
            state[:, :, 2], self.obs["goal"][self.env_ids, None, 2].repeat(1, state.shape[1])
        )

        # Additional "face the goal approach direction" cost
        goal_vec = self.obs["goal"][self.env_ids, None, :2] - state[:, :, :2]
        goal_heading = torch.atan2(goal_vec[..., 1], goal_vec[..., 0])
        heading_to_goal_error = torch.abs(self._wrap_to_pi(goal_heading - state[:, :, 2]))

        w_goal_heading = getattr(self.to_cfg, "terminal_cost_w_heading_to_goal", 0.0)

        heading_term = (
            goal_yaw_error * self.to_cfg.terminal_cost_w_rot_error
            + heading_to_goal_error * w_goal_heading
        )
        res = self.position_offset * self.to_cfg.terminal_cost_w_position_error + heading_term

        if self.to_cfg.terminal_cost_use_threshold:
            m = self.position_offset < self.to_cfg.terminal_cost_distance_offset
            res[m] /= self.to_cfg.terminal_cost_close_reward

        self.pose_cost = res.clone()

        return res

    def collision_cost(self, states: torch.Tensor, collision_traj: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the collision cost for a given estimated trajectory.

        Args:
            states: The estimated trajectory to evaluate, shape (BS, NR_TRAJ, TRAJ_LENGTH, STATE_DIM).
            collision_traj: The collision probability to evaluate, shape (BS, NR_TRAJ, TRAJ_LENGTH).

        Returns:
            torch.Tensor: Collision cost, shape (BS, NR_TRAJ).
        """

        threshold = self.fdm_model.cfg.collision_threshold - self.to_cfg.collision_cost_safety_factor

        # ------------------------------------------------------------------
        # 1. Base collision cost: make sure final shape is (BS, NR_TRAJ)
        # ------------------------------------------------------------------
        if self.fdm_model.cfg.unified_failure_prediction:
            # If collision_traj is already (BS, NR_TRAJ), keep it.
            # If it is (BS, NR_TRAJ, T), reduce over horizon.
            if collision_traj.ndim == 3:
                base_prob = collision_traj.mean(dim=-1)
                high_risk = torch.any(collision_traj > threshold, dim=-1)
            else:
                base_prob = collision_traj
                high_risk = collision_traj > threshold

            cost = base_prob * self.to_cfg.collision_cost_traj_factor
            cost = cost.clone()
            cost[high_risk] += self.to_cfg.collision_cost_high_risk_factor

        else:
            high_risk = torch.any(collision_traj > threshold, dim=-1)
            cost = torch.sum(collision_traj * self.to_cfg.collision_cost_traj_factor, dim=-1)
            cost = cost.clone()
            cost[high_risk] += self.to_cfg.collision_cost_high_risk_factor

        # ------------------------------------------------------------------
        # 2. Neighbor spread cost
        # ------------------------------------------------------------------
        num_envs, num_trajectories, T, _ = states.shape

        num_neighbors_cfg = int(getattr(self.to_cfg, "num_neighbors", 0))
        neighbor_spread_weight = float(getattr(self.to_cfg, "collision_cost_neighbor_spread_weight", 1.0))

        # If neighbor spreading is disabled, directly return base cost.
        if num_neighbors_cfg <= 0 or num_trajectories <= 1 or neighbor_spread_weight == 0.0:
            self.collision_traj_cost = cost.clone()
            return cost

        cost_pre = cost.clone()

        for env_id in range(num_envs):
            flattened_trajectories = states[env_id, :, :, :2].reshape(num_trajectories, -1)

            traj_np = flattened_trajectories.detach().cpu().numpy().astype(np.float32, copy=False)

            distance_matrix = cdist(traj_np, traj_np, metric="euclidean")
            distance_matrix = distance_matrix.astype(np.float32, copy=False)

            n = distance_matrix.shape[0]
            k = min(num_neighbors_cfg, n - 1)

            if k <= 0:
                continue

            # Avoid selecting itself as neighbor
            np.fill_diagonal(distance_matrix, np.inf)

            # Find only top-k nearest neighbors instead of full argsort
            neighbors = np.argpartition(distance_matrix, kth=k - 1, axis=1)[:, :k]

            # Sort the selected k neighbors by distance
            row_idx = np.arange(n)[:, None]
            order = np.argsort(distance_matrix[row_idx, neighbors], axis=1)
            neighbors = neighbors[row_idx, order]

            neighbors_t = torch.as_tensor(neighbors, device=states.device, dtype=torch.long)
            distance_matrix_t = torch.as_tensor(distance_matrix, device=states.device, dtype=cost.dtype)

            row_idx_t = torch.arange(n, device=states.device)[:, None].expand(n, k)

            neighbor_cost = cost_pre[env_id][neighbors_t]  # (NR_TRAJ, k)
            neighbor_dist = distance_matrix_t[row_idx_t, neighbors_t]  # (NR_TRAJ, k)

            propagated = torch.sum(neighbor_cost / (neighbor_dist + 1e-2), dim=-1)

            cost[env_id] += neighbor_spread_weight * propagated

        self.collision_traj_cost = cost.clone()

        return cost

    def near_obstacle_cost(self, states: torch.Tensor) -> torch.Tensor:
        """
        Near-obstacle penalty from local height scan, computed on GPU by dilating the obstacle map.

        Args:
            states: local-frame states, shape (BS, NR_TRAJ, TRAJ_LENGTH, 3)

        Returns:
            cost: (BS, NR_TRAJ, TRAJ_LENGTH)
        """
        if "extero_obs" not in self.obs:
            return torch.zeros(states.shape[:3], device=states.device, dtype=states.dtype)

        # (BS, H, W)
        height_scan = self._select_planner_batch_tensor(
            self.obs["extero_obs"],
            states.shape[0],
            states.device,
        ).squeeze(1).float()
        BS, H, W = height_scan.shape

        # local trajectory positions -> map indices
        center = torch.tensor(
            [[
                H / 2 - self.height_scan_offset[1] / self.height_scan_resolution,
                W / 2 - self.height_scan_offset[0] / self.height_scan_resolution,
            ]],
            device=states.device,
            dtype=torch.float32,
        )

        path_idx = center[:, None, None, :] + (
                states[..., [1, 0]] / self.height_scan_resolution
        ) * torch.tensor([-1.0, 1.0], device=states.device)

        path_idx = path_idx.long()
        path_idx[..., 0] = torch.clamp(path_idx[..., 0], 0, H - 1)
        path_idx[..., 1] = torch.clamp(path_idx[..., 1], 0, W - 1)

        finite_height_scan = height_scan.clone()
        finite_height_scan[~torch.isfinite(finite_height_scan)] = torch.nan
        ground_percentile = float(getattr(self.to_cfg, "state_cost_ground_percentile", 0.20))
        obstacle_height_th = float(getattr(self.to_cfg, "state_cost_obstacle_height_th", 0.08))
        ground_ref = torch.nanquantile(
            finite_height_scan.reshape(BS, -1),
            ground_percentile,
            dim=1,
        ).view(BS, 1, 1)
        ground_ref = torch.nan_to_num(ground_ref, nan=0.0)

        # height_scan is pelvis-relative in lab. Use height above local ground so lower obstacles are retained.
        obstacle_map = (height_scan - ground_ref) > obstacle_height_th  # (BS, H, W)

        batch_idx = torch.arange(BS, device=states.device)[:, None, None].expand(
            BS, states.shape[1], states.shape[2]
        )

        soft_dist_th = float(getattr(self.to_cfg, "state_cost_near_obstacle_soft_th", 0.30))
        hard_dist_th = float(getattr(self.to_cfg, "state_cost_near_obstacle_hard_th", 0.15))
        w_soft = float(getattr(self.to_cfg, "state_cost_w_near_obstacle_soft", 3.0))
        w_hard = float(getattr(self.to_cfg, "state_cost_w_near_obstacle_hard", 12.0))

        obstacle = obstacle_map.to(dtype=height_scan.dtype).unsqueeze(1)
        soft_radius = max(int(math.ceil(soft_dist_th / self.height_scan_resolution)), 0)
        hard_radius = max(int(math.ceil(hard_dist_th / self.height_scan_resolution)), 0)

        soft_map = F.max_pool2d(obstacle, kernel_size=2 * soft_radius + 1, stride=1, padding=soft_radius).squeeze(1)
        hard_map = F.max_pool2d(obstacle, kernel_size=2 * hard_radius + 1, stride=1, padding=hard_radius).squeeze(1)

        soft_mask = soft_map[batch_idx, path_idx[..., 0], path_idx[..., 1]].to(torch.bool)
        hard_mask = hard_map[batch_idx, path_idx[..., 0], path_idx[..., 1]].to(torch.bool)

        cost = torch.zeros(states.shape[:3], device=states.device, dtype=states.dtype)

        cost[soft_mask] += w_soft
        cost[hard_mask] += w_hard

        return cost

    def cost_map_cost(self, states: torch.Tensor) -> torch.Tensor:
        """Cost based on cost map generated from the height scan

        Args:
            states: States of the sampled trajectories in local frame

        Returns
            cost: Cost of the applied filters for every path
        """
        all_height_scan = self._select_planner_batch_tensor(
            self.obs["extero_obs"],
            states.shape[0],
            states.device,
        ).squeeze(1)

        # handle the whole code batched due to memory limitations
        num_envs_per_batch = 20
        num_batches = math.ceil(states.shape[0] / num_envs_per_batch)

        cost = torch.zeros(states.shape[0], states.shape[1], device=states.device)

        for batch_idx in range(num_batches):
            curr_env_idx_range = torch.arange(
                num_envs_per_batch * batch_idx,
                min(num_envs_per_batch * (batch_idx + 1), states.shape[0]),
                device=states.device,
            )

            # get height-scan
            height_scan = all_height_scan[curr_env_idx_range].to(self.device)
            num_envs, grid_size_x, grid_size_y = height_scan.shape
            # get the tragversability map
            trav_map = torch.zeros_like(height_scan)
            trav_map[:, 3:-3, 3:-3] = self.traversability_filter(height_scan.to(torch.float32).unsqueeze(1)).squeeze(1)

            # get the indexes of the points of the path on the height-map
            # NOTE: the height map is oriented with the robot x axis in the y direction of the height map and the robot y axis in the x direction of the height map
            path_idx = torch.tensor(
                [[
                    grid_size_x / 2 - self.height_scan_offset[1] / self.height_scan_resolution,
                    grid_size_y / 2 - self.height_scan_offset[0] / self.height_scan_resolution,
                ]],
                device=states.device,
                dtype=torch.int32,
            ) + (states[curr_env_idx_range][..., [1, 0]] / self.height_scan_resolution).to(torch.int32) * torch.tensor(
                [-1, 1], device=states.device, dtype=torch.int32
            )

            # get risky coordinates for the robot
            alpha = states[curr_env_idx_range][:, :, :, 2, None].repeat(1, 1, 1, self.risky_xy.shape[0]) - torch.pi / 2
            cells_xy = self.risky_xy.clone()[None, None, None, :, :].repeat(num_envs, *states.shape[1:3], 1, 1)

            so2 = torch.zeros(
                (num_envs, *states.shape[1:3], self.risky_xy.shape[0], 2, 2), device=path_idx.device
            )
            so2[:, :, :, :, 0, 0] = torch.cos(alpha)
            so2[:, :, :, :, 1, 0] = torch.sin(alpha)
            so2[:, :, :, :, 0, 1] = -torch.sin(alpha)
            so2[:, :, :, :, 1, 1] = torch.cos(alpha)

            coordinates = torch.bmm(so2.reshape(-1, 2, 2), cells_xy.reshape(-1, 2)[:, :, None])[:, :, 0] + path_idx[
                :, :, :, None, :
            ].repeat(1, 1, 1, self.risky_xy.shape[0], 1).reshape(-1, 2)
            coordinates = coordinates.type(torch.long)

            # clip the idx to max indexes of the height map
            coordinates[:, 0] = torch.clamp(coordinates[:, 0], 0, grid_size_x - 1)
            coordinates[:, 1] = torch.clamp(coordinates[:, 1], 0, grid_size_y - 1)

            # Check all points of the robot shape
            env_idx_tensor = (
                torch.arange(num_envs, device=states.device)[:, None]
                .repeat(1, states.shape[1] * states.shape[2] * self.risky_xy.shape[0])
                .reshape(-1)
            )
            filter_idx = trav_map[env_idx_tensor, coordinates[:, 0], coordinates[:, 1]] < 0.15
            path_filter = torch.any(filter_idx.reshape(num_envs, states.shape[1], -1), dim=-1)

            curr_cost = torch.zeros(num_envs, states.shape[1], device=states.device)
            curr_cost[path_filter] += self.to_cfg.state_cost_w_fatal_trav
            cost[curr_env_idx_range] += curr_cost

        return cost

    ###
    # Helper / additional cost terms
    ###

    def _wrap_to_pi(self, angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def smoothness_cost(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Penalize action changes along horizon.
        actions: (BS, NR_TRAJ, TRAJ_LENGTH, 3)
        returns: (BS, NR_TRAJ, TRAJ_LENGTH)
        """
        cost = torch.zeros(actions.shape[:3], device=actions.device, dtype=actions.dtype)
        if actions.shape[2] <= 1:
            return cost

        diff = actions[:, :, 1:, :] - actions[:, :, :-1, :]

        w_vx = getattr(self.to_cfg, "state_cost_w_smooth_vx", 0.0)
        w_vy = getattr(self.to_cfg, "state_cost_w_smooth_vy", 0.0)
        w_wz = getattr(self.to_cfg, "state_cost_w_smooth_wz", 0.0)

        smooth = (
            torch.abs(diff[..., 0]) * w_vx
            + torch.abs(diff[..., 1]) * w_vy
            + torch.abs(diff[..., 2]) * w_wz
        )

        cost[:, :, 1:] = smooth
        cost[:, :, 0] = smooth[:, :, 0]
        return cost

    def yaw_rate_change_cost(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Extra penalty for oscillatory yaw-rate profiles.
        """
        cost = torch.zeros(actions.shape[:3], device=actions.device, dtype=actions.dtype)
        if actions.shape[2] <= 1:
            return cost

        diff_wz = torch.abs(actions[:, :, 1:, 2] - actions[:, :, :-1, 2])
        w = getattr(self.to_cfg, "state_cost_w_yaw_rate_change", 0.0)

        cost[:, :, 1:] = diff_wz * w
        cost[:, :, 0] = diff_wz[:, :, 0] * w
        return cost

    def heading_running_cost(self, states: torch.Tensor) -> torch.Tensor:
        """
        Encourage trajectory heading to align with direction to goal during rollout.
        states: (BS, NR_TRAJ, TRAJ_LENGTH, 3)
        """
        goal_xy = self.obs["goal"][self.env_ids, None, None, :2]
        pos_xy = states[:, :, :, :2]
        goal_vec = goal_xy - pos_xy
        goal_heading = torch.atan2(goal_vec[..., 1], goal_vec[..., 0])

        yaw_err = torch.abs(self._wrap_to_pi(goal_heading - states[..., 2]))
        w = getattr(self.to_cfg, "state_cost_w_heading_running", 0.0)
        return yaw_err * w

    def _sample_height_gradient(self, local_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample local terrain gradient from extero_obs height scan at trajectory states in local/base frame.

        local_states: (BS, NR_TRAJ, TRAJ_LENGTH, 3)
        returns:
            grad_x, grad_y: (BS, NR_TRAJ, TRAJ_LENGTH)
        """
        assert "extero_obs" in self.obs, "extero_obs required for height-gradient based cost"

        height_scan = self._select_planner_batch_tensor(
            self.obs["extero_obs"],
            local_states.shape[0],
            local_states.device,
        ).squeeze(1).float()  # (BS, H, W)
        BS, H, W = height_scan.shape

        grad_x_map = torch.zeros_like(height_scan)
        grad_y_map = torch.zeros_like(height_scan)

        grad_x_map[:, 1:-1, :] = (height_scan[:, 2:, :] - height_scan[:, :-2, :]) / (2 * self.height_scan_resolution)
        grad_y_map[:, :, 1:-1] = (height_scan[:, :, 2:] - height_scan[:, :, :-2]) / (2 * self.height_scan_resolution)

        center = torch.tensor(
            [[
                H / 2 - self.height_scan_offset[1] / self.height_scan_resolution,
                W / 2 - self.height_scan_offset[0] / self.height_scan_resolution,
            ]],
            device=local_states.device,
        )

        idx = center[:, None, None, :] + (
            local_states[..., [1, 0]] / self.height_scan_resolution
        ) * torch.tensor([-1.0, 1.0], device=local_states.device)

        idx = idx.long()
        idx[..., 0] = torch.clamp(idx[..., 0], 0, H - 1)
        idx[..., 1] = torch.clamp(idx[..., 1], 0, W - 1)

        batch_idx = torch.arange(BS, device=local_states.device)[:, None, None].expand(
            BS, local_states.shape[1], local_states.shape[2]
        )

        gx = grad_x_map[batch_idx, idx[..., 0], idx[..., 1]]
        gy = grad_y_map[batch_idx, idx[..., 0], idx[..., 1]]

        return gx, gy

    def stair_alignment_cost(self, local_states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Encourage motion direction to align with terrain gradient direction in step-like regions.
        local_states: (BS, NR_TRAJ, TRAJ_LENGTH, 3) in local/base frame
        actions:      (BS, NR_TRAJ, TRAJ_LENGTH, 3)
        """
        if "extero_obs" not in self.obs:
            return torch.zeros(actions.shape[:3], device=actions.device, dtype=actions.dtype)

        gx, gy = self._sample_height_gradient(local_states)
        grad_norm = torch.sqrt(gx**2 + gy**2 + 1e-8)

        grad_thr = getattr(self.to_cfg, "state_cost_stair_grad_threshold", 1e9)
        active = grad_norm > grad_thr

        stair_heading = torch.atan2(gy, gx)

        cmd_heading = torch.atan2(actions[..., 1], actions[..., 0] + 1e-8)
        heading_err = torch.abs(self._wrap_to_pi(cmd_heading - stair_heading))

        cmd_speed = torch.norm(actions[..., :2], dim=-1)
        moving = cmd_speed > getattr(self.to_cfg, "state_cost_stair_speed_threshold", 0.05)

        w = getattr(self.to_cfg, "state_cost_w_stair_alignment", 0.0)

        cost = torch.zeros_like(heading_err)
        mask = active & moving
        cost[mask] = heading_err[mask] * w
        return cost

    ###
    # Original helper functions
    ###

    def get_start_state(self, batch_size: int, nr_traj: int, env_ids: list[int] | None = None) -> torch.Tensor:
        """
        Initializes the start state for a batch of trajectories.

        Args:
            batch_size (int): The batch size.
            nr_traj (int): The number of trajectories.
            env_ids (list[int]): The environment ids for which to get the start state.

        Returns:
            torch.Tensor: The start state replicated for each trajectory, with shape (BS, NR_TRAJ, STATE_DIM).
        """
        if env_ids is None:
            env_ids = self.env_ids

        return self.obs["start"].clone()[env_ids, None, :].repeat(1, nr_traj, 1)

    def debug_callback(self, states, total_cost):
        b = 0
        best_traj = torch.argmin(total_cost[b])

        for key in [
            "states_control_effort_rot",
            "states_control_effort_trans_side",
            "states_control_effort_trans_forward",
            "states_early_goal_cost",
            "states_early_stopping_cost",
            "velocity_tracking_cost",
            "heading_running_cost",
            "smoothness_cost",
            "stair_alignment_cost",
            "yaw_rate_change_cost",
            "states_running_cost",
        ]:
            if key not in self.debug_info:
                continue

    def record_cvae_executed_outcomes(
        self,
        executed_actions: torch.Tensor,
        env_ids: torch.Tensor | list[int],
        outcome_success: torch.Tensor,
    ) -> None:
        if self.to_cfg.cvae_dataset_dump_path is None:
            return

        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, device=executed_actions.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=executed_actions.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        outcome_success = outcome_success.to(device=executed_actions.device, dtype=torch.bool).reshape(-1)
        if outcome_success.numel() == executed_actions.shape[0]:
            outcome_success = outcome_success[env_ids]
        if outcome_success.numel() != env_ids.numel():
            raise ValueError(
                f"Outcome label count {outcome_success.numel()} does not match env count {env_ids.numel()}."
            )

        context = None
        try:
            context = self._build_cvae_context(env_ids)
        except (AttributeError, KeyError, IndexError):
            context = None
        if self.to_cfg.cvae_require_context and context is None:
            return

        target_actions = executed_actions[env_ids].detach().clone()
        mean_actions = self.optim.mean[env_ids].detach().clone()
        if mean_actions.shape != target_actions.shape:
            mean_actions = target_actions.clone()

        goal_state = torch.where(
            outcome_success,
            torch.full_like(outcome_success, 2, dtype=torch.int64),
            torch.full_like(outcome_success, 1, dtype=torch.int64),
        )
        risk_proxy = (~outcome_success).to(dtype=torch.float32)

        self._cvae_mean_buffer.append(mean_actions.to("cpu"))
        self._cvae_target_buffer.append(target_actions.to("cpu"))
        self._cvae_goal_state_buffer.append(goal_state.to("cpu"))
        self._cvae_outcome_success_buffer.append(outcome_success.to("cpu"))
        self._cvae_success_mask_buffer.append(torch.ones_like(outcome_success, dtype=torch.bool).to("cpu"))
        self._cvae_sample_weight_buffer.append(torch.ones_like(risk_proxy, dtype=torch.float32).to("cpu"))
        self._cvae_risk_max_buffer.append(risk_proxy.to("cpu"))
        self._cvae_risk_sum_buffer.append(risk_proxy.to("cpu"))
        if context is not None:
            self._cvae_context_buffer.append(context.detach().to("cpu"))
        self._cvae_samples_since_flush += int(env_ids.numel())

        self._flush_cvae_dataset()

    def logging_callback(self, population: torch.Tensor, values: torch.Tensor, iteration: int):
        if self.to_cfg.debug:
            min_v = values.min()
        if self.to_cfg.cvae_dataset_dump_path is None:
            return
        if self.to_cfg.cvae_collect_all_iterations:
            stride = max(int(self.to_cfg.cvae_collect_iteration_stride), 1)
            if iteration % stride != 0:
                return
        elif iteration != self.optim.num_iterations - 1:
            return
        if self.to_cfg.cvae_require_context and self._last_cvae_context is None:
            return

        # population: (N, BS, H, A), values: (N, BS)
        pop_size, bs = population.shape[0], population.shape[1]
        k = min(self.to_cfg.cvae_dataset_topk, pop_size)
        high_k = max(int(round(k * self.to_cfg.cvae_bucket_ratio_high)), 1)
        mid_k = max(int(round(k * self.to_cfg.cvae_bucket_ratio_mid)), 1)
        low_k = max(k - high_k - mid_k, 1)
        if high_k + mid_k + low_k > pop_size:
            low_k = max(pop_size - high_k - mid_k, 0)

        sorted_idx = torch.argsort(values, dim=0, descending=True)
        high_idx = sorted_idx[:high_k]
        mid_start = max((pop_size - mid_k) // 2, high_k)
        mid_idx = sorted_idx[mid_start : mid_start + mid_k]
        low_idx = sorted_idx[-low_k:] if low_k > 0 else sorted_idx[:0]
        mixed_idx = torch.cat([high_idx, mid_idx, low_idx], dim=0)

        batch_idx = torch.arange(bs, device=population.device).unsqueeze(0).expand(mixed_idx.shape[0], -1)
        mixed_traj = population[mixed_idx, batch_idx]  # (k_mix, BS, H, A)

        mean = self.optim.mean[self.env_ids].detach().clone()
        keep_k = mixed_traj.shape[0]
        mean_expand = mean.unsqueeze(0).expand(keep_k, -1, -1, -1)
        self._cvae_mean_buffer.append(mean_expand.reshape(-1, mean.shape[1], mean.shape[2]).to("cpu"))
        self._cvae_target_buffer.append(mixed_traj.reshape(-1, mixed_traj.shape[2], mixed_traj.shape[3]).to("cpu"))

        flat_n = keep_k * bs
        self._cvae_goal_state_buffer.append(torch.zeros(flat_n, dtype=torch.int64))
        self._cvae_outcome_success_buffer.append(torch.zeros(flat_n, dtype=torch.bool))
        self._cvae_success_mask_buffer.append(torch.zeros(flat_n, dtype=torch.bool))
        self._cvae_sample_weight_buffer.append(torch.ones(flat_n, dtype=torch.float32))
        risk_proxy = (-values[mixed_idx, batch_idx]).reshape(-1).detach().to("cpu")
        self._cvae_risk_max_buffer.append(risk_proxy)
        self._cvae_risk_sum_buffer.append(risk_proxy.clone())
        if self._last_cvae_context is not None:
            context_expand = self._last_cvae_context.unsqueeze(0).expand(keep_k, -1, -1)
            self._cvae_context_buffer.append(context_expand.reshape(-1, context_expand.shape[-1]).to("cpu"))
        self._cvae_samples_since_flush += int(flat_n)

        if self._cvae_samples_since_flush >= self.to_cfg.cvae_flush_every_n_samples:
            self._flush_cvae_dataset()

    def _flush_cvae_dataset(self) -> None:
        if self.to_cfg.cvae_dataset_dump_path is None:
            return
        mean_actions = torch.cat(self._cvae_mean_buffer, dim=0) if len(self._cvae_mean_buffer) > 0 else None
        target_actions = torch.cat(self._cvae_target_buffer, dim=0) if len(self._cvae_target_buffer) > 0 else None
        if mean_actions is None or target_actions is None:
            return

        goal_state = torch.cat(self._cvae_goal_state_buffer, dim=0) if len(self._cvae_goal_state_buffer) > 0 else None
        outcome_success = (
            torch.cat(self._cvae_outcome_success_buffer, dim=0) if len(self._cvae_outcome_success_buffer) > 0 else None
        )
        risk_max = torch.cat(self._cvae_risk_max_buffer, dim=0) if len(self._cvae_risk_max_buffer) > 0 else None
        risk_sum = torch.cat(self._cvae_risk_sum_buffer, dim=0) if len(self._cvae_risk_sum_buffer) > 0 else None
        success_mask = torch.cat(self._cvae_success_mask_buffer, dim=0) if len(self._cvae_success_mask_buffer) > 0 else None
        sample_weight = torch.cat(self._cvae_sample_weight_buffer, dim=0) if len(self._cvae_sample_weight_buffer) > 0 else None

        max_n = self.to_cfg.cvae_dataset_max_samples
        if mean_actions.shape[0] > max_n:
            mean_actions = mean_actions[-max_n:]
            target_actions = target_actions[-max_n:]
            if len(self._cvae_context_buffer) > 0:
                context = torch.cat(self._cvae_context_buffer, dim=0)[-max_n:]
                self._cvae_context_buffer = [context]
            self._cvae_mean_buffer = [mean_actions]
            self._cvae_target_buffer = [target_actions]
            if goal_state is not None:
                goal_state = goal_state[-max_n:]
                self._cvae_goal_state_buffer = [goal_state]
            if outcome_success is not None:
                outcome_success = outcome_success[-max_n:]
                self._cvae_outcome_success_buffer = [outcome_success]
            if risk_max is not None:
                risk_max = risk_max[-max_n:]
                self._cvae_risk_max_buffer = [risk_max]
            if risk_sum is not None:
                risk_sum = risk_sum[-max_n:]
                self._cvae_risk_sum_buffer = [risk_sum]
            if success_mask is not None:
                success_mask = success_mask[-max_n:]
                self._cvae_success_mask_buffer = [success_mask]
            if sample_weight is not None:
                sample_weight = sample_weight[-max_n:]
                self._cvae_sample_weight_buffer = [sample_weight]

        if success_mask is not None:
            labeled_ratio = success_mask.float().mean().item()
            if labeled_ratio < self.to_cfg.cvae_labeled_ratio_min:
                labeled_idx = torch.where(success_mask)[0]
                unlabeled_idx = torch.where(~success_mask)[0]
                if labeled_idx.numel() > 0:
                    max_unlabeled = int(
                        labeled_idx.numel() * (1 - self.to_cfg.cvae_labeled_ratio_min)
                        / self.to_cfg.cvae_labeled_ratio_min
                    )
                    keep_unlabeled = unlabeled_idx[-max_unlabeled:] if max_unlabeled > 0 else unlabeled_idx[:0]
                    keep_idx = torch.cat([labeled_idx, keep_unlabeled], dim=0)
                    mean_actions = mean_actions[keep_idx]
                    target_actions = target_actions[keep_idx]
                    goal_state = goal_state[keep_idx] if goal_state is not None else None
                    outcome_success = outcome_success[keep_idx] if outcome_success is not None else None
                    risk_max = risk_max[keep_idx] if risk_max is not None else None
                    risk_sum = risk_sum[keep_idx] if risk_sum is not None else None
                    success_mask = success_mask[keep_idx]
                    sample_weight = sample_weight[keep_idx] if sample_weight is not None else None
                    if len(self._cvae_context_buffer) > 0:
                        context = torch.cat(self._cvae_context_buffer, dim=0)[keep_idx]
                        self._cvae_context_buffer = [context]
                    self._cvae_mean_buffer = [mean_actions]
                    self._cvae_target_buffer = [target_actions]
                    self._cvae_goal_state_buffer = [goal_state] if goal_state is not None else []
                    self._cvae_outcome_success_buffer = [outcome_success] if outcome_success is not None else []
                    self._cvae_risk_max_buffer = [risk_max] if risk_max is not None else []
                    self._cvae_risk_sum_buffer = [risk_sum] if risk_sum is not None else []
                    self._cvae_success_mask_buffer = [success_mask]
                    self._cvae_sample_weight_buffer = [sample_weight] if sample_weight is not None else []

        payload = {"mean_actions": mean_actions, "target_actions": target_actions}
        if len(self._cvae_context_buffer) > 0:
            payload["context"] = torch.cat(self._cvae_context_buffer, dim=0)
        if goal_state is not None:
            payload["goal_state_3way"] = goal_state
        if outcome_success is not None:
            payload["outcome_success"] = outcome_success
        if risk_max is not None:
            payload["outcome_risk_max"] = risk_max
        if risk_sum is not None:
            payload["outcome_risk_sum"] = risk_sum
        if success_mask is not None:
            payload["success_supervision_mask"] = success_mask
        if sample_weight is not None:
            payload["sample_weight"] = sample_weight

        dump_dir = os.path.dirname(self.to_cfg.cvae_dataset_dump_path)
        if len(dump_dir) > 0:
            os.makedirs(dump_dir, exist_ok=True)
        torch.save(payload, self.to_cfg.cvae_dataset_dump_path)
        self._cvae_samples_since_flush = 0

    def forward_dynamics(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Single sted forward dynamics model.
        """
        # Unit m + m/s * dt planning (actions are already integrated)
        return state + action
