# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from prettytable import PrettyTable
from torch.utils.data import Dataset
from typing import TYPE_CHECKING

import pypose as pp

import isaaclab.utils.math as math_utils

from fdm import VEL_RANGE_X, VEL_RANGE_Y, VEL_RANGE_YAW

if TYPE_CHECKING:
    from fdm.data_buffers import ReplayBuffer, ReplayBufferCfg
    from fdm.model import FDMBaseModelCfg
    from fdm.runner import TrainerBaseCfg


class TrajectoryDataset(Dataset):
    def __init__(
        self, cfg: TrainerBaseCfg, model_cfg: FDMBaseModelCfg, replay_buffer_cfg: ReplayBufferCfg, return_device: str
    ):
        # save configs
        self.cfg: TrainerBaseCfg = cfg
        self.model_cfg: FDMBaseModelCfg = model_cfg
        self.replay_buffer_cfg: ReplayBufferCfg = replay_buffer_cfg
        self._actual_nbr_samples: int = self.cfg.num_samples
        self.return_device: str = return_device

        # save min and max of the hard_contact_obs (as part of the state) for normalization
        self.min_hard_contact_obs = torch.tensor([torch.inf])
        self.max_hard_contact_obs = torch.zeros(1)

        # init extereoceptive noise model
        if self.cfg.extereoceptive_noise_model is not None:
            self.extereoceptive_noise_model = self.cfg.extereoceptive_noise_model.noise_model(
                self.cfg.extereoceptive_noise_model, device=self.replay_buffer_cfg.buffer_device
            )
        else:
            self.extereoceptive_noise_model = None

    def __str__(self) -> str:
        msg = (
            "#############################################################################################\n"
            f"<RandomTrajectoryDataset> with command trajectory (length {self.replay_buffer_cfg.trajectory_length})"
            " contains\n"
            f"\tIntended Number: \t{self.cfg.num_samples}\n"
            f"\tCollision rate : \t{self.collision_rate})\n"
            f"\tReturn Device  : \t{self.return_device}\n"
            "#############################################################################################"
        )

        return msg

    ##
    # Properties
    ##

    @property
    def collision_sample_nb(self) -> int:
        return torch.any(self.states[..., 4], axis=1).sum()

    @property
    def collision_rate(self) -> float:
        return self.collision_sample_nb / self.__len__()

    @property
    def num_samples(self) -> int:
        return self._actual_nbr_samples

    ##
    # Operations
    ##

    def populate(  # noqa: C901
        self, replay_buffer: ReplayBuffer, regular_slicing: bool = False, start_idx: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Update data in the buffer for specified indexes.

        Args:
            replay_buffer: The replay buffer to get the data from.
            regular_slicing: If True, the data is sliced regularly.
            start_idx: The start indexes for the different environments where the samples should be taken from.
        """
        # get start and end indexes for the different environments
        num_trajectories = replay_buffer.states.shape[0]
        if regular_slicing and start_idx is None:
            trajectory_idx = torch.arange(
                0,
                self.replay_buffer_cfg.trajectory_length - self.model_cfg.prediction_horizon - 1,
                device=self.replay_buffer_cfg.buffer_device,
            )[1 :: self.model_cfg.prediction_horizon]
            start_idx = torch.vstack((
                torch.arange(0, num_trajectories, device=self.replay_buffer_cfg.buffer_device)[:, None]
                .repeat(1, len(trajectory_idx))
                .flatten(),
                trajectory_idx.repeat(num_trajectories),
            )).T
            terminal_idx = getattr(
                replay_buffer,
                "terminal_idx",
                torch.full(
                    (num_trajectories,),
                    self.replay_buffer_cfg.trajectory_length,
                    device=self.replay_buffer_cfg.buffer_device,
                ),
            )
            valid_idx = getattr(
                replay_buffer,
                "valid_idx",
                torch.full(
                    (num_trajectories,),
                    self.replay_buffer_cfg.trajectory_length,
                    device=self.replay_buffer_cfg.buffer_device,
                ),
            )
            valid_start = start_idx[:, 1] + self.model_cfg.prediction_horizon < valid_idx[start_idx[:, 0]]
            valid_start &= start_idx[:, 1] + self.model_cfg.prediction_horizon < terminal_idx[start_idx[:, 0]]
            start_idx = start_idx[valid_start]
        elif start_idx is None:
            traj_start_idx = self._sample_random_traj_idx(replay_buffer)
            coll_start_idx = self._sample_collision_traj(replay_buffer)

            # balance the data
            if self.cfg.collision_rate is not None:
                num_regular = int(self.cfg.num_samples * (1 - self.cfg.collision_rate))
                num_collision = self.cfg.num_samples - num_regular
                if coll_start_idx.shape[0] == 0:
                    num_regular = self.cfg.num_samples
                    num_collision = 0
                assert traj_start_idx.shape[0] > 0, "No valid regular samples found in replay buffer!"
                assert (
                    num_regular <= traj_start_idx.shape[0]
                ), "Not enough regular samples to balance data!"
                perm = torch.randperm(traj_start_idx.shape[0], device=self.replay_buffer_cfg.buffer_device)
                traj_start_idx = traj_start_idx[perm[:num_regular]]
                if num_collision <= coll_start_idx.shape[0]:
                    perm = torch.randperm(coll_start_idx.shape[0], device=self.replay_buffer_cfg.buffer_device)
                    coll_start_idx = coll_start_idx[perm[:num_collision]]
                else:
                    coll_start_idx = coll_start_idx.repeat(
                        num_collision // coll_start_idx.shape[0] + 1, 1
                    )
                    coll_start_idx = coll_start_idx[:num_collision]
                start_idx = torch.vstack([traj_start_idx, coll_start_idx])
            else:
                start_idx = torch.vstack([traj_start_idx, coll_start_idx])

        ###
        # Actions
        ###
        self.actions = torch.concatenate(
            [
                replay_buffer.actions[start_idx[:, 0], start_idx[:, 1] + idx][:, None, :]
                for idx in range(self.model_cfg.prediction_horizon)
            ],
            dim=1,
        )

        ###
        # States and state history
        ###

        # get current state and use it to transform the previous and following states into the local robot frame
        # shape: [N, 7] with [x, y, z, qx, qy, qz, qw]
        initial_states = replay_buffer.states[start_idx[:, 0], start_idx[:, 1], 0][:, None, :7]
        initial_states_SE3 = pp.SE3(initial_states.repeat(1, self.model_cfg.prediction_horizon, 1).reshape(-1, 7))

        # get the state history
        self.state_history = self.state_history_transformer(
            replay_buffer,
            start_idx,
            initial_states,
            self.model_cfg.history_length,
            self.model_cfg.exclude_state_idx_from_input,
        )

        # get the future positions along the trajectory
        states = torch.concatenate(
            [
                replay_buffer.states[start_idx[:, 0], start_idx[:, 1] + idx + 1, 0][:, None]
                for idx in range(self.model_cfg.prediction_horizon)
            ],
            dim=1,
        )
        states_SE3 = pp.SE3(states[..., :7].reshape(-1, 7))
        states_SE3 = (pp.Inv(initial_states_SE3) * states_SE3).tensor()
        states_yaw = math_utils.euler_xyz_from_quat(states_SE3[..., [6, 3, 4, 5]])[2]
        # rotation encoded as [sin(yaw), cos(yaw)] to avoid jump in representation
        # Check: Learning with 3D rotations, a hitchhiker’s guide to SO(3), 2024, Frey et al.
        states_yaw_sin_cos = torch.stack([torch.sin(states_yaw), torch.cos(states_yaw)], dim=1)
        # final state: [N, Prediction Horizon, 3 (pos) + 2 (yaw) + 1 (collision) + rest of the state]
        self.states = torch.concatenate(
            [
                states_SE3.reshape(-1, self.model_cfg.prediction_horizon, 7)[..., :2],
                states_yaw_sin_cos.reshape(-1, self.model_cfg.prediction_horizon, 2),
                states[..., 7:],
            ],
            dim=2,
        )
        if self.model_cfg.hard_contact_metric == "contact" or self.model_cfg.hard_contact_metric == "torque":
            max_metric = torch.concatenate(
                [
                    replay_buffer.states[start_idx[:, 0], start_idx[:, 1] + idx + 1, :, 8].unsqueeze(1)
                    for idx in range(self.model_cfg.prediction_horizon)
                ],
                dim=1,
            )
            max_metric = torch.max(max_metric, dim=-1)[0]
            if self.model_cfg.hard_contact_metric == "contact":
                self.states[..., 5] = torch.log(max_metric)
            else:
                self.states[..., 5] = max_metric

        ###
        # Observations
        ###

        self.obs_proprioceptive = replay_buffer.observations_proprioceptive[start_idx[:, 0], start_idx[:, 1]]
        if replay_buffer.observations_exteroceptive is not None:
            self.obs_exteroceptive = replay_buffer.observations_exteroceptive[start_idx[:, 0], start_idx[:, 1]]
        else:
            self.obs_exteroceptive = None
        if replay_buffer.add_observations_exteroceptive is not None:
            self.add_obs_exteroceptive = replay_buffer.add_observations_exteroceptive[start_idx[:, 0], start_idx[:, 1]]
        else:
            self.add_obs_exteroceptive = None

        ###
        # Perfect velocity following - EVALUATION ONLY
        ###

        # get the resulting change in position and angle when applying the commands perfectly
        # velocity command units x: [m/s], y: [m/s], phi: [rad/s]
        perfect_velocity_following_individual_frame = self.actions * self.model_cfg.command_timestep

        # Cumsum is an inplace operation therefore the clone is necesasry
        cummulative_yaw = perfect_velocity_following_individual_frame.clone()[..., -1].cumsum(-1)

        # We need to take the non-linearity by the rotation into account
        r_vec1 = torch.stack([torch.cos(cummulative_yaw), -torch.sin(cummulative_yaw)], dim=-1)
        r_vec2 = torch.stack([torch.sin(cummulative_yaw), torch.cos(cummulative_yaw)], dim=-1)
        so2 = torch.stack([r_vec1, r_vec2], dim=2)

        # Move the rotation in time and fill first timestep with identity - see math chapter
        so2 = torch.roll(so2, shifts=1, dims=1)
        so2[:, 0, :, :] = torch.eye(2, device=so2.device)[None].repeat(so2.shape[0], 1, 1)

        actions_local_frame = so2.contiguous().reshape(-1, 2, 2) @ perfect_velocity_following_individual_frame[
            ..., :2
        ].contiguous().reshape(-1, 2, 1)
        actions_local_frame = actions_local_frame.contiguous().reshape(so2.shape[0], so2.shape[1], 2)
        cumulative_position = (actions_local_frame).cumsum(-2)
        self.perfect_velocity_following_local_frame = torch.cat(
            [cumulative_position, torch.sin(cummulative_yaw)[:, :, None], torch.cos(cummulative_yaw)[:, :, None]],
            dim=-1,
        )

        ###
        # Filter data
        ###
        # init keep index array
        keep_idx = torch.ones(
            self.state_history.shape[0], dtype=torch.bool, device=self.replay_buffer_cfg.buffer_device
        )
        # ===== DEBUG (print once) =====
        # DEBUG once
        if not hasattr(self, "_dbg_coll"):
            self._dbg_coll = True
            print("coll mean BEFORE filter:", self.states[..., 4].float().mean().item(), flush=True)
            print("coll any BEFORE filter:", (self.states[..., 4] != 0).any(dim=1).float().mean().item(), flush=True)

        if not hasattr(self, "_dbg_printed"):
            self._dbg_printed = True

            print("\n========== DEBUG: state layout ==========")
            print("states shape:", tuple(self.states.shape))           # [N, H, D]
            print("state_history shape:", tuple(self.state_history.shape))  # [N, hist, D?]

            # 关键：看看第4维到底是什么
            d = self.states.shape[-1]
            show = min(d, 12)
            print("states[0,0,0:show] =", self.states[0, 0, :show].detach().cpu())

            # 看每一维的范围，判断哪一维像 collision (0/1)
            x = self.states.reshape(-1, d)
            mins = x.min(0).values.detach().cpu()
            maxs = x.max(0).values.detach().cpu()
            means = x.mean(0).detach().cpu()
            print("per-dim min :", mins[:show])
            print("per-dim max :", maxs[:show])
            print("per-dim mean:", means[:show])

            # 直接检查 index=4 的统计
            col = self.states[..., 4]
            print("states[...,4] dtype:", col.dtype)
            print("states[...,4] unique-ish (min/max/mean):",
                col.min().item(), col.max().item(), col.float().mean().item())

            # 看看是不是“几乎全 True”
            if col.dtype != torch.bool:
                mask = col != 0
            else:
                mask = col
            print("states[...,4] nonzero ratio:", mask.float().mean().item())
            print("========================================\n")

        raw = replay_buffer.states[start_idx[0,0], start_idx[0,1] + 1]
        print("raw per-agent state shape:", raw.shape, flush=True)        # 期望 [num_agents?, state_dim?]
        print("raw[0, :10] =", raw[0, :10], flush=True)                  # 0号 agent 的前10维
        col = (self.states[...,4] != 0)
        print("collision per-step:", col.float().mean().item(), flush=True)
        print("collision any(dim=1):", col.any(dim=1).float().mean().item(), flush=True)
        print("states (SE3 input) sample:", states[0,0,:7], flush=True)

        if not hasattr(self, "_dbg_obs_dim"):
            self._dbg_obs_dim = True
            print("obs_proprioceptive shape:", tuple(self.obs_proprioceptive.shape), flush=True)


        # ===== END DEBUG =====

        # -----------------------------------------------------------------------------
        # Filtering statistics
        # -----------------------------------------------------------------------------
        num_total = self.states.shape[0]
        print(f"[FILTER] total samples before filtering: {num_total}")

        # 统计各项单独命中的数量
        filter_stats = {}

        # -----------------------------------------------------------------------------
        # 1) filter every sample that has a collision in the first K steps
        # -----------------------------------------------------------------------------
        collision_env, collision_idx = torch.where(self.states[..., 4])
        first_steps_mask = collision_idx < self.cfg.sample_filter_first_steps_coll
        remove_idx_first_coll = torch.unique(collision_env[first_steps_mask])

        filter_stats["first_steps_collision"] = remove_idx_first_coll.numel()
        keep_idx[remove_idx_first_coll] = False

        print(
            f"[FILTER] first_steps_collision (< {self.cfg.sample_filter_first_steps_coll} steps): "
            f"{filter_stats['first_steps_collision']}"
        )

        # -----------------------------------------------------------------------------
        # 2) filter every sample that has a collision in its initial position
        # -----------------------------------------------------------------------------
        collision_env_init = torch.where(self.state_history[:, 0, 4])[0]

        filter_stats["initial_collision"] = collision_env_init.numel()
        keep_idx[collision_env_init] = False

        print(f"[FILTER] initial_collision: {filter_stats['initial_collision']}")

        # -----------------------------------------------------------------------------
        # 3) restrict ratio of small-motion samples
        # -----------------------------------------------------------------------------
        if self.cfg.small_motion_ratio is not None:
            small_movement_idx_all = torch.where(
                torch.norm(torch.abs(self.states[:, -1, :2]), dim=1) < self.cfg.small_motion_threshold
            )[0]
            small_movement_ratio = small_movement_idx_all.shape[0] / self.states.shape[0]

            print(
                f"[FILTER] small_motion candidates: {small_movement_idx_all.numel()} / {self.states.shape[0]} "
                f"({small_movement_ratio:.4f}), threshold={self.cfg.small_motion_threshold}, "
                f"target_ratio={self.cfg.small_motion_ratio}"
            )

            if small_movement_ratio > self.cfg.small_motion_ratio:
                # solve for num_remove in: r = (x - num_remove) / (N - num_remove)
                num_remove = int(
                    (self.cfg.small_motion_ratio * self.states.shape[0] - small_movement_idx_all.shape[0])
                    / (self.cfg.small_motion_ratio - 1)
                )
                small_movement_idx = small_movement_idx_all[:num_remove]
                filter_stats["small_motion_removed"] = small_movement_idx.numel()
                keep_idx[small_movement_idx] = False
            else:
                filter_stats["small_motion_removed"] = 0

            print(f"[FILTER] small_motion_removed: {filter_stats['small_motion_removed']}")

        # -----------------------------------------------------------------------------
        # 4) filter samples with too little height difference
        # -----------------------------------------------------------------------------
        if self.cfg.height_threshold is not None:
            state_height = states_SE3[..., 2].reshape(-1, self.model_cfg.prediction_horizon)
            height_diff = torch.max(torch.abs(state_height[:, 1:] - state_height[:, :-1]), dim=-1)[0]
            low_height_idx = torch.where(height_diff < self.cfg.height_threshold)[0]

            filter_stats["low_height_diff"] = low_height_idx.numel()
            keep_idx[low_height_idx] = False

            print(
                f"[FILTER] low_height_diff (< {self.cfg.height_threshold}): "
                f"{filter_stats['low_height_diff']}"
            )

        # -----------------------------------------------------------------------------
        # summary before actual filtering
        # -----------------------------------------------------------------------------
        num_keep = keep_idx.sum().item()
        num_remove_total = (~keep_idx).sum().item()

        print(f"[FILTER] total removed after union: {num_remove_total}")
        print(f"[FILTER] total kept after union:    {num_keep}")

        # 如果你想看各过滤项之间的重叠导致“单项之和 > 总移除数”，这个对理解很有帮助
        sum_individual = sum(filter_stats.values())
        print(f"[FILTER] sum of individual removals: {sum_individual}")
        print(f"[FILTER] overlap count: {sum_individual - num_remove_total}")

        # -----------------------------------------------------------------------------
        # filter samples
        # -----------------------------------------------------------------------------
        initial_states = initial_states.repeat(1, self.model_cfg.prediction_horizon, 1)[keep_idx]
        states = states[keep_idx]
        self._filter_idx(keep_idx)
        
        N, H, _ = self.states.shape
        dt = self.model_cfg.command_timestep

        # ---------- 1. 局部坐标最终位移（只杀 teleport） ----------
        delta_xy = self.states[:, -1, :2] - self.states[:, 0, :2]
        max_distance = torch.norm(delta_xy, dim=1)

        # 两足：10m 仍然是一个合理的 teleport 判据
        bad_dist = max_distance > 10.0


        # ---------- 2. 单步“几何跳变”（不是速度） ----------
        step_xy = self.states[:, 1:, :2] - self.states[:, :-1, :2]
        step_norm = torch.norm(step_xy, dim=-1)

        # 两足这里要放宽
        step_spike = (step_norm > 5.0)            # (N, H-1)
        bad_step = (step_spike.float().mean(dim=1) > 0.10)  # 超过10%的步异常才丢



        # ---------- 3. yaw 跳变（两足非常关键） ----------
        yaw = self.states[:, :, 3]  # 直接就是 yaw(rad)
        yaw_diff = math_utils.wrap_to_pi(yaw[:, 1:] - yaw[:, :-1]).abs()

        bad_yaw = ((yaw_diff / dt) > 10.0).any(dim=1)   # 建议 4~6 rad/s；你想更宽可以设到 10
        outlier_mask = bad_dist | bad_yaw | bad_step
        outlier_idx = torch.where(outlier_mask)[0]

        outlier_ratio = outlier_mask.float().mean().item()
        print("outlier ratio:", outlier_ratio)
        print("outlier count:", int(outlier_mask.sum().item()), "/", outlier_mask.numel())



        if len(outlier_idx) > 0:
            print("[WARNING] Found outliers with max position > 10.0!")
            keep_idx = torch.ones(
                self.state_history.shape[0], dtype=torch.bool, device=self.replay_buffer_cfg.buffer_device
            )
            keep_idx[outlier_idx] = False
            initial_states = initial_states[keep_idx]
            max_distance = max_distance[keep_idx]
            states = states[keep_idx]
            self._filter_idx(keep_idx)
            if hasattr(self, "_dbg_coll") and self._dbg_coll:
                print("coll mean AFTER filter:", self.states[..., 4].float().mean().item(), flush=True)
                print("coll any AFTER filter:", (self.states[..., 4] != 0).any(dim=1).float().mean().item(), flush=True)


        if getattr(self.model_cfg, "use_geometric_collision_head", False):
            geometric_collision = self._geometric_collision_target_from_height_scan()
            self.states = torch.cat([self.states, geometric_collision.unsqueeze(-1)], dim=-1)

        ###
        # Normalize the hard contact observation
        ###

        # get min and max torque
        self.max_hard_contact_obs = torch.max(torch.max(self.states[..., 5]), self.max_hard_contact_obs)
        self.min_hard_contact_obs = torch.min(torch.min(self.states[..., 5]), self.min_hard_contact_obs)

        # normalize torque
        self.state_history[..., 5] = (self.state_history[..., 5] - self.min_hard_contact_obs) / (
            self.max_hard_contact_obs - self.min_hard_contact_obs
        )
        self.states[..., 5] = (self.states[..., 5] - self.min_hard_contact_obs) / (
            self.max_hard_contact_obs - self.min_hard_contact_obs
        )

        ###
        # Extract maximum physical values of the system to constrain model
        ###

        # get the maximum observed velocity
        lin_velocity = torch.abs((self.states[:, 1:, :2] - self.states[:, :-1, :2]) / self.model_cfg.command_timestep)
        heading = torch.atan2(self.states[:, :, 2], self.states[:, :, 3])
        # enforce periodicity of the heading
        yaw_diff = torch.abs(heading[:, 1:] - heading[:, :-1])
        yaw_diff = math_utils.wrap_to_pi(yaw_diff)
        ang_velocity = torch.abs(yaw_diff / self.model_cfg.command_timestep)
        max_velocity = torch.concatenate(
            [torch.max(lin_velocity.reshape(-1, 2), dim=0)[0], torch.max(ang_velocity.reshape(-1, 1), dim=0)[0]], dim=0
        )

        # get the maximum observed acceleration
        max_lin_acceleration = torch.max(
            torch.abs((lin_velocity[:, 1:] - lin_velocity[:, :-1]) / self.model_cfg.command_timestep).reshape(-1, 2),
            dim=0,
        )[0]
        max_ang_acceleration = torch.max(
            torch.abs((ang_velocity[:, 1:] - ang_velocity[:, :-1]) / self.model_cfg.command_timestep).reshape(-1, 1),
            dim=0,
        )[0]
        max_acceleration = torch.concatenate([max_lin_acceleration, max_ang_acceleration], dim=0)

        # check the maximum velocity is not more than the maximum commanded velocity
        max_possible_velocity = (VEL_RANGE_X[1] ** 2 + VEL_RANGE_Y[1] ** 2) ** 0.5
        collision_samples = self.states[..., 4].any(dim=1)
        max_velocity_non_collision = torch.concatenate(
            [
                torch.max(lin_velocity[~collision_samples].reshape(-1, 2), dim=0)[0],
                torch.max(ang_velocity[~collision_samples].reshape(-1, 1), dim=0)[0],
            ],
            dim=0,
        )
        if torch.any(
            max_velocity_non_collision
            > torch.tensor([max_possible_velocity, max_possible_velocity, VEL_RANGE_YAW[1]]) * 1.1
        ):
            # NOTE: When the robot is "falling" the velocity, especially the angular velocity, can be very high
            #       while the robot does not necessary collide
            non_colliding_vels = torch.concatenate(
                [lin_velocity[~collision_samples], ang_velocity[~collision_samples].unsqueeze(-1)], dim=-1
            )
            exceeding_cases = torch.any(
                non_colliding_vels
                > torch.tensor([[[max_possible_velocity, max_possible_velocity, VEL_RANGE_YAW[1]]]]) * 1.1,
                dim=-1,
            )
            # get the z diff for the exceeding cases
            z_diff = states[~collision_samples, 1:, 2] - states[~collision_samples, :-1, 2]
            z_diff_exceeding = z_diff[exceeding_cases]
            print(
                f"[WARNING] Maximum observed velocity {max_velocity_non_collision.cpu().tolist()} is higher in"
                f" {exceeding_cases.sum().item()} cases than the maximum commanded velocity"
                f" {[max_possible_velocity, max_possible_velocity, VEL_RANGE_YAW[1]]}! In"
                f" {(torch.abs(z_diff_exceeding) > 0.1).sum().item()} cases the z diff is larger than 0.1m!"
            )

            # restrict maximum applied velocity to the maximum observed velocity for non-collision cases and cases
            # without a jump in z coordinate
            if torch.any(~exceeding_cases.any(dim=1)):
                max_velocity_applied = non_colliding_vels[~exceeding_cases.any(dim=1)].reshape(-1, 3).max(dim=0)[0]
                non_colliding_vels = torch.abs(non_colliding_vels)
                max_acceleration_applied = torch.max(
                    torch.abs(
                        (
                            non_colliding_vels[~exceeding_cases.any(dim=1)][:, 1:]
                            - non_colliding_vels[~exceeding_cases.any(dim=1)][:, :-1]
                        )
                        / self.model_cfg.command_timestep
                    ).reshape(-1, 3),
                    dim=0,
                )[0]
            else:
                max_velocity_applied = max_velocity.clone()
                max_acceleration_applied = max_acceleration.clone()
        else:
            max_velocity_applied = max_velocity.clone()
            max_acceleration_applied = max_acceleration.clone()

        # scale the applied limits with a safety factor of 5% to allow for larger corrections
        max_velocity_applied *= 1.05
        max_acceleration_applied *= 1.05

        ###
        # Extract further statistics
        ###

        # compare states and perfect veloicty estimate
        pos_diff = torch.norm(self.states[..., :2] - self.perfect_velocity_following_local_frame[..., :2], dim=-1)
        cummulative_yaw_states = torch.atan2(self.states[..., 2], self.states[..., 3])
        cummulative_yaw_perfect_velocity_following = torch.atan2(
            self.perfect_velocity_following_local_frame[..., 2], self.perfect_velocity_following_local_frame[..., 3]
        )
        yaw_diff = torch.abs(cummulative_yaw_states - cummulative_yaw_perfect_velocity_following)
        # account for the periodicity of the yaw
        yaw_diff = math_utils.wrap_to_pi(yaw_diff)

        # Mean and Varatity of actions
        action_var = self.actions.view(-1, 3).std(dim=0)
        action_mean = self.actions.view(-1, 3).mean(dim=0)

        ###
        # Check for nan and inf values
        ###

        if torch.any(torch.isnan(self.states)) or torch.any(torch.isinf(self.states)):
            raise ValueError("Nan/ Inf values in states!")
        if torch.any(torch.isnan(self.state_history)) or torch.any(torch.isinf(self.state_history)):
            raise ValueError("Nan/ Inf values in state history!")
        if torch.any(torch.isnan(self.obs_proprioceptive)) or torch.any(torch.isinf(self.obs_proprioceptive)):
            raise ValueError("Nan/ Inf values in proprioceptive observations!")
        if self.obs_exteroceptive is not None and (
            torch.any(torch.isnan(self.obs_exteroceptive)) or torch.any(torch.isinf(self.obs_exteroceptive))
        ):
            raise ValueError("Nan/ Inf values in exteroceptive observations!")
        if self.add_obs_exteroceptive is not None and (
            torch.any(torch.isnan(self.add_obs_exteroceptive)) or torch.any(torch.isinf(self.add_obs_exteroceptive))
        ):
            raise ValueError("Nan/ Inf values in additional exteroceptive observations!")
        if torch.any(torch.isnan(self.actions)) or torch.any(torch.isinf(self.actions)):
            raise ValueError("Nan/ Inf values in actions!")
        if torch.any(torch.isnan(self.perfect_velocity_following_local_frame)) or torch.any(
            torch.isinf(self.perfect_velocity_following_local_frame)
        ):
            raise ValueError("Nan/ Inf values in perfect velocity following!")

        ###
        # Ablation studies
        ###

        if self.cfg.ablation_no_state_obs:
            self.state_history *= 0.0
        elif self.cfg.ablation_no_proprio_obs:
            self.obs_proprioceptive *= 0.0
        elif self.cfg.ablation_no_height_scan and self.obs_exteroceptive is not None:
            self.obs_exteroceptive *= 0.0

        ###
        # Print meta information
        print("lin_velocity abs max:", lin_velocity.abs().max().item())
        print("pos delta abs max:", (self.states[:, 1:, :2]-self.states[:, :-1, :2]).abs().max().item())


        table = PrettyTable()
        table.field_names = ["Metric", "Value"]
        table.align["Metric"] = "l"
        table.align["Value"] = "r"

        # Add rows with formatted values
        #table.add_row(("Average max distance", f"{torch.mean(torch.abs(max_distance), dim=0).item():.4f}"))
        table.add_row(("Average collision rate", f"{self.collision_rate:.4f}"))
        table.add_row(("Max velocity", [f"{v:.4f}" for v in max_velocity.cpu().tolist()]))
        table.add_row(("Max acceleration", [f"{a:.4f}" for a in max_acceleration.cpu().tolist()]))
        table.add_row(("Max velocity applied", [f"{v:.4f}" for v in max_velocity_applied.cpu().tolist()]))
        table.add_row(("Max acceleration applied", [f"{a:.4f}" for a in max_acceleration_applied.cpu().tolist()]))
        table.add_row(("Max hard contact observation", f"{self.max_hard_contact_obs.item():.4f}"))
        table.add_row(("Min hard contact observation", f"{self.min_hard_contact_obs.item():.4f}"))

        # Print distance percentages
        #for distance in range(1, int(torch.max(torch.abs(max_distance)).item()) + 2):
        #    ratio = (
        #        torch.sum(torch.all(torch.vstack((max_distance > distance - 1, max_distance < distance)), dim=0))
        #        / self.states.shape[0]
        #    )
        #    table.add_row((f"Ratio between {distance - 1} - {distance}m", f"{ratio.item():.4f}"))
        #for distance in range(1, int(torch.ceil(torch.max(torch.abs(self.states[:, -1, 0])))) + 1):
        #    ratio = (
        #        torch.sum(
        #            torch.all(
        #                torch.vstack((
        #                    torch.abs(self.states[:, -1, 0]) > distance - 1,
        #                    torch.abs(self.states[:, -1, 0]) < distance,
        #               )),
        #                dim=0,
        #            )
        #        )
        #        / self.states.shape[0]
        #    )
        #    table.add_row((f"Ratio between {distance - 1} - {distance}m in x", f"{ratio.item():.4f}"))
        #for distance in range(1, int(torch.ceil(torch.max(torch.abs(self.states[:, -1, 1])))) + 1):
        #    ratio = (
        #        torch.sum(
        #            torch.all(
        #                torch.vstack((
        #                    torch.abs(self.states[:, -1, 1]) > distance - 1,
        #                    torch.abs(self.states[:, -1, 1]) < distance,
        #                )),
        #                dim=0,
        #            )
        #        )
        #        / self.states.shape[0]
        #    )
        #    table.add_row((f"Ratio between {distance - 1} - {distance}m in y", f"{ratio.item():.4f}"))

        # Print differences between states and perfect velocity following
        table.add_row((
            "Perf Vel Position difference",
            f"{torch.mean(pos_diff).item():.4f}" + " \u00b1 " + f"{torch.std(pos_diff).item():.4f}",
        ))
        table.add_row((
            "Perf Vel Yaw difference",
            f"{torch.mean(yaw_diff).item():.4f}" + " \u00b1 " + f"{torch.std(yaw_diff).item():.4f}",
        ))
        table.add_row(("Perf Vel Max position difference", f"{torch.max(pos_diff).item():.4f}"))
        table.add_row(("Perf Vel Max yaw difference", f"{torch.max(yaw_diff).item():.4f}"))

        # Print action variance
        table.add_row(("Action Mean", [f"{v:.4f}" for v in action_mean.cpu().tolist()]))
        table.add_row(("Action Variance", [f"{v:.4f}" for v in action_var.cpu().tolist()]))

        # add info about ablation studies
        table.add_row(("Ablation no state obs", self.cfg.ablation_no_state_obs))
        table.add_row(("Ablation no proprio obs", self.cfg.ablation_no_proprio_obs))
        table.add_row(("Ablation no height scan", self.cfg.ablation_no_height_scan))

        # Print table
        print(f"[INFO] Dataset Metrics {self.states.shape[0]} samples\n", table)

        if False:
            ###
            # Debug try to crop height scan to current position
            ###

            # visualize the split up height scane for each step of the FDM
            print("Debugging")
            import math

            height_scan_res = 0.1
            # Define the bounds of the subregion to extract
            x_min, x_max = -0.5, 1.0  # height --> x
            y_min, y_max = -1.0, 1.0  # width --> y

            height_scan_shape = (self.obs_exteroceptive.shape[-2], self.obs_exteroceptive.shape[-1])
            height_scan_robot_center = [height_scan_shape[0] / 2, 0.5 / height_scan_res]

            # get effective translation
            # since in robot frame, the y translation is against the height axis x direction, has to be negative
            effective_translation_tensor_x = (
                -self.states[:, :, 1].reshape(-1) / height_scan_res + height_scan_robot_center[0]
            )
            effective_translation_tensor_y = (
                self.states[:, :, 0].reshape(-1) / height_scan_res + height_scan_robot_center[1]
            )

            # Create a meshgrid of coordinates
            idx_tensor_x, idx_tensor_y = torch.meshgrid(
                torch.arange(y_min / height_scan_res, (y_max / height_scan_res) + 1),
                torch.arange(x_min / height_scan_res, (x_max / height_scan_res) + 1),
                indexing="ij",
            )
            idx_tensor_x = idx_tensor_x.flatten().float().repeat(self.states.shape[0] * self.states.shape[1], 1)
            idx_tensor_y = idx_tensor_y.flatten().float().repeat(self.states.shape[0] * self.states.shape[1], 1)

            # angle definition for the height scan coordinate system is opposite of the tensor system, so negative
            s = self.states[:, :, 2].reshape(-1).unsqueeze(1)
            c = self.states[:, :, 3].reshape(-1).unsqueeze(1)
            idx_crop_x = (c * idx_tensor_x - s * idx_tensor_y + effective_translation_tensor_x.unsqueeze(1)).int()
            idx_crop_y = (s * idx_tensor_x + c * idx_tensor_y + effective_translation_tensor_y.unsqueeze(1)).int()

            # move idx tensors of the new image to 0,0 in upper left corner
            idx_tensor_x += torch.abs(torch.min(idx_tensor_x, dim=-1)[0]).unsqueeze(1)
            idx_tensor_y += torch.abs(torch.min(idx_tensor_y, dim=-1)[0]).unsqueeze(1)

            # filter_idx outside the image
            filter_idx = (
                (idx_crop_x >= 0)
                & (idx_crop_x < height_scan_shape[0])
                & (idx_crop_y >= 0)
                & (idx_crop_y < height_scan_shape[1])
            )
            idx_crop_x[~filter_idx] = 0
            idx_crop_y[~filter_idx] = 0

            new_image = torch.zeros((
                self.states.shape[0] * self.states.shape[1],
                math.ceil((y_max - y_min) / height_scan_res + 1),
                math.ceil((x_max - x_min) / height_scan_res + 1),
            ))
            ALL_INDICES = torch.arange(self.states.shape[0] * self.states.shape[1]).int()[:, None].repeat(1, 336)
            new_image[ALL_INDICES, idx_tensor_x.int(), idx_tensor_y.int()] = self.obs_exteroceptive.repeat(
                1, self.states.shape[1], 1, 1
            ).reshape(-1, *height_scan_shape)[ALL_INDICES, idx_crop_x.int(), idx_crop_y.int()]

            filter_idx_nonzero = (~filter_idx).nonzero()
            new_image[
                filter_idx_nonzero[:, 0].int(),
                idx_tensor_x[filter_idx_nonzero[:, 0], filter_idx_nonzero[:, 1]].int(),
                idx_tensor_y[filter_idx_nonzero[:, 0], filter_idx_nonzero[:, 1]].int(),
            ] = -1

            import matplotlib.pyplot as plt

            # Visualization using matplotlib
            idx = 1
            fig, axs = plt.subplots(2, 11, figsize=(55, 10))

            vmin = -1
            vmax = torch.max(self.obs_exteroceptive[idx, 0]).item()

            img = axs[0, 0].imshow(self.obs_exteroceptive[idx, 0].numpy(), cmap="viridis", vmin=vmin, vmax=vmax)
            axs[0, 0].set_title("Large Height Scan")
            axs[0, 0].set_xlabel("X")
            axs[0, 0].set_ylabel("Y")

            for i in range(10):
                print(i)
                axs[0, i + 1].imshow(
                    new_image[idx * self.states.shape[1] + i].numpy(), cmap="viridis", vmin=vmin, vmax=vmax
                )
                axs[0, i + 1].set_title(
                    f"{i}:"
                    f" {self.states[idx, i, 0].float():.4f} {self.states[idx, i, 1].float():.4f} {torch.atan2(self.states[idx, i, 2], self.states[idx, i, 3]).float():.4f}"
                )
                axs[0, i + 1].set_xlabel("X")
                axs[0, i + 1].set_ylabel("Y")

                mask = torch.zeros(*height_scan_shape, dtype=torch.bool)
                mask[idx_crop_x[idx * self.states.shape[1] + i], idx_crop_y[idx * self.states.shape[1] + i]] = True
                masked_image = torch.where(mask, self.obs_exteroceptive[idx, 0], torch.tensor(-1))

                axs[1, i + 1].imshow(masked_image.numpy(), cmap="viridis", vmin=vmin, vmax=vmax)
                axs[1, i + 1].set_xlabel("X")
                axs[1, i + 1].set_ylabel("Y")

            # Create a colorbar
            cbar = fig.colorbar(img, ax=axs, fraction=0.02, pad=0.04)
            cbar.set_label("Color Scale")

            plt.tight_layout()
            plt.savefig("height_scan.png")

        return initial_states, max_velocity_applied, max_acceleration_applied

    """
    Private functions
    """

    def _sample_random_traj_idx(self, replay_buffer: ReplayBuffer):
        device = self.replay_buffer_cfg.buffer_device
        horizon = self.model_cfg.prediction_horizon
        max_start = self.replay_buffer_cfg.trajectory_length - horizon - 1
        candidate_starts = torch.arange(1, max_start, device=device)
        num_trajectories = replay_buffer.states.shape[0]
        traj_idx = torch.arange(num_trajectories, device=device)
        traj_grid, start_grid = torch.meshgrid(traj_idx, candidate_starts, indexing="ij")

        terminal_idx = getattr(
            replay_buffer,
            "terminal_idx",
            torch.full((num_trajectories,), self.replay_buffer_cfg.trajectory_length, device=device),
        )
        valid_idx = getattr(
            replay_buffer,
            "valid_idx",
            torch.full((num_trajectories,), self.replay_buffer_cfg.trajectory_length, device=device),
        )
        valid_mask = start_grid + horizon < valid_idx[:, None]
        valid_mask &= start_grid + horizon < terminal_idx[:, None]
        valid_mask &= ~replay_buffer.states[traj_grid, start_grid, 0, 7].to(torch.bool)

        valid_pairs = torch.vstack([traj_grid[valid_mask], start_grid[valid_mask]]).T
        if valid_pairs.shape[0] == 0:
            return valid_pairs

        if valid_pairs.shape[0] >= self.cfg.num_samples:
            perm = torch.randperm(valid_pairs.shape[0], device=device)[: self.cfg.num_samples]
            return valid_pairs[perm]

        repeat_times = self.cfg.num_samples // valid_pairs.shape[0] + 1
        perm = torch.randperm(valid_pairs.shape[0], device=device)
        return valid_pairs[perm].repeat(repeat_times, 1)[: self.cfg.num_samples]

    def _sample_collision_traj(self, replay_buffer: ReplayBuffer):
        device = self.replay_buffer_cfg.buffer_device
        horizon = self.model_cfg.prediction_horizon
        terminal_idx = getattr(replay_buffer, "terminal_idx", None)
        valid_idx = getattr(replay_buffer, "valid_idx", None)

        if terminal_idx is None:
            collision_samples = torch.where(replay_buffer.states[:, 1 : -horizon + 2, 0, 7])
            collision_start_idx = torch.randint(
                2,
                horizon,
                (collision_samples[0].shape[0],),
                device=device,
            )
            collision_start_idx = torch.clip(collision_samples[1] - collision_start_idx, 0)
            return torch.vstack([collision_samples[0], collision_start_idx]).T

        if valid_idx is None:
            num_trajectories = replay_buffer.states.shape[0]
            valid_idx = torch.full(
                (num_trajectories,),
                self.replay_buffer_cfg.trajectory_length,
                device=device,
                dtype=torch.long,
            )

        valid_terminal_envs = torch.where((terminal_idx > 1) & (terminal_idx < valid_idx))[0]
        if valid_terminal_envs.shape[0] == 0:
            return torch.empty((0, 2), device=device, dtype=torch.long)

        terminal_steps = terminal_idx[valid_terminal_envs]
        valid_steps = valid_idx[valid_terminal_envs]
        start_min = torch.clamp(terminal_steps - horizon, min=1)
        start_max = torch.minimum(terminal_steps - 1, valid_steps - horizon - 1)
        valid_terminal_envs = valid_terminal_envs[start_max >= start_min]
        terminal_steps = terminal_idx[valid_terminal_envs]
        valid_steps = valid_idx[valid_terminal_envs]
        start_min = torch.clamp(terminal_steps - horizon, min=1)
        start_max = torch.minimum(terminal_steps - 1, valid_steps - horizon - 1)

        if valid_terminal_envs.shape[0] == 0:
            return torch.empty((0, 2), device=device, dtype=torch.long)

        rand = torch.rand(valid_terminal_envs.shape[0], device=device)
        start_idx = start_min + torch.floor(rand * (start_max - start_min + 1).float()).long()
        non_initial_collision = ~replay_buffer.states[valid_terminal_envs, start_idx, 0, 7].to(torch.bool)
        valid_terminal_envs = valid_terminal_envs[non_initial_collision]
        start_idx = start_idx[non_initial_collision]

        return torch.vstack([valid_terminal_envs, start_idx]).T

    def _filter_idx(self, keep_idx: torch.Tensor):
        """Filter data and only keep the given indexes. After filtering, update the number of samples"""
        # filter data
        self.state_history = self.state_history[keep_idx]
        self.obs_proprioceptive = self.obs_proprioceptive[keep_idx]
        self.actions = self.actions[keep_idx]
        self.states = self.states[keep_idx]
        self.perfect_velocity_following_local_frame = self.perfect_velocity_following_local_frame[keep_idx]
        if self.obs_exteroceptive is not None:
            self.obs_exteroceptive = self.obs_exteroceptive[keep_idx]
        if self.add_obs_exteroceptive is not None:
            self.add_obs_exteroceptive = self.add_obs_exteroceptive[keep_idx]

        # update sample number
        self._actual_nbr_samples = torch.sum(keep_idx).item()

    def _geometric_collision_target_from_height_scan(self) -> torch.Tensor:
        """Build a geometry-only collision target from the initial height scan and future local states."""
        if self.obs_exteroceptive is None:
            return torch.zeros(
                self.states.shape[0],
                self.states.shape[1],
                device=self.states.device,
                dtype=self.states.dtype,
            )

        height_scan = self.obs_exteroceptive.float()
        if height_scan.dim() == 4:
            height_scan = height_scan.squeeze(1)
        if height_scan.dim() != 3:
            return torch.zeros(
                self.states.shape[0],
                self.states.shape[1],
                device=self.states.device,
                dtype=self.states.dtype,
            )

        batch_size, height, width = height_scan.shape
        traj_len = self.states.shape[1]
        resolution = float(getattr(self.model_cfg, "geometric_label_scan_resolution", 0.1))
        radius = float(getattr(self.model_cfg, "geometric_label_robot_radius", 0.2))
        height_threshold = float(getattr(self.model_cfg, "geometric_label_height_threshold", 0.25))
        forward_offset = float(getattr(self.model_cfg, "geometric_label_forward_offset", 0.5))

        footprint_radius_px = max(int(round(radius / resolution)), 0)
        local_offsets = torch.arange(
            -footprint_radius_px, footprint_radius_px + 1, device=self.states.device, dtype=torch.long
        )
        off_x, off_y = torch.meshgrid(local_offsets, local_offsets, indexing="ij")
        footprint_mask = (off_x.float() ** 2 + off_y.float() ** 2) <= float(footprint_radius_px**2)
        off_x = off_x[footprint_mask]
        off_y = off_y[footprint_mask]

        yaw = torch.atan2(self.states[..., 2], self.states[..., 3])
        cos_yaw = torch.cos(yaw).unsqueeze(-1)
        sin_yaw = torch.sin(yaw).unsqueeze(-1)
        rot_x = (cos_yaw * off_x.float() - sin_yaw * off_y.float()).round().long()
        rot_y = (sin_yaw * off_x.float() + cos_yaw * off_y.float()).round().long()

        center_x = height / 2
        center_y = forward_offset / resolution
        idx_x = (-self.states[..., 1] / resolution + center_x).round().long()
        idx_y = (self.states[..., 0] / resolution + center_y).round().long()
        px = (idx_x.unsqueeze(-1) + rot_x).clamp(0, height - 1)
        py = (idx_y.unsqueeze(-1) + rot_y).clamp(0, width - 1)

        batch_idx = torch.arange(batch_size, device=self.states.device)[:, None, None].expand(
            batch_size, traj_len, px.shape[-1]
        )
        obstacle = height_scan.to(self.states.device) > height_threshold
        return obstacle[batch_idx, px, py].any(dim=-1).to(dtype=self.states.dtype)

    """
    Static helper functions
    """

    @staticmethod
    def state_history_transformer(
        replay_buffer: ReplayBuffer,
        start_idx: torch.Tensor,
        initial_states: torch.Tensor,
        history_length: int,
        exclude_index: list[int] | None = None,
    ):
        """transform the state history into the local robot frame

        Individual function as also used for evaluation call when the model should only do predictions.
        """
        # repeat initial state to match the state history
        initial_states_SE3 = pp.SE3(initial_states.repeat(1, history_length, 1).reshape(-1, 7))
        # transform the state history into the local robot frame
        state_history = replay_buffer.states[start_idx[:, 0], start_idx[:, 1], :, :7]
        state_history = pp.SE3(state_history.reshape(-1, 7))
        state_history_local = (pp.Inv(initial_states_SE3) * state_history).tensor()
        state_history_pos = state_history_local.reshape(-1, history_length, 7)[..., :2]
        state_history_yaw = math_utils.euler_xyz_from_quat(state_history_local[..., [6, 3, 4, 5]])[2]
        # rotation encoded as [sin(yaw), cos(yaw)] to avoid jump in representation
        # Check: Learning with 3D rotations, a hitchhiker’s guide to SO(3), 2024, Frey et al.
        state_history_yaw = torch.stack([torch.sin(state_history_yaw), torch.cos(state_history_yaw)], dim=1)
        state_history_yaw = state_history_yaw.reshape(-1, history_length, 2)
        # get the rest of the state and potentially exclude some indices
        rest_of_state = replay_buffer.states[start_idx[:, 0], start_idx[:, 1], :, 7:]
        if exclude_index is not None:
            keep_idx = torch.ones(replay_buffer.states.shape[-1], device=replay_buffer.states.device, dtype=torch.bool)
            keep_idx[exclude_index] = False
            keep_idx = keep_idx[7:]
            rest_of_state = rest_of_state[..., keep_idx]
        # final state history: [N, History Length, 3 (pos) + 2 (yaw) + 1 (collision) + rest of the state]
        return torch.concatenate([state_history_pos, state_history_yaw, rest_of_state], dim=2)

    """
    Properties called when accessing the data
    """

    def __len__(self):
        return self._actual_nbr_samples

    def __getitem__(self, index: int):
        # get extereoceptive and apply noise model
        if self.obs_exteroceptive is not None and self.extereoceptive_noise_model is None:
            exteroceptive = self.obs_exteroceptive[index].type(torch.float32)
        elif self.obs_exteroceptive is not None:
            exteroceptive = self.extereoceptive_noise_model(self.obs_exteroceptive[index].type(torch.float32))
        else:
            exteroceptive = torch.zeros(1)

        # get additional exteroceptive observation
        if self.add_obs_exteroceptive is not None:
            add_exteroceptive = self.add_obs_exteroceptive[index].type(torch.float32)
        else:
            add_exteroceptive = torch.zeros(1)

        return (
            # model inputs
            self.state_history[index],
            self.obs_proprioceptive[index],
            exteroceptive,
            self.actions[index],
            add_exteroceptive,
            # model targets
            self.states[index],
            # eval data
            self.perfect_velocity_following_local_frame[index],
        )
