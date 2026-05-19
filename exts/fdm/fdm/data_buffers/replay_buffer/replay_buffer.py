# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv

from fdm.model.fdm_model_cfg import FDMBaseModelCfg

from .replay_buffer_cfg import ReplayBufferCfg


class ReplayBuffer:
    """A replay buffer with support for training/validation iterators and ensembles."""

    def __init__(
        self,
        cfg: ReplayBufferCfg,
        model_cfg: FDMBaseModelCfg,
        env: ManagerBasedRLEnv,
    ):
        # get parameters
        self.cfg = cfg
        self.model_cfg = model_cfg

        # get env
        self.env: ManagerBasedRLEnv = env

        # exteroceptive observation flags
        self._has_exteroceptive_observation = (
            "fdm_obs_exteroceptive" in self.env.observation_manager.group_obs_dim
        )
        self._has_add_exteroceptive_observation = (
            "fdm_add_obs_exteroceptive" in self.env.observation_manager.group_obs_dim
        )

        # init buffers
        self._init_buffers()

        # parameters
        self._ALL_INDICES = torch.arange(
            self.env.num_envs, device=self.device, dtype=torch.long
        )

    """
    Properties
    """

    @property
    def data_collection_interval(self):
        """The interval at which data is collected."""
        return self._data_collection_interval

    @property
    def history_collection_interval(self):
        """The interval at which history is collected."""
        return self._history_collection_interval

    @property
    def env_buffer_filled(self):
        return self.fill_idx >= self.cfg.trajectory_length

    @property
    def is_filled(self) -> bool:
        return torch.all(self.env_buffer_filled)

    @property
    def fill_ratio(self) -> float:
        return torch.mean(self.fill_idx / self.cfg.trajectory_length).item()

    @property
    def state_dim(self) -> tuple[int, ...]:
        return self.env.observation_manager.group_obs_dim["fdm_state"]

    @property
    def proprioceptive_observation_dim(self) -> tuple[int, ...]:
        return self.env.observation_manager.group_obs_dim["fdm_obs_proprioception"]

    @property
    def exteroceptive_observation_dim(self) -> tuple[int, ...] | None:
        if self._has_exteroceptive_observation:
            return self.env.observation_manager.group_obs_dim["fdm_obs_exteroceptive"]
        return None

    @property
    def add_exteroceptive_observation_dim(self) -> tuple[int, ...] | None:
        if self._has_add_exteroceptive_observation:
            return self.env.observation_manager.group_obs_dim["fdm_add_obs_exteroceptive"]
        return None

    @property
    def action_dim(self) -> int:
        return self.env.action_manager.action.shape[1]

    @property
    def device(self) -> str:
        return self.cfg.buffer_device

    """
    Operations to fill the buffer
    """

    def add(
        self,
        states: torch.Tensor,
        obersevations_proprioceptive: torch.Tensor,
        obersevations_exteroceptive: torch.Tensor | None,
        actions: torch.Tensor,
        dones: torch.Tensor,
        feet_contact: torch.Tensor,
        add_observation_exteroceptive: torch.Tensor | None = None,
    ):
        # IMPORTANT:
        # In Isaac/manager-style envs, when done=True, the returned obs/state is often already from the reset episode.
        # So we must clear ALL local collector state for done envs BEFORE using current states.
        done_env_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1)
        if done_env_ids.numel() > 0:
            self.reset_local_history(done_env_ids)

        # raw collision from state label
        colliding_envs = states[..., 7].to(torch.bool)

        # current obs of done envs belongs to a fresh/reset episode -> never treat it as a valid collision sample
        colliding_envs[dones] = False

        # keep the original FDM-style "do not trust very first recording window"
        colliding_envs[self.env_step_counter < self._data_collection_interval] = False

        # update local history / touchdown bookkeeping
        self._update_local_history_buffers(
            colliding_envs, states, obersevations_proprioceptive, feet_contact
        )

        # update collector arm state
        self._update_collection_armed(states, actions)

        # update full trajectory buffers
        self._update_full_trajectory_buffers(
            colliding_envs,
            obersevations_exteroceptive,
            actions,
            feet_contact,
            add_observation_exteroceptive,
        )

        # step counters
        self.env_step_counter += 1
        self._steps_since_arm[self._collector_armed] += 1
    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset the full buffer state for the given environments."""
        if env_ids is None:
            env_ids = self._ALL_INDICES

        self.env_step_counter[env_ids] = 0
        self.fill_idx[env_ids] = 0

        self.local_state_history[env_ids] = 0
        self.local_proprioceptive_observation_history[env_ids] = 0

        self.states[env_ids] = 0
        self.observations_proprioceptive[env_ids] = 0
        self.actions[env_ids] = 0

        if self._has_add_exteroceptive_observation:
            self.add_observations_exteroceptive[env_ids] = 0
        if self._has_exteroceptive_observation:
            self.observations_exteroceptive[env_ids] = 0

        self._has_touched_ground[env_ids] = False
        self._steps_since_first_touchdown[env_ids] = 0

        self._collector_armed[env_ids] = False
        self._steps_since_arm[env_ids] = 0
        self._touchdown_xy[env_ids] = 0.0
        self._cmd_active_counter[env_ids] = 0
        self._first_interval_after_arm_pending[env_ids] = False

    def reset_local_history(self, env_ids):
        """Reset only local collector state for environments that just got reset in sim."""
        self.env_step_counter[env_ids] = 0
        self.local_state_history[env_ids] = 0
        self.local_proprioceptive_observation_history[env_ids] = 0

        self._has_touched_ground[env_ids] = False
        self._steps_since_first_touchdown[env_ids] = 0

        self._collector_armed[env_ids] = False
        self._steps_since_arm[env_ids] = 0
        self._touchdown_xy[env_ids] = 0.0
        self._cmd_active_counter[env_ids] = 0
        self._first_interval_after_arm_pending[env_ids] = False

    def fill_leftover_envs(self):
        """Fill the buffer for the environments that are not yet filled."""
        collision_indices = torch.nonzero(torch.any(self.states[..., 7], dim=-1))

        if collision_indices.numel() == 0:
            envs_to_fill = self._ALL_INDICES[~self.env_buffer_filled].type(torch.long)
            source_env_idxs = self._ALL_INDICES[self.fill_idx >= self.cfg.trajectory_length]
            if source_env_idxs.shape[0] == 0:
                print("[Warning]: No source environments available to fill leftover envs.")
                return
            if source_env_idxs.shape[0] < len(envs_to_fill):
                repeat_times = len(envs_to_fill) // source_env_idxs.shape[0]
                source_env_idxs = source_env_idxs.repeat(repeat_times + 1)[: len(envs_to_fill)]

            for target_env_idx, source_env_idx in zip(envs_to_fill, source_env_idxs):
                self.states[target_env_idx] = self.states[source_env_idx]
                self.observations_proprioceptive[target_env_idx] = self.observations_proprioceptive[source_env_idx]
                if self._has_exteroceptive_observation:
                    self.observations_exteroceptive[target_env_idx] = self.observations_exteroceptive[source_env_idx]
                self.actions[target_env_idx] = self.actions[source_env_idx]
                if self._has_add_exteroceptive_observation:
                    self.add_observations_exteroceptive[target_env_idx] = (
                        self.add_observations_exteroceptive[source_env_idx]
                    )
            self.fill_idx[envs_to_fill] = self.cfg.trajectory_length
            return

        collision_envs, unique_indices = torch.unique(collision_indices[:, 0], return_inverse=True)
        env_split_data = torch.split(collision_indices[:, 1], torch.bincount(unique_indices).tolist())
        collision_max_indices = torch.tensor(
            [torch.max(env_indices) for env_indices in env_split_data], device=self.device
        )

        # add 1 to avoid cropping the collision event itself
        collision_max_indices += 1

        not_collided_envs = list(set(self._ALL_INDICES.tolist()) - set(collision_envs.tolist()))
        collision_envs = torch.concatenate(
            (collision_envs, torch.tensor(not_collided_envs, device=self.device))
        )
        collision_max_indices = torch.concatenate(
            (collision_max_indices, torch.zeros(len(not_collided_envs), device=self.device, dtype=torch.long))
        )

        collision_envs, sort_indices = collision_envs.sort()
        collision_max_indices = collision_max_indices[sort_indices]

        envs_to_fill = collision_envs[~self.env_buffer_filled].type(torch.long)
        env_fill_from_indices = collision_max_indices[~self.env_buffer_filled].type(torch.long)

        source_env_idxs = self._ALL_INDICES[
            self.fill_idx >= self.cfg.trajectory_length - torch.min(env_fill_from_indices)
        ]
        if source_env_idxs.shape[0] < len(envs_to_fill):
            print("[Warning]: Not enough environments to fill the buffer. Repeating the source environments.")
            repeat_times = len(envs_to_fill) // source_env_idxs.shape[0]
            source_env_idxs = source_env_idxs.repeat(repeat_times + 1)[: len(envs_to_fill)]

        for target_env_idx, source_env_idx, collision_max_idx in zip(
            envs_to_fill, source_env_idxs, env_fill_from_indices
        ):
            use_until_idx = int(self.cfg.trajectory_length - collision_max_idx)
            self.states[target_env_idx, int(collision_max_idx):] = self.states[source_env_idx, :use_until_idx]
            self.observations_proprioceptive[target_env_idx, int(collision_max_idx):] = (
                self.observations_proprioceptive[source_env_idx, :use_until_idx]
            )
            if self._has_exteroceptive_observation:
                self.observations_exteroceptive[target_env_idx, int(collision_max_idx):] = (
                    self.observations_exteroceptive[source_env_idx, :use_until_idx]
                )
            self.actions[target_env_idx, int(collision_max_idx):] = self.actions[source_env_idx, :use_until_idx]
            if self._has_add_exteroceptive_observation:
                self.add_observations_exteroceptive[target_env_idx, int(collision_max_idx):] = (
                    self.add_observations_exteroceptive[source_env_idx, :use_until_idx]
                )

        self.fill_idx[envs_to_fill] = self.cfg.trajectory_length

    """
    Helper functions
    """

    def _init_buffers(self):
        # keep your current sampling frequency unchanged
        self._data_collection_interval = (self.model_cfg.command_timestep / self.env.step_dt)
        print(f"Data collection interval: {self._data_collection_interval} steps")
        if self._data_collection_interval % 1 != 0:
            print("[WARNING]: Data collection interval is not an integer. Can influence data collection.")

        if self.model_cfg.history_time_step is not None:
            self._history_collection_interval = self.model_cfg.history_time_step / self.env.step_dt
        else:
            self._history_collection_interval = self._data_collection_interval / self.model_cfg.history_length

        assert self._history_collection_interval >= 1, (
            "History collection frequency calculated as must be larger than physics frequency! "
            "Decrease history length as collection timestep is calculated by division of the command timestep "
            "by the history length."
        )
        if self._history_collection_interval % 1 != 0:
            print(
                "[WARNING]: History collection interval is not an integer. Will make data collection "
                "not equidistant, i.e. with an interval of 2.5 will sample at env step [3, 5, 8, 10, 13, ...]."
            )

        # full trajectory buffers
        self.states = torch.zeros(
            (self.env.num_envs, self.cfg.trajectory_length, self.model_cfg.history_length, *(self.state_dim)),
            device=self.device,
        )
        self.observations_proprioceptive = torch.zeros(
            (
                self.env.num_envs,
                self.cfg.trajectory_length,
                self.model_cfg.history_length,
                *(self.proprioceptive_observation_dim),
            ),
            device=self.device,
        )

        if self._has_exteroceptive_observation:
            self.observations_exteroceptive = torch.zeros(
                (self.env.num_envs, self.cfg.trajectory_length, *(self.exteroceptive_observation_dim)),
                device=self.device,
                dtype=getattr(torch, self.cfg.exteroceptive_obs_precision),
            )
        else:
            self.observations_exteroceptive = None

        if self._has_add_exteroceptive_observation:
            self.add_observations_exteroceptive = torch.zeros(
                (self.env.num_envs, self.cfg.trajectory_length, *(self.add_exteroceptive_observation_dim)),
                device=self.device,
                dtype=getattr(torch, self.cfg.exteroceptive_obs_precision),
            )
        else:
            self.add_observations_exteroceptive = None

        self.actions = torch.zeros(
            (self.env.num_envs, self.cfg.trajectory_length, self.action_dim),
            device=self.device,
        )

        # local history buffers
        self.local_state_history = torch.zeros(
            (self.env.num_envs, self.model_cfg.history_length, *(self.state_dim)),
            device=self.device,
        )
        self.local_proprioceptive_observation_history = torch.zeros(
            (self.env.num_envs, self.model_cfg.history_length, *(self.proprioceptive_observation_dim)),
            device=self.device,
        )

        # index buffers
        self.fill_idx = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.long)
        self.env_step_counter = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.long)

        # touchdown / anti-reset-pollution bookkeeping
        self._has_touched_ground = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.bool)
        self._steps_since_first_touchdown = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.long)

        # collector arm state
        self._collector_armed = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.bool)
        self._steps_since_arm = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.long)
        self._touchdown_xy = torch.zeros((self.env.num_envs, 2), device=self.device)
        self._cmd_active_counter = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.long)
        self._first_interval_after_arm_pending = torch.zeros(
            self.env.num_envs, device=self.device, dtype=torch.bool
        )

    def _update_local_history_buffers(
        self,
        colliding_envs: torch.Tensor,
        state: torch.Tensor,
        obersevations_proprioceptive: torch.Tensor,
        feet_contact: torch.Tensor,
    ):
        colliding_envs = colliding_envs.to(self.device, dtype=torch.bool)
        feet_contact = feet_contact.to(self.device)
        feet_any = feet_contact.any(dim=-1) if feet_contact.ndim == 2 else feet_contact.to(torch.bool)

        # touchdown bookkeeping
        prev_touched = self._has_touched_ground.clone()
        self._has_touched_ground |= feet_any

        new_touchdown = (~prev_touched) & feet_any
        self._steps_since_first_touchdown[new_touchdown] = 0
        self._touchdown_xy[new_touchdown] = state[new_touchdown, :2].to(self.device)

        touched_mask = self._has_touched_ground
        self._steps_since_first_touchdown[touched_mask] += 1

        # keep original FDM-style colliding update, but only after collector is armed
        valid_colliding_envs = colliding_envs & self._collector_armed

        updatable_envs = ((self.env_step_counter % self._history_collection_interval) == 0).to(self.device)
        updatable_envs |= valid_colliding_envs
        updatable_envs &= self._has_touched_ground

        self.local_state_history[updatable_envs] = torch.roll(
            self.local_state_history[updatable_envs], 1, dims=1
        )
        self.local_state_history[updatable_envs, 0] = state[updatable_envs].to(self.device)

        self.local_proprioceptive_observation_history[updatable_envs] = torch.roll(
            self.local_proprioceptive_observation_history[updatable_envs], 1, dims=1
        )
        self.local_proprioceptive_observation_history[updatable_envs, 0] = (
            obersevations_proprioceptive[updatable_envs].to(self.device)
        )

    def _update_collection_armed(self, state: torch.Tensor, actions: torch.Tensor):
        """
        Arm the collector only after the env has clearly left reset-settling phase.

        Once armed:
          - collision samples are allowed to use continuous colliding update
          - full trajectory samples are allowed to be written

        Critical anti-pollution step:
          on the FIRST not-armed -> armed transition, clear local history and reset steps_since_arm,
          so pre-armed reset/fall traces cannot leak into stored windows.
        """
        xy = state[:, :2].to(self.device)
        actions = actions.to(self.device)

        # align with your hard_failure command threshold
        cmd_mag = torch.norm(actions[:, :2], dim=-1) + 0.5 * torch.abs(actions[:, 2])
        cmd_active = cmd_mag > 0.15

        active_mask = self._has_touched_ground & cmd_active
        self._cmd_active_counter[active_mask] += 1
        self._cmd_active_counter[~active_mask] = 0

        progress_from_touchdown = torch.norm(xy - self._touchdown_xy, dim=-1)

        # stricter than quadruped: explicitly block reset-settling pseudo-falls
        arm_min_steps = 8
        arm_progress_threshold = 0.15
        arm_cmd_steps = 3

        arm_now = (
            self._has_touched_ground
            & (self._steps_since_first_touchdown >= arm_min_steps)
            & (progress_from_touchdown > arm_progress_threshold)
            & (self._cmd_active_counter >= arm_cmd_steps)
        )

        newly_armed = (~self._collector_armed) & arm_now
        self._collector_armed |= arm_now

        if torch.any(newly_armed):
            self.local_state_history[newly_armed] = 0
            self.local_proprioceptive_observation_history[newly_armed] = 0
            self._steps_since_arm[newly_armed] = 0
            self._first_interval_after_arm_pending[newly_armed] = True

    def _update_full_trajectory_buffers(
        self,
        colliding_envs: torch.Tensor,
        obersevations_exteroceptive: torch.Tensor | None,
        actions: torch.Tensor,
        feet_contact: torch.Tensor,
        add_observation_exteroceptive: torch.Tensor | None,
    ):
        colliding_envs = colliding_envs.to(self.device, dtype=torch.bool)

        steps = self._steps_since_first_touchdown
        arm_steps = self._steps_since_arm
        interval = int(self._data_collection_interval)
        post_touchdown_block_steps = 3

        hist = self.local_state_history
        xy_now = hist[:, 0, :2]
        xy_old = hist[:, -1, :2]
        move_dist = torch.norm(xy_now - xy_old, dim=-1)

        min_move_dist = 0.08
        meaningful_motion = move_dist > min_move_dist

        # -------- regular sampling path only --------
        sampling_mask = (
            (steps > 0)
            & (steps % interval == 0)
        )
        sampling_mask &= self._has_touched_ground
        sampling_mask &= (steps >= post_touchdown_block_steps)
        sampling_mask &= meaningful_motion
        sampling_mask &= (self.env_step_counter > 0)
        sampling_mask &= self._collector_armed
        sampling_mask &= (arm_steps > 0)  # block same-step arm write

        first_valid = (
            self._has_touched_ground
            & (steps == post_touchdown_block_steps)
            & meaningful_motion
            & self._collector_armed
            & (arm_steps > 0)
        )
        sampling_mask |= first_valid

        terminal_mask = (
            colliding_envs
            & self._collector_armed
            & self._has_touched_ground
            & (arm_steps > 0)
            & ~self.env_buffer_filled
        )

        terminal_idxs = self._ALL_INDICES[terminal_mask]
        if len(terminal_idxs) > 0:
            self._write_terminal_and_fill(
                terminal_idxs,
                obersevations_exteroceptive,
                actions,
                add_observation_exteroceptive,
            )
            self.reset_local_history(terminal_idxs)

        sampling_mask &= ~terminal_mask
        sampling_mask &= ~colliding_envs

        # quadruped-style protection can stay, but is now mostly redundant
        first_interval_after_arm_hit = (
            self._first_interval_after_arm_pending
            & self._collector_armed
            & (arm_steps > 0)
            & self._has_touched_ground
            & (steps > 0)
            & (steps % interval == 0)
        )
        self._first_interval_after_arm_pending[first_interval_after_arm_hit] = False

        # -------- write --------
        updatable_idxs = self._ALL_INDICES[sampling_mask]
        env_non_full = ~self.env_buffer_filled[updatable_idxs]
        updatable_idxs = updatable_idxs[env_non_full]

        if len(updatable_idxs) == 0:
            return

        self.states[updatable_idxs, self.fill_idx[updatable_idxs]] = (
            self.local_state_history[updatable_idxs].clone()
        )
        self.observations_proprioceptive[updatable_idxs, self.fill_idx[updatable_idxs]] = (
            self.local_proprioceptive_observation_history[updatable_idxs].clone()
        )

        if self._has_exteroceptive_observation:
            self.observations_exteroceptive[updatable_idxs, self.fill_idx[updatable_idxs]] = (
                obersevations_exteroceptive[updatable_idxs]
                .to(self.device)
                .type(getattr(torch, self.cfg.exteroceptive_obs_precision))
            )

        self.actions[updatable_idxs, self.fill_idx[updatable_idxs]] = actions[updatable_idxs].to(self.device)

        if self._has_add_exteroceptive_observation:
            self.add_observations_exteroceptive[updatable_idxs, self.fill_idx[updatable_idxs]] = (
                add_observation_exteroceptive[updatable_idxs]
                .to(self.device)
                .type(getattr(torch, self.cfg.exteroceptive_obs_precision))
            )

        self.fill_idx[updatable_idxs] += 1

    def _write_terminal_and_fill(
        self,
        env_ids: torch.Tensor,
        obersevations_exteroceptive: torch.Tensor | None,
        actions: torch.Tensor,
        add_observation_exteroceptive: torch.Tensor | None,
    ):
        """Write the terminal frame and make the remaining fixed buffer an absorbing terminal state."""
        env_ids = env_ids.to(self.device, dtype=torch.long)
        write_idx = self.fill_idx[env_ids]

        self.states[env_ids, write_idx] = self.local_state_history[env_ids].clone()
        self.observations_proprioceptive[env_ids, write_idx] = (
            self.local_proprioceptive_observation_history[env_ids].clone()
        )
        self.actions[env_ids, write_idx] = actions[env_ids].to(self.device)

        if self._has_exteroceptive_observation:
            self.observations_exteroceptive[env_ids, write_idx] = (
                obersevations_exteroceptive[env_ids]
                .to(self.device)
                .type(getattr(torch, self.cfg.exteroceptive_obs_precision))
            )

        if self._has_add_exteroceptive_observation:
            self.add_observations_exteroceptive[env_ids, write_idx] = (
                add_observation_exteroceptive[env_ids]
                .to(self.device)
                .type(getattr(torch, self.cfg.exteroceptive_obs_precision))
            )

        for env_id, terminal_idx in zip(env_ids.tolist(), write_idx.tolist()):
            terminal_idx = int(terminal_idx)
            if terminal_idx + 1 >= self.cfg.trajectory_length:
                continue
            self.states[env_id, terminal_idx + 1 :] = self.states[env_id, terminal_idx].unsqueeze(0)
            self.observations_proprioceptive[env_id, terminal_idx + 1 :] = (
                self.observations_proprioceptive[env_id, terminal_idx].unsqueeze(0)
            )
            self.actions[env_id, terminal_idx + 1 :] = self.actions[env_id, terminal_idx].unsqueeze(0)
            if self._has_exteroceptive_observation:
                self.observations_exteroceptive[env_id, terminal_idx + 1 :] = (
                    self.observations_exteroceptive[env_id, terminal_idx].unsqueeze(0)
                )
            if self._has_add_exteroceptive_observation:
                self.add_observations_exteroceptive[env_id, terminal_idx + 1 :] = (
                    self.add_observations_exteroceptive[env_id, terminal_idx].unsqueeze(0)
                )

        self.fill_idx[env_ids] = self.cfg.trajectory_length
