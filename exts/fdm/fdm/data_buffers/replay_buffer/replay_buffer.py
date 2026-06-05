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
        self._ALL_SLOT_INDICES = torch.arange(
            self.num_slots, device=self.device, dtype=torch.long
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
        return self.env_slot_idx < 0

    @property
    def is_filled(self) -> bool:
        return torch.all(self.slot_closed)

    @property
    def fill_ratio(self) -> float:
        slot_progress = self.slot_fill_idx.float() / float(self.cfg.trajectory_length)
        slot_progress = torch.clamp(slot_progress, max=1.0)
        return torch.mean(torch.where(self.slot_closed, torch.ones_like(slot_progress), slot_progress)).item()

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

    @property
    def fill_idx(self) -> torch.Tensor:
        fill_idx = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.long)
        active = self.env_slot_idx >= 0
        if torch.any(active):
            fill_idx[active] = self.slot_fill_idx[self.env_slot_idx[active]]
        return fill_idx

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
        done_env_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1)

        # raw collision from state label
        colliding_envs = states[..., 7].to(torch.bool)

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
            dones,
        )

        # step counters
        self.env_step_counter += 1
        self._steps_since_arm[self._collector_armed] += 1

        if done_env_ids.numel() > 0:
            self.reset_local_history(done_env_ids)

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset the full buffer state for the given environments."""
        if env_ids is None:
            env_ids = self._ALL_INDICES
            self.slot_fill_idx[:] = 0
            self.terminal_idx[:] = self.cfg.trajectory_length
            self.valid_idx[:] = self.cfg.trajectory_length
            self.slot_closed[:] = False
            self.states[:] = 0
            self.observations_proprioceptive[:] = 0
            self.actions[:] = 0
            if self._has_add_exteroceptive_observation:
                self.add_observations_exteroceptive[:] = 0
            if self._has_exteroceptive_observation:
                self.observations_exteroceptive[:] = 0
            self._next_free_slot = 0
            self.env_slot_idx[:] = -1
            self._assign_new_slots(env_ids)
        else:
            env_ids = env_ids.to(self.device, dtype=torch.long)
            active_slots = self.env_slot_idx[env_ids]
            active_mask = active_slots >= 0
            slots_to_clear = active_slots[active_mask]
            if slots_to_clear.numel() > 0:
                self.slot_fill_idx[slots_to_clear] = 0
                self.terminal_idx[slots_to_clear] = self.cfg.trajectory_length
                self.valid_idx[slots_to_clear] = self.cfg.trajectory_length
                self.slot_closed[slots_to_clear] = False
                self.states[slots_to_clear] = 0
                self.observations_proprioceptive[slots_to_clear] = 0
                self.actions[slots_to_clear] = 0
                if self._has_add_exteroceptive_observation:
                    self.add_observations_exteroceptive[slots_to_clear] = 0
                if self._has_exteroceptive_observation:
                    self.observations_exteroceptive[slots_to_clear] = 0
            self._assign_new_slots(env_ids[~active_mask])

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

    def reset_local_history(self, env_ids):
        """Reset only local collector state for environments that just got reset in sim."""
        env_ids = env_ids.to(self.device, dtype=torch.long)
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

        needs_slot = env_ids[self.env_slot_idx[env_ids] < 0]
        self._assign_new_slots(needs_slot)

    def _assign_new_slots(self, env_ids: torch.Tensor):
        """Assign fresh trajectory slots to envs that can still contribute data."""
        env_ids = env_ids.to(self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        remaining = self.num_slots - int(self._next_free_slot)
        if remaining <= 0:
            self.env_slot_idx[env_ids] = -1
            return

        assign_count = min(env_ids.numel(), remaining)
        assign_envs = env_ids[:assign_count]
        new_slots = torch.arange(
            self._next_free_slot,
            self._next_free_slot + assign_count,
            device=self.device,
            dtype=torch.long,
        )
        self.env_slot_idx[assign_envs] = new_slots
        self.slot_fill_idx[new_slots] = 0
        self.terminal_idx[new_slots] = self.cfg.trajectory_length
        self.valid_idx[new_slots] = self.cfg.trajectory_length
        self.slot_closed[new_slots] = False
        self.states[new_slots] = 0
        self.observations_proprioceptive[new_slots] = 0
        self.actions[new_slots] = 0
        if self._has_add_exteroceptive_observation:
            self.add_observations_exteroceptive[new_slots] = 0
        if self._has_exteroceptive_observation:
            self.observations_exteroceptive[new_slots] = 0
        self._next_free_slot += assign_count

        if assign_count < env_ids.numel():
            self.env_slot_idx[env_ids[assign_count:]] = -1

    def fill_leftover_envs(self):
        """Close every remaining open trajectory slot without synthetic copying."""
        open_slots = ~self.slot_closed
        if not torch.any(open_slots):
            return

        min_valid_length = self.model_cfg.prediction_horizon + 2
        valid_open_slots = open_slots & (self.slot_fill_idx >= min_valid_length)
        invalid_open_slots = open_slots & ~valid_open_slots

        self.valid_idx[valid_open_slots] = self.slot_fill_idx[valid_open_slots]
        self.valid_idx[invalid_open_slots] = 0
        self.terminal_idx[invalid_open_slots] = 0
        self.slot_fill_idx[invalid_open_slots] = 0
        self.slot_closed[open_slots] = True
        self.env_slot_idx[:] = -1

    """
    Helper functions
    """

    def _init_buffers(self):
        self.num_slots = self.env.num_envs * int(getattr(self.cfg, "slot_multiplier", 1))
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
            (self.num_slots, self.cfg.trajectory_length, self.model_cfg.history_length, *(self.state_dim)),
            device=self.device,
        )
        self.observations_proprioceptive = torch.zeros(
            (
                self.num_slots,
                self.cfg.trajectory_length,
                self.model_cfg.history_length,
                *(self.proprioceptive_observation_dim),
            ),
            device=self.device,
        )

        if self._has_exteroceptive_observation:
            self.observations_exteroceptive = torch.zeros(
                (self.num_slots, self.cfg.trajectory_length, *(self.exteroceptive_observation_dim)),
                device=self.device,
                dtype=getattr(torch, self.cfg.exteroceptive_obs_precision),
            )
        else:
            self.observations_exteroceptive = None

        if self._has_add_exteroceptive_observation:
            self.add_observations_exteroceptive = torch.zeros(
                (self.num_slots, self.cfg.trajectory_length, *(self.add_exteroceptive_observation_dim)),
                device=self.device,
                dtype=getattr(torch, self.cfg.exteroceptive_obs_precision),
            )
        else:
            self.add_observations_exteroceptive = None

        self.actions = torch.zeros(
            (self.num_slots, self.cfg.trajectory_length, self.action_dim),
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
        self.slot_fill_idx = torch.zeros(self.num_slots, device=self.device, dtype=torch.long)
        self.terminal_idx = torch.full(
            (self.num_slots,),
            self.cfg.trajectory_length,
            device=self.device,
            dtype=torch.long,
        )
        self.valid_idx = torch.full(
            (self.num_slots,),
            self.cfg.trajectory_length,
            device=self.device,
            dtype=torch.long,
        )
        self.slot_closed = torch.zeros(self.num_slots, device=self.device, dtype=torch.bool)
        self.env_slot_idx = torch.full((self.env.num_envs,), -1, device=self.device, dtype=torch.long)
        self._next_free_slot = 0
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
        dones: torch.Tensor,
    ):
        colliding_envs = colliding_envs.to(self.device, dtype=torch.bool)
        dones = dones.to(self.device, dtype=torch.bool)

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

        active_slots = self.env_slot_idx.clamp(min=0)
        active_envs = self.env_slot_idx >= 0
        has_slot_history = active_envs & (self.slot_fill_idx[active_slots] > 0)
        collision_sample_mask = (
            colliding_envs
            & self._collector_armed
            & self._has_touched_ground
            & (arm_steps > 0)
            & has_slot_history
            & ~dones
        )
        first_collision_mask = (
            collision_sample_mask
            & (self.terminal_idx[active_slots] == self.cfg.trajectory_length)
        )
        sampling_mask |= collision_sample_mask
        sampling_mask &= ~dones

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

        first_collision_idxs = self._ALL_INDICES[first_collision_mask]
        if len(first_collision_idxs) > 0:
            first_collision_slots = self.env_slot_idx[first_collision_idxs]
            self.terminal_idx[first_collision_slots] = self.slot_fill_idx[first_collision_slots]

        min_valid_length = self.model_cfg.prediction_horizon + 2
        close_done_mask = dones & active_envs & (self.slot_fill_idx[active_slots] >= min_valid_length)
        close_done_idxs = self._ALL_INDICES[close_done_mask]
        if len(close_done_idxs) > 0:
            close_slots = self.env_slot_idx[close_done_idxs]
            self.valid_idx[close_slots] = self.slot_fill_idx[close_slots]
            self.slot_closed[close_slots] = True
            self.env_slot_idx[close_done_idxs] = -1

        discard_done_mask = dones & active_envs & (self.slot_fill_idx[active_slots] < min_valid_length)
        discard_done_idxs = self._ALL_INDICES[discard_done_mask]
        if len(discard_done_idxs) > 0:
            discard_slots = self.env_slot_idx[discard_done_idxs]
            self.slot_fill_idx[discard_slots] = 0
            self.terminal_idx[discard_slots] = self.cfg.trajectory_length
            self.valid_idx[discard_slots] = self.cfg.trajectory_length
            self.slot_closed[discard_slots] = False
            self.states[discard_slots] = 0
            self.observations_proprioceptive[discard_slots] = 0
            self.actions[discard_slots] = 0
            if self._has_add_exteroceptive_observation:
                self.add_observations_exteroceptive[discard_slots] = 0
            if self._has_exteroceptive_observation:
                self.observations_exteroceptive[discard_slots] = 0

        # -------- write --------
        updatable_idxs = self._ALL_INDICES[sampling_mask]
        env_active = self.env_slot_idx[updatable_idxs] >= 0
        updatable_idxs = updatable_idxs[env_active]

        if len(updatable_idxs) == 0:
            return
        updatable_slots = self.env_slot_idx[updatable_idxs]

        self.states[updatable_slots, self.slot_fill_idx[updatable_slots]] = (
            self.local_state_history[updatable_idxs].clone()
        )
        self.observations_proprioceptive[updatable_slots, self.slot_fill_idx[updatable_slots]] = (
            self.local_proprioceptive_observation_history[updatable_idxs].clone()
        )

        if self._has_exteroceptive_observation:
            self.observations_exteroceptive[updatable_slots, self.slot_fill_idx[updatable_slots]] = (
                obersevations_exteroceptive[updatable_idxs]
                .to(self.device)
                .type(getattr(torch, self.cfg.exteroceptive_obs_precision))
            )

        self.actions[updatable_slots, self.slot_fill_idx[updatable_slots]] = actions[updatable_idxs].to(self.device)

        if self._has_add_exteroceptive_observation:
            self.add_observations_exteroceptive[updatable_slots, self.slot_fill_idx[updatable_slots]] = (
                add_observation_exteroceptive[updatable_idxs]
                .to(self.device)
                .type(getattr(torch, self.cfg.exteroceptive_obs_precision))
            )

        self.slot_fill_idx[updatable_slots] += 1
        full_mask = self.slot_fill_idx[updatable_slots] >= self.cfg.trajectory_length
        full_slots = updatable_slots[full_mask]
        if len(full_slots) > 0:
            full_envs = updatable_idxs[full_mask]
            self.valid_idx[full_slots] = self.cfg.trajectory_length
            self.slot_closed[full_slots] = True
            self.env_slot_idx[full_envs] = -1
            continue_envs = full_envs[~colliding_envs[full_envs] & ~dones[full_envs]]
            self._assign_new_slots(continue_envs)
