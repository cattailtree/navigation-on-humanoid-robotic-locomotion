# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fdm.runner import FDMRunner

    from .base_agent_cfg import AgentCfg


class Agent(ABC):
    def __init__(self, cfg: AgentCfg, runner: FDMRunner):
        self.cfg: AgentCfg = cfg
        self._runner: FDMRunner = runner

        # init buffers
        self._init_buffers()

    """
    Properties
    """

    @property
    def device(self):
        return self._runner.env.device

    @property
    def action_dim(self):
        return self._runner.env.action_manager.action.shape[1]

    @property
    def resample_interval(self):
        return self._runner.cfg.model_cfg.command_timestep / self._runner.env.step_dt

    """
    Operations
    """

    def _compute_support_contact(self, feet_contact: torch.Tensor) -> torch.Tensor:
        """
        Two-foot friendly support definition.

        Accepts:
          - (N,) bool  : already support-contact per env
          - (N,2) bool : left/right contact per env -> any contact
          - (N,K) bool : multi-feet contact -> any contact (works for quad too)

        Returns:
          support_contact: (N,) bool
        """
        if feet_contact.dim() == 1:
            return feet_contact.to(torch.bool)
        else:
            # any foot in contact counts as support
            return feet_contact.to(torch.bool).any(dim=-1)

    def act(self, obs: dict, dones: torch.Tensor, feet_contact: torch.Tensor):
        """
        Two-foot friendly replanning gate:
        - Warmup period after reset: do NOT advance plan, do NOT execute motion plan.
        - Only start consuming plan after stable support is established.
        """

        colliding_envs = obs["fdm_state"][..., 7].to(torch.bool)

        # reset step counter for envs reset by simulator
        self.env_step_counter[dones] = 0
        self._plan_step[dones] = 0

        support_contact = self._compute_support_contact(feet_contact)

        warmup_steps = getattr(self.cfg, "warmup_steps", 0)
        in_warmup = self.env_step_counter < warmup_steps
        collision_guard_steps = getattr(self.cfg, "collision_guard_steps", 2)
        collision_valid = self.env_step_counter >= collision_guard_steps


        # --------------------------------------------------
        # 1) During warmup: freeze plan execution
        # --------------------------------------------------
        # Only count warmup if there is at least some support OR you explicitly want pure time-based warmup
        warmup_count_mask = in_warmup & support_contact
        self.env_step_counter[warmup_count_mask] += 1

        # If still in warmup, return zero action directly
        still_warmup = self.env_step_counter < warmup_steps
        if torch.any(still_warmup):
            # make sure those envs do not consume plan
            self._plan_step[still_warmup] = 0

        # --------------------------------------------------
        # 2) After warmup: normal logic
        # --------------------------------------------------
        active_envs = ~still_warmup

        # collision-triggered reset only after warmup
        
        coll_reset_mask = colliding_envs & active_envs & collision_valid
        if torch.any(coll_reset_mask):
            self.reset(obs=obs, env_ids=self._ALL_INDICES[coll_reset_mask], return_actions=False)
            self.env_step_counter[coll_reset_mask] = 0
            self._plan_step[coll_reset_mask] = 0
        
        bootstrap_replan_step = getattr(self.cfg, "bootstrap_replan_step", 2)
        bootstrap_envs = active_envs & (self.env_step_counter == bootstrap_replan_step)
        if torch.any(bootstrap_envs):
            env_ids_bootstrap = self._ALL_INDICES[bootstrap_envs]
            self.plan(env_ids=env_ids_bootstrap, obs=obs, random_init=False)
            self._plan_step[env_ids_bootstrap] = 0

        updatable_envs = torch.zeros_like(active_envs, dtype=torch.bool)
        updatable_envs[active_envs] = (self.env_step_counter[active_envs] % self.resample_interval == 0)
        #updatable_envs[self.env_step_counter == 0] = False
        #updatable_envs[~support_contact] = False

        interval_mask = active_envs & (self.env_step_counter == self.resample_interval)
        if torch.any(interval_mask):
            updatable_envs[interval_mask] = ~(colliding_envs[interval_mask])

        self._plan_step[updatable_envs] += 1

        planner_reset_after = updatable_envs & (self._plan_step == 1)
        if torch.any(planner_reset_after):
            self.plan_reset(obs=obs, env_ids=self._ALL_INDICES[planner_reset_after])

        env_to_replan = self._ALL_INDICES[active_envs & (self._plan_step >= (self.cfg.horizon - 1))]
        if env_to_replan.numel() > 0:
            self.plan(env_ids=env_to_replan, obs=obs, random_init=False)
            self._plan_step[env_to_replan] = 0

        # count active env steps only when supported
        count_mask = active_envs & support_contact
        self.env_step_counter[count_mask] += 1

        # --------------------------------------------------
        # 3) Compose action
        # --------------------------------------------------
        actions = self._plan[self._ALL_INDICES, self._plan_step].clone()

        # freeze warmup envs with zero action
        actions[still_warmup] = 0.0

        return actions

    def reset(self, obs: dict | None = None, env_ids: torch.Tensor | None = None, return_actions: bool = True):
        if env_ids is None:
            env_ids = self._ALL_INDICES

        # reset buffers
        self._plan_step[env_ids] = 0
        self._plan[env_ids] = 0

        # generate initial plan
        self.plan(env_ids=env_ids, obs=obs, random_init=True)

        # IMPORTANT:
        # freeze first few actions to zero so humanoid can settle after reset
        warmup_plan_steps = getattr(self.cfg, "warmup_plan_steps", 0)
        self._plan[env_ids, :warmup_plan_steps] = 0.0

        if return_actions:
            actions = self._plan[self._ALL_INDICES, self._plan_step].clone()
            return actions
        return None

    @abstractmethod
    def plan(self, obs: dict | None = None, env_ids: torch.Tensor | None = None, random_init: bool = True):
        pass

    def plan_reset(self, obs: dict, env_ids: torch.Tensor):
        """Replan for already reset environments in the simulator.

        Necessary for the sampling-planner agent that depends on the new observations from the reset environment."""
        pass

    def debug_viz(self, env_ids: list[int] | None = None):
        pass

    """
    Helper functions
    """

    def _init_buffers(self):
        self._ALL_INDICES = torch.arange(self._runner.env.num_envs, device=self.device, dtype=torch.long)
        # plan buffers
        self._plan_step = torch.zeros(self._runner.env.num_envs, device=self.device, dtype=torch.long)
        self._plan = torch.zeros((self._runner.env.num_envs, self.cfg.horizon, self.action_dim), device=self.device)
        # env step counter
        self.env_step_counter = torch.zeros(self._runner.env.num_envs, device=self.device, dtype=torch.long)
