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
          - Use support_contact (any foot) instead of "all feet".
          - Warmup: after reset, allow counting/updating even if contact is noisy.
          - Avoid repeated plan resets due to transient collisions in first few steps.
        """

        # ---------- 0) collision flag ----------
        # NOTE: 你这里用的是 obs["fdm_state"][...,7]，保持不变
        colliding_envs = obs["fdm_state"][..., 7].to(torch.bool)

        # ---------- 1) done reset ----------
        # reset env counter when env is reset in simulation
        self.env_step_counter[dones] = 0

        # ---------- 2) compute support contact (two-foot) ----------
        support_contact = self._compute_support_contact(feet_contact)  # (N,)

        # ---------- 3) warmup window ----------
        # 两足：reset 后前几步非常容易出现“单脚接触/接触抖动”，如果硬等双脚落地会卡死
        # warmup_steps: 3~10 都行；先用 6（0.12s@50Hz 或 0.24s@25Hz）
        warmup_steps = getattr(self.cfg, "warmup_steps", 6)
        in_warmup = self.env_step_counter < warmup_steps  # (N,)

        # ---------- 4) decide which envs are updatable ----------
        # 原逻辑：counter%interval==0 且 counter!=0 且 all-feet-contact
        # 改：support_contact 或者 warmup 期间放行（不然 plan_step 卡 0）
        updatable_envs = (self.env_step_counter % self.resample_interval == 0)
        updatable_envs[self.env_step_counter == 0] = False

        # 两足修正：只要有支撑就允许更新；warmup 期间也允许更新
        updatable_envs[~(support_contact | in_warmup)] = False

        # 原来的特殊规则：在 interval 边界避免 collision env 更新
        # 这个保留，但要注意 warmup 期间也别被卡死
        interval_mask = (self.env_step_counter == self.resample_interval)
        if torch.any(interval_mask):
            updatable_envs[interval_mask] = ~(colliding_envs[interval_mask])

        # ---------- 5) advance plan step ----------
        self._plan_step[updatable_envs] += 1

        # ---------- 6) collision-triggered plan reset (but soften for startup) ----------
        # 四足里碰撞就 reset plan 没太大问题；两足起步阶段 contact 噪声更大，容易“反复清零动作”
        # 改：warmup 期间如果碰撞就别立刻 reset plan（最多冻结/或者延后）
        # 你要更狠也可以只对非 warmup 的 env reset
        coll_reset_mask = colliding_envs & (~in_warmup)
        if torch.any(coll_reset_mask):
            self.reset(obs=obs, env_ids=self._ALL_INDICES[coll_reset_mask], return_actions=False)

        # ---------- 7) planner_reset_after ----------
        planner_reset_after = updatable_envs & (self._plan_step == 1)
        if torch.any(planner_reset_after):
            self.plan_reset(obs=obs, env_ids=self._ALL_INDICES[planner_reset_after])

        # ---------- 8) horizon end -> replan ----------
        env_to_replan = self._ALL_INDICES[self._plan_step >= (self.cfg.horizon - 1)]
        if env_to_replan.numel() > 0:
            self.plan(env_ids=env_to_replan, obs=obs, random_init=False)
            self._plan_step[env_to_replan] = 0

        # ---------- 9) step counter ----------
        # 原逻辑：只有 all-feet-contact 才开始计数
        # 改：有支撑就计数；warmup 期间也计数（否则 counter 可能永远 0）
        count_mask = support_contact | in_warmup
        self.env_step_counter[count_mask] += 1

        # ---------- 10) return action ----------
        # 注意：_plan_step 是每个 env 的索引；用 advanced indexing 取每个 env 当前 step 的 action
        return self._plan[self._ALL_INDICES, self._plan_step]

    def reset(self, obs: dict | None = None, env_ids: torch.Tensor | None = None, return_actions: bool = True):
        """
        Reset agent internal buffers and create a new plan for specified env_ids.
        Keeps your original behavior but is safe for two-foot startup.
        """
        if env_ids is None:
            env_ids = self._ALL_INDICES

        # reset buffers
        self._plan_step[env_ids] = 0
        self._plan[env_ids] = 0

        # plan random init
        self.plan(env_ids=env_ids, obs=obs, random_init=True)

        if return_actions:
            return self._plan[self._ALL_INDICES, self._plan_step]
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
