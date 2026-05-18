# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

import omni.log
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz, sample_uniform

from nav_suite.terrain_analysis import TerrainAnalysisCfg, TerrainAnalysisSingletonCfg

from .commands import FixGoalCommand, GoalCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


###
# Terrain Analysis based Reset
###
class TerrainAnalysisRootReset:
    def __init__(self, cfg: TerrainAnalysisCfg | TerrainAnalysisSingletonCfg, robot_dim: float = 0.5):
        self.cfg = cfg
        self.robot_idx_dim = math.ceil(robot_dim / self.cfg.grid_resolution)

        # G1-friendly spawn tuning
        self._spawn_z_offset = 0.65
        self._safety_margin_min = 0.05

    def _run_analysis(self, env: ManagerBasedRLEnv):
        if hasattr(self.cfg.class_type, "instance") and self.cfg.class_type.instance() is not None:
            self.analyser = self.cfg.class_type.instance()
        else:
            self.analyser = self.cfg.class_type(self.cfg, env.scene)
        omni.log.info("Running terrain analysis")
        self.analyser.analyse()
        omni.log.info("Terrain analysis completed")

    def _get_spawn_height(self, positions: torch.Tensor) -> torch.Tensor:
        pos_idx = (
            (
                positions[..., :2]
                - torch.tensor(
                    [self.analyser.mesh_dimensions[2], self.analyser.mesh_dimensions[3]],
                    device=positions.device,
                )
            )
            / self.cfg.grid_resolution
        ).int()

        pos_idx[:, 0] = torch.clamp(
            pos_idx[:, 0],
            self.robot_idx_dim,
            self.analyser.height_grid.shape[0] - 1 - self.robot_idx_dim,
        )
        pos_idx[:, 1] = torch.clamp(
            pos_idx[:, 1],
            self.robot_idx_dim,
            self.analyser.height_grid.shape[1] - 1 - self.robot_idx_dim,
        )

        local_height_map_max = [
            torch.max(
                self.analyser.height_grid[
                    curr_pos_idx[0] - self.robot_idx_dim : curr_pos_idx[0] + self.robot_idx_dim,
                    curr_pos_idx[1] - self.robot_idx_dim : curr_pos_idx[1] + self.robot_idx_dim,
                ]
            )
            for curr_pos_idx in pos_idx
        ]
        return torch.stack(local_height_map_max).to(positions.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        yaw_range: tuple[float, float],
        velocity_range: dict[str, tuple[float, float]],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ):
        asset: RigidObject | Articulation = env.scene[asset_cfg.name]
        root_states = asset.data.default_root_state[env_ids].clone()

        if not hasattr(self, "analyser"):
            self._run_analysis(env)

        positions = self.analyser.points[torch.randperm(self.analyser.points.shape[0])[: len(env_ids)]].clone()

        # IMPORTANT:
        # If analyser.points is local to each env / terrain tile, uncomment the next line.
        # positions[:, :3] += env.scene.env_origins[env_ids]

        local_h = self._get_spawn_height(positions)
        margin = max(self.analyser.cfg.grid_resolution * 2, self._safety_margin_min)
        positions[:, 2] = local_h + self._spawn_z_offset + margin

        # Conservative yaw for humanoid reset
        safe_yaw_min = max(yaw_range[0], -0.1)
        safe_yaw_max = min(yaw_range[1], 0.1)
        yaw_samples = sample_uniform(safe_yaw_min, safe_yaw_max, (len(env_ids), 1), device=asset.device)

        orientations = quat_from_euler_xyz(
            torch.zeros_like(yaw_samples), torch.zeros_like(yaw_samples), yaw_samples
        ).squeeze(1)

        # IMPORTANT: zero root velocity at reset for humanoid stability
        velocities = torch.zeros((len(env_ids), 6), device=asset.device)

        asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)

        default_joint_pos = asset.data.default_joint_pos[env_ids].clone()
        default_joint_vel = torch.zeros_like(asset.data.default_joint_vel[env_ids])
        asset.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)

        try:
            omni.log.info(
                f"TerrainAnalysisRootReset: local_h[:5]={local_h[:5].cpu().numpy()} "
                f"spawn_z[:5]={positions[:5,2].cpu().numpy()} "
                f"margin={float(margin)} spawn_offset={self._spawn_z_offset}"
            )
            actual_root_pos = asset.data.root_pos_w[env_ids, 2].detach().cpu()
            omni.log.info(f"TerrainAnalysisRootReset: wrote root_pos_w z[:5]={actual_root_pos[:5].numpy()}")
        except Exception:
            pass

    def __name__(self):
        return "TerrainAnalysisRootReset"