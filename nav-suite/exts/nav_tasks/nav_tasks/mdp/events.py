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
    """
    Reset root pose using terrain-analysis points, but spawn at a controlled height above terrain.

    NOTE:
    - Signature kept identical to avoid any call-site breakage.
    - This version is intended for your G1 use-case (no need to be compatible with ANYmal behavior).
    """

    def __init__(self, cfg: TerrainAnalysisCfg | TerrainAnalysisSingletonCfg, robot_dim: float = 0.5):
        self.cfg = cfg
        # robot footprint radius in grid cells
        self.robot_idx_dim = math.ceil(robot_dim / self.cfg.grid_resolution)

        # -------- G1-specific spawn tuning (NO new args) --------
        # Height (meters) added above local terrain max height.
        # For humanoid, DO NOT reuse asset.default_root_state[z] (often ~0.8) + 0.3 margin -> spawns too high.
        self._spawn_z_offset = 0.50   # ✅ 先用 0.60；若仍高/低，调到 0.55~0.75
        # Extra safety margin to prevent initial penetration. 0.3m is too large for humanoid here.
        self._safety_margin_min = 0.05

    def _run_analysis(self, env: ManagerBasedRLEnv):
        """Run the terrain analysis to compute the root state reset."""
        if hasattr(self.cfg.class_type, "instance") and self.cfg.class_type.instance() is not None:
            self.analyser = self.cfg.class_type.instance()
        else:
            self.analyser = self.cfg.class_type(self.cfg, env.scene)
        omni.log.info("Running terrain analysis")
        self.analyser.analyse()
        omni.log.info("Terrain analysis completed")

    def _get_spawn_height(self, positions: torch.Tensor) -> torch.Tensor:
        """Get a conservative spawn height based on local max height in the height grid."""
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
        """Sample new start positions and random initial velocities."""
        asset: RigidObject | Articulation = env.scene[asset_cfg.name]

        # default root state (for velocities baseline)
        root_states = asset.data.default_root_state[env_ids].clone()

        # ensure analysis is ready
        if not hasattr(self, "analyser"):
            self._run_analysis(env)

        # pick spawn XY candidates
        positions = self.analyser.points[torch.randperm(self.analyser.points.shape[0])[: len(env_ids)]].clone()

        # --- FIX: spawn Z computation (humanoid-safe) ---
        # local terrain max height (avoid spawning inside terrain)
        local_h = self._get_spawn_height(positions)

        # small safety margin: grid-based, but clamp to a small min (NOT 0.3)
        margin = max(self.analyser.cfg.grid_resolution * 2, self._safety_margin_min)

        # final spawn z = local terrain max height + humanoid offset + small margin
        positions[:, 2] = local_h + self._spawn_z_offset + margin

        # DEBUG: print computed spawn heights for inspection
        try:
            omni.log.info(f"TerrainAnalysisRootReset: local_h[:5]={local_h[:5].cpu().numpy()} spawn_z[:5]={positions[:5,2].cpu().numpy()} margin={float(margin)} spawn_offset={self._spawn_z_offset}")
            omni.log.info(f"TerrainAnalysisRootReset: default_root_z[:5]={root_states[:5,2].cpu().numpy()}")
            # also print env origin(s) for the selected env ids to check framing
            try:
                env_origins = env.scene.env_origins[env_ids].cpu().numpy()
                omni.log.info(f"TerrainAnalysisRootReset: env_origins[:5]={env_origins[:5]}")
                omni.log.info(f"TerrainAnalysisRootReset: positions_world_minus_env_origins[:5]={(positions[:5,:3].cpu().numpy() - env_origins[:5]).tolist()}")
            except Exception:
                omni.log.info("TerrainAnalysisRootReset: failed to read env.scene.env_origins for debug")
        except Exception:
            pass

        # yaw
        yaw_samples = sample_uniform(yaw_range[0], yaw_range[1], (len(env_ids), 1), device=asset.device)
        orientations = quat_from_euler_xyz(
            torch.zeros_like(yaw_samples), torch.zeros_like(yaw_samples), yaw_samples
        ).squeeze(1)

        # velocities
        range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
        velocities = root_states[:, 7:13] + rand_samples

        # write to sim
        asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)

        # DEBUG: read back and report actual root z written into the simulation
        try:
            # reading world root positions after write
            actual_root_pos = asset.data.root_pos_w[env_ids, 2].detach().cpu()
            omni.log.info(f"TerrainAnalysisRootReset: wrote root_pos_w z[:5]={actual_root_pos[:5].numpy()}")
        except Exception:
            pass

        # set joints to default
        default_joint_pos = asset.data.default_joint_pos[env_ids].clone()
        default_joint_vel = asset.data.default_joint_vel[env_ids].clone()
        asset.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)

    def __name__(self):
        return "TerrainAnalysisRootReset"
