# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Planner configuration"""
from fdm import VEL_RANGE_X, VEL_RANGE_Y, VEL_RANGE_YAW
def get_planner_cfg(
    num_envs: int,
    traj_dim: int = 10,
    debug: bool = False,
    device: str = "cuda",
    population_size: int = 1024,
) -> dict:
    cfg_dict = {
        "traj_dim": traj_dim,
        "action_cfg": {
            "_target_": "fdm.planner.ActionCfg",
            "action_dim": 3,
            "traj_dim": traj_dim,

            # 避障任务：允许一定 lateral，但不要过大，否则容易绕圈
            "lower_bound": [VEL_RANGE_X[0], VEL_RANGE_Y[0], VEL_RANGE_YAW[0]],
            "upper_bound": [VEL_RANGE_X[1], VEL_RANGE_Y[1], VEL_RANGE_YAW[1]],
        },
        "to_cfg": {
            "_target_": "fdm.planner.TrajectoryOptimizerCfg",
            "init_debug": debug,
            "debug": debug,

            "dt": 0.25,
            "n_step_fwd": True,
            "control": "fdm",

            # 关键：尽量别把试探性动作直接截成 0
            "set_actions_below_threshold_to_0": True,
            "vel_limit_lin": 0.02,
            "vel_limit_ang": 0.02,

            # -----------------------------
            # 基础项
            # -----------------------------
            "state_cost_w_early_goal_reaching": 0.0,
            "state_cost_w_early_stopping": 0.0,

            "state_cost_w_action_trans_forward": 0.0,
            "state_cost_w_action_trans_side": 0.0,
            "state_cost_w_action_rot": 0.0,

            # -----------------------------
            # 平滑项：保留，但不能强到抑制启动
            # -----------------------------
            "state_cost_w_action_trans_side_biped": 1.0,
            "state_cost_w_heading_running": 0.6,
            "state_cost_w_smooth_vx": 0.02,
            "state_cost_w_smooth_vy": 0.02,
            "state_cost_w_smooth_wz": 0.02,
            "state_cost_w_yaw_rate_change": 0.01,

            # -----------------------------
            # 楼梯项关闭
            # -----------------------------
            "state_cost_w_stair_alignment": 0.0,
            "state_cost_stair_grad_threshold": 0.000008,
            "state_cost_stair_speed_threshold": 0.000005,

            # -----------------------------
            # 暂不使用 cost map
            # -----------------------------
            "state_cost_w_fatal_trav": 40.0,
            "state_cost_w_fatal_unknown": 0.0,
            "state_cost_w_risky_unknown": 0.0,
            "state_cost_w_risky_trav": 0.0,
            "state_cost_w_cautious_unknown": 0.0,
            "state_cost_w_cautious_trav": 0.0,
            "state_cost_w_near_obstacle_soft": 5.0,
            "state_cost_w_near_obstacle_hard": 50.0,
            "state_cost_near_obstacle_soft_th": 0.10,
            "state_cost_near_obstacle_hard_th": 0.20,

            # -----------------------------
            # 起步/推进意愿：比上一版更强
            # -----------------------------
            "state_cost_velocity_tracking": 0.55,
            "state_cost_desired_velocity": 0.35,

            # -----------------------------
            # goal 阈值
            # -----------------------------
            "state_cost_early_goal_distance_offset": 0.3,
            "state_cost_early_goal_heading_offset": 100.0,

            # -----------------------------
            # terminal：显著增强目标吸引，避免只绕不去
            # -----------------------------
            "terminal_cost_w_rot_error": 5.0,
            "terminal_cost_w_position_error": 12.0,
            "terminal_cost_w_heading_to_goal": 2.0,
            "terminal_cost_distance_offset": 0.3,
            "terminal_cost_close_reward": 2.0,
            "terminal_cost_use_threshold": False,

            # -----------------------------
            # collision：仍主导避障，但别强到让 goal 永久失效
            # -----------------------------
            "collision_cost_traj_factor": 12.0,
            "collision_cost_high_risk_factor": 1200.0,
            "collision_cost_safety_factor": 0.0,

            # 仍保留一定邻域扩散，但不要过强，否则容易不断远离柱子
            "num_neighbors": 2,
            "collision_cost_neighbor_spread_weight": 0.6,

            # -----------------------------
            # 风险阈值：稍微放松一点，减少“过度保守绕圈”
            # -----------------------------
            "pp_safe_th": 0.25,
            "pp_risky_th": 0.40,
            "pp_fatal_th": 0.65,
            "pp_risky_value": 0.6,
            "pp_fatal_value": 1.5,

            "states_cost_w_cost_map": False,

            "batch_size": 15000,
            # optional online CVAE dataset dumping
            "cvae_dataset_dump_path": None,
            "cvae_dataset_topk": 4,
            "cvae_dataset_max_samples": 50000,
            "cvae_require_context": True,
            "cvae_collect_all_iterations": False,
            "cvae_collect_iteration_stride": 1,
            "cvae_flush_every_n_samples": 4096,
            "cvae_bucket_ratio_high": 0.4,
            "cvae_bucket_ratio_mid": 0.3,
            "cvae_bucket_ratio_low": 0.3,
            "cvae_labeled_ratio_min": 0.60,
            "cvae_use_threeway_goal_state": True,
        },
        "optim": {
            "_target_": "fdm.planner.BatchedMPPIOptimizer",
            "num_iterations": 8,
            "population_size": population_size,

            # 更偏向挑出兼顾安全和目标推进的少数好轨迹
            "gamma": 1.0,

            # 适中探索，避免围绕柱子附近小范围打转
            "sigma":0.8,

            # 保留一定 warm-start，但别太黏
            "beta": 0.6,

            "lower_bound": ["${action_cfg.lower_bound}" for _ in range(traj_dim)],
            "upper_bound": ["${action_cfg.upper_bound}" for _ in range(traj_dim)],
            "device": device,
            "batch_size": num_envs,
            "sampling_strategy": "gaussian",  # set to "cvae" to sample with a trained CVAE
            "cvae_checkpoint": None,
            "cvae_latent_dim": 16,
            "cvae_temperature": 1.0,
        },
        "robot_cfg": {
            "_target_": "fdm.planner.RobotCfg",
        },
        "to": {
            "_target_": "fdm.planner.SimpleSE2TrajectoryOptimizer",
            "action_cfg": "${action_cfg}",
            "robot_cfg": "${robot_cfg}",
            "to_cfg": "${to_cfg}",
            "optim": "${optim}",
            "device": device,
        },
    }
    return cfg_dict
