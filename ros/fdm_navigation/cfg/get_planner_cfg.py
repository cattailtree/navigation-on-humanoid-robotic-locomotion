
from typing import Tuple


def get_planner_cfg(
    num_envs: int,
    traj_dim: int = 10,
    debug: bool = False,
    device: str = "cuda",
    population_size: int = 1024,
    optim: str = "MPPI",
    vel_range_x: Tuple[float, float] = (-0.2, 0.8),
    vel_range_y: Tuple[float, float] = (-0.1, 0.1),
    vel_range_yaw: Tuple[float, float] = (-1.2, 1.2),
):
    """
    Build planner config dict for Hydra instantiate.

    Args:
        num_envs: batch size / number of environments
        traj_dim: planning horizon
        debug: enable planner debug
        device: cpu / cuda
        population_size: optimizer population size
        optim: "MPPI" or "iCEM"
        vel_range_x: forward velocity range
        vel_range_y: lateral velocity range
        vel_range_yaw: yaw velocity range
    """

    if optim == "MPPI":
        optim_cfg = {
            "_target_": "fdm_navigation.trajectory_optimizer.BatchedMPPIOptimizer",
            "num_iterations": 3,
            "population_size": population_size,
            "gamma": 2.0,
            "sigma": 0.25,
            "beta": 0.6,
            "lower_bound": ["${action_cfg.lower_bound}" for _ in range(traj_dim)],
            "upper_bound": ["${action_cfg.upper_bound}" for _ in range(traj_dim)],
            "provide_zero_action": True,
            "device": device,
        }

    elif optim == "iCEM":
        optim_cfg = {
            "_target_": "fdm_navigation.trajectory_optimizer.BatchedICEMOptimizer",
            "num_iterations": 5,
            "elite_ratio": 0.03,
            "alpha": 0.1,
            "population_size": population_size,
            "return_mean_elites": False,
            "clipped_normal": False,
            "population_size_module": None,
            "population_decay_factor": 1.0,
            "colored_noise_exponent": 1.5,
            "initial_var_factor": 3.0,
            "lower_bound": ["${action_cfg.lower_bound}" for _ in range(traj_dim)],
            "upper_bound": ["${action_cfg.upper_bound}" for _ in range(traj_dim)],
            "keep_elite_frac": 1.0,
            "elite_shifting_n_step": 1,
            "provide_zero_action": True,
            "device": device,
        }
    else:
        raise ValueError(f"Unknown optimizer: {optim}")

    cfg_dict = {
        "action_cfg": {
            "_target_": "fdm_navigation.cfg.ActionCfg",
            "action_dim": 3,
            "traj_dim": traj_dim,
            "lower_bound": [vel_range_x[0], vel_range_y[0], vel_range_yaw[0]],
            "upper_bound": [vel_range_x[1], vel_range_y[1], vel_range_yaw[1]],
        },
        "robot_cfg": {
            "_target_": "fdm_navigation.cfg.RobotCfg",
        },
        "to_cfg": {
            "_target_": "fdm_navigation.cfg.TrajectoryOptimizerCfg",
            "control": "fdm",
            "init_debug": debug,
            "debug": debug,

            # rollout settings
            "dt": 0.25,
            "n_step_fwd": True,
            "batch_size": max(1, num_envs),
            "set_actions_below_threshold_to_0": True,

            # IMPORTANT: use the actual field names from your dataclass
            "vel_limit_lin": 0.1,
            "vel_limit_ang": 0.1,

            # running action costs
            "state_cost_w_action_trans_forward": 0.0,
            "state_cost_w_action_trans_side": 0.0,
            "state_cost_w_action_rot": 0.0,

            # biped-specific extra side penalty
            "state_cost_w_action_trans_side_biped": 4.0,

            # early rewards
            "state_cost_w_early_goal_reaching": 0.0,
            "state_cost_w_early_stopping": 0.0,
            "state_cost_early_goal_distance_offset": 0.3,
            "state_cost_early_goal_heading_offset": 100.0,

            # velocity tracking
            "state_cost_velocity_tracking": 0.5,
            "state_cost_desired_velocity": 0.35,

            # keep these OFF initially; they were likely the ones that over-biased heading
            "state_cost_w_heading_running": 0.0,
            "state_cost_w_smooth_vx": 0.5,
            "state_cost_w_smooth_vy": 1.0,
            "state_cost_w_smooth_wz": 1.0,
            "state_cost_w_yaw_rate_change": 0.8,

            # stair-specific term OFF by default until baseline behavior is stable
            "state_cost_w_stair_alignment": 0.0,
            "state_cost_stair_grad_threshold": 0.08,
            "state_cost_stair_speed_threshold": 0.05,

            # terrain / traversability
            "states_cost_w_cost_map": False,
            "states_cost_w_cost_map_height_diff_thres": 0.3,
            "state_cost_w_fatal_unknown": 0.0,
            "state_cost_w_fatal_trav": 0.0,
            "state_cost_w_risky_unknown": 0.0,
            "state_cost_w_risky_trav": 0.0,
            "state_cost_w_cautious_unknown": 0.0,
            "state_cost_w_cautious_trav": 0.0,

            # terminal cost
            "pos_error_3d": False,
            "terminal_cost_w_rot_error": 10.0,
            "terminal_cost_w_position_error": 20.0,
            "terminal_cost_w_heading_to_goal": 0.0,
            "terminal_cost_distance_offset": 0.3,
            "terminal_cost_close_reward": 10.0,
            "terminal_cost_use_threshold": True,

            # collision
            "collision_cost_traj_factor": 0.3,
            "collision_cost_high_risk_factor": 100.0,
            "collision_cost_safety_factor": 0.0,
            "num_neighbors": 2,
            "collision_cost_neighbor_spread_weight": 0.0,

            # pp
            "pp_safe_th": 0.0,
            "pp_risky_th": 0.0,
            "pp_fatal_th": 0.0,
            "pp_risky_value": 0.0,
            "pp_fatal_value": 0.0,
        },
        "optim": optim_cfg,
        "to": {
            "_target_": "fdm_navigation.trajectory_optimizer.SimpleSE2TrajectoryOptimizer",
            "action_cfg": "${action_cfg}",
            "robot_cfg": "${robot_cfg}",
            "to_cfg": "${to_cfg}",
            "optim": "${optim}",
            "device": device,
        },
    }

    return cfg_dict
