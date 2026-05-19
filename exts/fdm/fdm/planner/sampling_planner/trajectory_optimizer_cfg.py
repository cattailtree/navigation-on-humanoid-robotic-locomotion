from dataclasses import dataclass, field


@dataclass
class ActionCfg:
    # action = (vx, vy, wz)
    action_dim: int
    traj_dim: int

    # Units:
    #   vx: m/s
    #   vy: m/s
    #   wz: rad/s
    #
    # Recommended for humanoid / biped terrain navigation:
    #   keep lateral range small
    lower_bound: list[float]
    upper_bound: list[float]


@dataclass
class CEMCfg:
    num_iterations: int = 3
    elite_ratio: float = 0.03
    population_size: int = 128
    alpha: float = 0.1
    return_mean_elites: bool = False

    # CEM specific
    clipped_normal: bool = False

    # ICEM specific
    population_size_module: int | None = None
    population_decay_factor: float = 1.0
    colored_noise_exponent: float = 1.0
    colored_noise_exponent_inital: float = 1.0
    initial_var_factor: float = 1.0

    keep_elite_frac: float = 1.0
    elite_shifting_n_step: int = 1
    provide_zero_action: bool = True


@dataclass
class MPPICfg:
    num_iterations: int = 4
    population_size: int = 512

    gamma: float = 2.0
    """
    Reward scaling term.

    gamma=0 means almost equal weighting of trajectories.
    Larger gamma focuses more on high-reward trajectories when composing the next action mean.
    """

    sigma: float = 0.45
    """
    Noise scaling term used in action sampling.

    If you enlarge action sampling range, especially yaw, sigma usually also needs to increase.
    """

    beta: float = 0.6
    """
    Temporal time correlation.

    beta=0 means action is strongly repeated across horizon.
    beta=1 means actions are fully uncorrelated in time.
    """
    sampling_strategy: str = "gaussian"
    """Noise source for trajectory sampling. One of: ``gaussian`` or ``cvae``."""
    cvae_checkpoint: str | None = None
    """Optional CVAE checkpoint path used when ``sampling_strategy='cvae'``."""
    cvae_latent_dim: int = 16
    """Latent dimension for the CVAE sampler."""
    cvae_temperature: float = 1.0
    """Sampling temperature for latent variables in the CVAE sampler."""


@dataclass
class TrajectoryOptimizerCfg:
    # ----------------------------------------------------------------------
    # rollout / optimizer mode
    # ----------------------------------------------------------------------
    dt: float = 0.1
    n_step_fwd: bool = True
    control: str = "position_control"
    init_debug: bool = True

    # FDM model needs minibatches due to memory restrictions
    batch_size: int = 5

    # Can be modified via dynamic reconfigure
    replan_every_n: int = 1
    debug: bool = False
    set_actions_below_threshold_to_0: bool = False
    vel_limit_lin: float = 0.1
    vel_limit_ang: float = 0.1
    cvae_dataset_dump_path: str | None = None
    """Optional output path to dump CVAE training tuples from online MPPI planning."""
    cvae_dataset_topk: int = 4
    """Top-k trajectories per environment used as supervised targets in dataset dumping."""
    cvae_dataset_max_samples: int = 50000
    """Maximum number of dumped CVAE samples kept on disk."""
    cvae_require_context: bool = True
    """Skip CVAE dataset dumping when no planner context can be built."""
    cvae_collect_all_iterations: bool = False
    """Collect CVAE tuples at every optimizer iteration, subject to stride."""
    cvae_collect_iteration_stride: int = 1
    """Collect one CVAE sample round every N optimizer iterations."""
    cvae_flush_every_n_samples: int = 4096
    """Flush CVAE dataset to disk after at least this many new samples."""
    cvae_bucket_ratio_high: float = 0.4
    """Fraction of sampled CVAE targets drawn from high-scoring trajectories."""
    cvae_bucket_ratio_mid: float = 0.3
    """Fraction of sampled CVAE targets drawn from middle-scoring trajectories."""
    cvae_bucket_ratio_low: float = 0.3
    """Fraction of sampled CVAE targets drawn from low-scoring trajectories."""
    cvae_labeled_ratio_min: float = 0.60
    """Minimum labeled share to keep when executed labels are present."""
    cvae_use_threeway_goal_state: bool = True
    """Emit goal_state_3way labels in the dumped CVAE dataset."""

    # ----------------------------------------------------------------------
    # optional cost map cost from height scan
    # ----------------------------------------------------------------------
    states_cost_w_cost_map: bool = False
    states_cost_w_cost_map_height_diff_thres: float = 0.3

    # ----------------------------------------------------------------------
    # running action costs
    # ----------------------------------------------------------------------
    state_cost_w_action_rot: float = 1.0
    state_cost_w_action_trans_forward: float = 1.0
    state_cost_w_action_trans_side: float = 1.0

    # NEW: stronger side-motion penalty for humanoid / biped
    # Used by the upgraded optimizer, but harmless for old logic
    state_cost_w_action_trans_side_biped: float = 5.0

    # ----------------------------------------------------------------------
    # terrain / traversability penalties
    # ----------------------------------------------------------------------
    state_cost_w_fatal_trav: float = 6.5
    state_cost_w_fatal_unknown: float = 10.0

    state_cost_w_risky_trav: float = 6.5
    state_cost_w_risky_unknown: float = 10.0

    state_cost_w_cautious_trav: float = 6.5
    state_cost_w_cautious_unknown: float = 10.0

    # ----------------------------------------------------------------------
    # early goal / early stopping
    # ----------------------------------------------------------------------
    state_cost_w_early_goal_reaching: float = 2.0
    state_cost_early_goal_distance_offset: float = 0.3
    state_cost_early_goal_heading_offset: float = 0.3

    state_cost_w_early_stopping: float = 1.0
    state_cost_w_near_obstacle_soft: float = 8.0
    state_cost_w_near_obstacle_hard: float = 30.0
    state_cost_near_obstacle_soft_th: float = 0.10
    state_cost_near_obstacle_hard_th: float = 0.20
    state_cost_obstacle_height_th: float = 0.08
    """Minimum height above local ground to treat a height-scan cell as an obstacle."""
    state_cost_ground_percentile: float = 0.20
    """Lower height-scan percentile used as local ground reference for obstacle extraction."""

    # ----------------------------------------------------------------------
    # velocity tracking
    # ----------------------------------------------------------------------
    state_cost_velocity_tracking: float = 1.0
    state_cost_desired_velocity: float = 0.35
    # For humanoid stair / rough terrain approach, 0.35 ~ 0.55 is often better
    # than 1.0 m/s. You can still tune this later.

    # ----------------------------------------------------------------------
    # NEW: running heading / smoothness / zig-zag suppression
    # ----------------------------------------------------------------------
    # Encourage the whole rollout to face the goal direction, not just the end.
    state_cost_w_heading_running: float = 0.0

    # Penalize action changes across horizon
    state_cost_w_smooth_vx: float = 0.5
    state_cost_w_smooth_vy: float = 1.5
    state_cost_w_smooth_wz: float = 1.2

    # Additional yaw oscillation suppression
    state_cost_w_yaw_rate_change: float = 1.0

    # ----------------------------------------------------------------------
    # NEW: stair / step alignment cost
    # ----------------------------------------------------------------------
    # Start with 0.0, then turn on after you verify heading/smoothness help.
    state_cost_w_stair_alignment: float = 0.0

    # Height-gradient threshold above which we regard the area as "step-like"
    state_cost_stair_grad_threshold: float = 0.08

    # Only apply stair alignment when command speed is above this threshold
    state_cost_stair_speed_threshold: float = 0.05

    # ----------------------------------------------------------------------
    # terminal cost
    # ----------------------------------------------------------------------
    pos_error_3d: bool = False
    """Compute the position error in 3D space instead of 2D space."""

    terminal_cost_w_rot_error: float = 10.0
    terminal_cost_w_position_error: float = 20.0

    # NEW: encourage the final pose to face the approach direction to the goal
    terminal_cost_w_heading_to_goal: float = 5.0

    terminal_cost_close_reward: float = 100.0
    terminal_cost_distance_offset: float = 0.3
    terminal_cost_use_threshold: bool = True

    # ----------------------------------------------------------------------
    # collision cost
    # ----------------------------------------------------------------------
    collision_cost_traj_factor: float = 0.5
    collision_cost_high_risk_factor: float = 100.0
    collision_cost_safety_factor: float = 1.0

    # sum the cost of neighbors to suppress isolated wrong collision predictions
    num_neighbors: int = 2

    # Neighbor spread weight for suppressing isolated wrong collision predictions.
    collision_cost_neighbor_spread_weight: float = 1.0

    # ----------------------------------------------------------------------
    # Preprocessing [pp]
    #
    # Ramp function to define the cost
    #
    #                      -------   fatal_value
    #                      |
    #                      |
    #               .------          risky_value
    #             .
    #           .
    # ---------                       0
    #
    #      safe_th  risky_th fatal_th
    # ----------------------------------------------------------------------
    pp_safe_th: float = 0.1
    pp_risky_th: float = 0.6
    pp_fatal_th: float = 0.8
    pp_risky_value: float = 0.5
    pp_fatal_value: float = 1.0


@dataclass
class RobotCfg:
    # Point resolution used to rasterize rectangles into robot shape points
    resolution: float = 0.04

    # ----------------------------------------------------------------------
    # Humanoid / biped approximate footprint
    #
    # This replaces the ANYmal-style large quadruped footprint.
    # The goal is not exact CAD geometry, but a planner-friendly footprint that
    # better reflects a humanoid body in navigation / rough-terrain planning.
    # ----------------------------------------------------------------------

    # Conservative core body / pelvis footprint
    fatal: list[tuple[tuple[float, float], tuple[float, float]]] = field(
        default_factory=lambda: [
            ((-0.22, -0.12), (0.22, 0.12)),  # torso / pelvis projected footprint
        ]
    )

    # Slightly expanded "risky" zone
    risky: list[tuple[tuple[float, float], tuple[float, float]]] = field(
        default_factory=lambda: [
            ((-0.30, -0.18), (0.30, 0.18)),
        ]
    )

    # More cautious outer shell
    cautious: list[tuple[tuple[float, float], tuple[float, float]]] = field(
        default_factory=lambda: [
            ((-0.36, -0.24), (0.36, 0.24)),
        ]
    )
