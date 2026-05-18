from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from config import Sim2SimConfig
from fdm_adapter import ConstantCommandAdapter, FDMPlannerAdapter, GoalTrackingAdapter, PlannerObservation, ZeroPlannerAdapter
from fdm_model_bridge import DEFAULT_RUN_DIR
from height_scan import FlatHeightScan, RaycastHeightScan
from low_level_controller import LowLevelCommand, MissingGaitController, TorchPolicyPDController, ZeroTorqueController
from mujoco_g1_env import MissingMuJoCoError, MujocoG1Env
from scene_builder import (
    generate_fdm_terrain_obstacles,
    generate_scene_with_obstacles,
    load_obstacle_csv,
    parse_obstacle_box,
)


FDM_DEBUG_COLUMNS = [
    "fdm_replan",
    "fdm_progress_guard",
    "fdm_best_cost",
    "fdm_mppi_best_value",
    "fdm_mppi_mean_value",
    "fdm_pred_x",
    "fdm_pred_y",
    "fdm_pred_yaw",
    "fdm_best_risk_max",
    "fdm_best_risk_mean",
    "fdm_collision_threshold",
    "fdm_best_energy",
    "fdm_front_clearance",
    "fdm_front_vx_limit",
    "fdm_cost_terminal",
    "fdm_cost_position_offset",
    "fdm_cost_rot_error",
    "fdm_cost_heading_to_goal_error",
    "fdm_cost_collision",
    "fdm_cost_collision_mean",
    "fdm_cost_collision_max",
    "fdm_cost_high_risk",
    "fdm_cost_energy",
    "fdm_cost_velocity_tracking",
    "fdm_cost_smooth",
    "fdm_cost_yaw_rate_change",
    "fdm_cost_tracking_prior",
    "fdm_cost_command_progress",
    "fdm_cost_terminal_progress",
    "fdm_cost_lateral_command",
    "fdm_cost_heading_running",
    "fdm_cost_scan_obstacle",
    "fdm_cost_obstacle_speed",
    "fdm_cost_goal_distance",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MuJoCo sim2sim scaffold for G1 + FDM/MPPI.")
    parser.add_argument("--root", type=Path, default=Sim2SimConfig.root)
    parser.add_argument("--g1-xml", type=Path, default=Sim2SimConfig.g1_xml)
    parser.add_argument("--low-level-policy", type=Path, default=Sim2SimConfig.low_level_policy)
    parser.add_argument("--fdm-run", type=str, default=None)
    parser.add_argument("--fdm-run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--fdm-checkpoint", type=Path, default=Sim2SimConfig.fdm_checkpoint)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--control-decimation", type=int, default=Sim2SimConfig.control_decimation)
    parser.add_argument("--physics-dt", type=float, default=Sim2SimConfig.physics_dt)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--zero-controller", action="store_true", help="Use zero ctrl instead of requiring gait policy.")
    parser.add_argument("--zero-planner", action="store_true", help="Use zero high-level command for smoke tests.")
    parser.add_argument("--planner", choices=("zero", "constant", "goal", "fdm"), default="zero")
    parser.add_argument("--test-command", type=float, nargs=3, metavar=("VX", "VY", "WZ"), default=None)
    parser.add_argument("--goal", type=float, nargs=3, metavar=("X", "Y", "YAW"), default=Sim2SimConfig.goal_xy_yaw)
    parser.add_argument("--max-vx", type=float, default=1.0)
    parser.add_argument("--max-vy", type=float, default=0.10)
    parser.add_argument("--max-wz", type=float, default=0.66)
    parser.add_argument("--fdm-population-size", type=int, default=512)
    parser.add_argument("--fdm-mppi-iterations", type=int, default=8)
    parser.add_argument("--fdm-mppi-gamma", type=float, default=1.0)
    parser.add_argument("--fdm-mppi-sigma", type=float, default=0.8)
    parser.add_argument("--fdm-mppi-beta", type=float, default=0.6)
    parser.add_argument("--fdm-action-min-vx", type=float, default=-0.10)
    parser.add_argument("--fdm-action-max-vx", type=float, default=1.50)
    parser.add_argument("--fdm-action-max-vy", type=float, default=0.10)
    parser.add_argument("--fdm-action-max-wz", type=float, default=1.00)
    parser.add_argument("--fdm-gait-min-vx", type=float, default=-0.10)
    parser.add_argument("--fdm-gait-max-vx", type=float, default=1.00)
    parser.add_argument("--fdm-gait-max-vy", type=float, default=0.10)
    parser.add_argument("--fdm-gait-max-wz", type=float, default=0.66)
    parser.add_argument("--fdm-device", type=str, default=None)
    parser.add_argument("--fdm-seed", type=int, default=7)
    parser.add_argument("--fdm-replan-interval", type=int, default=5)
    parser.add_argument("--fdm-tracking-prior-weight", type=float, default=0.0)
    parser.add_argument("--fdm-command-progress-weight", type=float, default=0.0)
    parser.add_argument("--fdm-terminal-progress-weight", type=float, default=0.0)
    parser.add_argument("--fdm-lateral-command-weight", type=float, default=1.0)
    parser.add_argument("--fdm-goal-tolerance", type=float, default=0.08)
    parser.add_argument("--fdm-progress-guard-ratio", type=float, default=0.4)
    parser.add_argument("--fdm-progress-guard-max-risk", type=float, default=0.25)
    parser.add_argument("--fdm-progress-guard-max-scan-cost", type=float, default=3.0)
    parser.add_argument("--fdm-terminal-position-weight", type=float, default=12.0)
    parser.add_argument("--fdm-terminal-rot-weight", type=float, default=5.0)
    parser.add_argument("--fdm-terminal-heading-to-goal-weight", type=float, default=2.0)
    parser.add_argument("--fdm-collision-traj-factor", type=float, default=12.0)
    parser.add_argument("--fdm-collision-high-risk-factor", type=float, default=1200.0)
    parser.add_argument("--fdm-collision-threshold", type=float, default=0.5)
    parser.add_argument("--fdm-collision-safety-factor", type=float, default=0.0)
    parser.add_argument("--fdm-collision-num-neighbors", type=int, default=2)
    parser.add_argument("--fdm-collision-neighbor-spread-weight", type=float, default=0.6)
    parser.add_argument("--fdm-velocity-tracking-weight", type=float, default=0.55)
    parser.add_argument("--fdm-desired-velocity", type=float, default=0.35)
    parser.add_argument("--fdm-action-cost-dt", type=float, default=0.25)
    parser.add_argument("--fdm-heading-running-weight", type=float, default=0.6)
    parser.add_argument("--fdm-smooth-vx-weight", type=float, default=0.02)
    parser.add_argument("--fdm-smooth-vy-weight", type=float, default=0.02)
    parser.add_argument("--fdm-smooth-wz-weight", type=float, default=0.02)
    parser.add_argument("--fdm-yaw-rate-change-weight", type=float, default=0.01)
    parser.add_argument("--fdm-near-obstacle-soft-weight", type=float, default=6.0)
    parser.add_argument("--fdm-near-obstacle-hard-weight", type=float, default=30.0)
    parser.add_argument("--fdm-near-obstacle-soft-distance", type=float, default=0.30)
    parser.add_argument("--fdm-near-obstacle-hard-distance", type=float, default=0.15)
    parser.add_argument("--fdm-scan-obstacle-weight", type=float, default=1.0)
    parser.add_argument("--fdm-scan-obstacle-clearance", type=float, default=0.30)
    parser.add_argument("--fdm-scan-obstacle-height-threshold", type=float, default=0.08)
    parser.add_argument("--fdm-scan-use-footprint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fdm-scan-footprint-front", type=float, default=0.45)
    parser.add_argument("--fdm-scan-footprint-back", type=float, default=0.15)
    parser.add_argument("--fdm-scan-footprint-half-width", type=float, default=0.28)
    parser.add_argument("--fdm-near-obstacle-speed-weight", type=float, default=0.0)
    parser.add_argument("--fdm-near-obstacle-slow-distance", type=float, default=0.90)
    parser.add_argument("--fdm-near-obstacle-stop-distance", type=float, default=0.35)
    parser.add_argument("--fdm-front-obstacle-width", type=float, default=0.55)
    parser.add_argument("--fdm-front-obstacle-lookahead", type=float, default=1.20)
    parser.add_argument("--fdm-front-obstacle-min-vx", type=float, default=0.30)
    parser.add_argument("--height-scan-offset-x", type=float, default=0.0)
    parser.add_argument("--height-scan-offset-y", type=float, default=0.0)
    parser.add_argument("--height-scan-z-start", type=float, default=0.5)
    parser.add_argument("--fdm-stabilize-command", action="store_true")
    parser.add_argument("--fdm-yaw-command-limit", type=float, default=0.45)
    parser.add_argument("--fdm-lateral-command-limit", type=float, default=0.04)
    parser.add_argument("--fdm-yaw-drift-limit", type=float, default=0.55)
    parser.add_argument("--height-scan", choices=("flat", "raycast"), default="flat")
    parser.add_argument(
        "--fdm-terrain-cfg",
        choices=(
            "none",
            "planner_eval",
            "planner_eval_2d",
            "planner_eval_humanoid",
            "paper_figure",
            "sparse_boxes",
            "humanoid_plan_test",
        ),
        default="none",
        help="Generate MuJoCo primitive obstacles from the FDM terrain_cfg preset used by plan_test.",
    )
    parser.add_argument("--fdm-terrain-seed", type=int, default=0)
    parser.add_argument("--obstacle-csv", type=Path, default=None)
    parser.add_argument(
        "--obstacle-box",
        type=float,
        nargs="+",
        action="append",
        default=None,
        metavar="VALUE",
        help="Add a box obstacle. Repeatable. Uses full dimensions, not MuJoCo half-size: X Y LENGTH WIDTH HEIGHT [YAW].",
    )
    parser.add_argument("--generated-scene-path", type=Path, default=None)
    parser.add_argument("--log-csv", type=Path, default=None)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--policy-device", type=str, default="cpu")
    parser.add_argument("--policy-obs-dim", type=int, default=480)
    parser.add_argument("--policy-action-dim", type=int, default=29)
    parser.add_argument("--policy-history", type=int, default=5)
    parser.add_argument(
        "--policy-inference-decimation",
        type=int,
        default=4,
        help="Low-level policy inference decimation in MuJoCo physics steps. Lab NavigationSE2Action uses 4 at 200 Hz, i.e. 50 Hz.",
    )
    parser.add_argument("--allow-partial-policy-joints", action="store_true")
    parser.add_argument("--policy-action-clip", type=float, default=10.0)
    parser.add_argument(
        "--g1-obs-layout",
        choices=("auto", "g1_nav_96", "g1_policy_99"),
        default="g1_nav_96",
        help="G1 low-level policy observation layout. Lab NavigationSE2Action uses g1_nav_96: 96 x history 5 = 480.",
    )
    parser.add_argument(
        "--obs-axis-transform",
        choices=("identity", "rot_x_pos_90", "rot_x_neg_90", "swap_yz", "swap_yz_neg"),
        default="identity",
        help="Optional body-frame vector transform for base velocity, angular velocity, and projected gravity diagnostics.",
    )
    return parser.parse_args()


def missing_mesh_files(xml_path: Path) -> list[Path]:
    if not xml_path.exists():
        return []

    visited: set[Path] = set()
    missing: list[Path] = []

    def visit(path: Path) -> None:
        path = path.resolve()
        if path in visited or not path.exists():
            return
        visited.add(path)

        root = ET.parse(path).getroot()
        for include in root.findall("include"):
            include_file = include.attrib.get("file")
            if include_file:
                visit(path.parent / include_file)

        collect_missing_meshes(path, root)

    def collect_missing_meshes(path: Path, root: ET.Element) -> None:
        compiler = root.find("compiler")
        meshdir = Path(compiler.attrib.get("meshdir", "")) if compiler is not None else Path()
        for mesh in root.findall(".//mesh"):
            mesh_file = mesh.attrib.get("file")
            if not mesh_file:
                continue
            mesh_path = Path(mesh_file)
            if not mesh_path.is_absolute():
                mesh_path = path.parent / meshdir / mesh_path
            if not mesh_path.exists():
                missing.append(mesh_path)

    visit(xml_path)
    return missing


def report_missing(args: argparse.Namespace) -> list[str]:
    missing: list[str] = []
    try:
        import mujoco  # noqa: F401
    except ImportError:
        missing.append("Python package `mujoco` in the active IsaacLab/conda environment")
    if not args.g1_xml.exists():
        missing.append(f"G1 MuJoCo XML: {args.g1_xml}")
    else:
        mesh_missing = missing_mesh_files(args.g1_xml)
        if mesh_missing:
            preview = ", ".join(str(path) for path in mesh_missing[:5])
            suffix = "" if len(mesh_missing) <= 5 else f", ... ({len(mesh_missing)} total)"
            missing.append(f"G1 mesh STL files referenced by XML: {preview}{suffix}")
    if args.low_level_policy is not None and not args.low_level_policy.exists():
        missing.append(f"G1 low-level policy file: {args.low_level_policy}")
    if args.fdm_run is None and args.fdm_checkpoint is None:
        missing.append("FDM run/checkpoint for MPPI model predictions")
    elif args.fdm_checkpoint is not None and not args.fdm_checkpoint.exists():
        missing.append(f"FDM checkpoint: {args.fdm_checkpoint}")
    return missing


def main() -> None:
    args = parse_args()

    missing = report_missing(args)
    if missing:
        print("[SIM2SIM] Missing inputs:")
        for item in missing:
            print(f"  - {item}")
    else:
        print("[SIM2SIM] All configured input paths exist.")

    if args.check_only:
        return
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "logs").mkdir(parents=True, exist_ok=True)
    obstacles = []
    if args.fdm_terrain_cfg != "none":
        obstacles.extend(generate_fdm_terrain_obstacles(args.fdm_terrain_cfg, seed=args.fdm_terrain_seed))
    if args.obstacle_csv is not None:
        obstacles.extend(load_obstacle_csv(args.obstacle_csv))
    if args.obstacle_box:
        obstacles.extend(parse_obstacle_box(values, idx) for idx, values in enumerate(args.obstacle_box))
    if obstacles:
        generated_scene = args.generated_scene_path or (
            args.g1_xml.parent / f"{args.g1_xml.stem}_obstacles.xml"
        )
        args.g1_xml = generate_scene_with_obstacles(args.g1_xml, obstacles, generated_scene)
        print(f"[SIM2SIM] Generated obstacle scene: {args.g1_xml}")
        print(f"[SIM2SIM] Obstacles: {len(obstacles)}")
        if args.fdm_terrain_cfg != "none":
            print(f"[SIM2SIM] FDM terrain cfg preset: {args.fdm_terrain_cfg} seed={args.fdm_terrain_seed}")
    if not args.g1_xml.exists():
        raise FileNotFoundError(f"Cannot run without G1 MuJoCo XML: {args.g1_xml}")

    if args.zero_controller:
        controller = ZeroTorqueController()
    elif args.low_level_policy is not None and args.low_level_policy.exists():
        controller = TorchPolicyPDController(
            policy_path=args.low_level_policy,
            device=args.policy_device,
            obs_history=args.policy_history,
            obs_dim=args.policy_obs_dim,
            action_dim=args.policy_action_dim,
            action_clip=args.policy_action_clip,
            obs_layout=args.g1_obs_layout,
            inference_decimation=args.policy_inference_decimation,
            obs_axis_transform=args.obs_axis_transform,
            strict_joints=not args.allow_partial_policy_joints,
        )
    else:
        controller = MissingGaitController()
    if args.test_command is not None:
        planner = ConstantCommandAdapter(*args.test_command)
    elif args.planner == "constant":
        planner = ConstantCommandAdapter(0.0, 0.0, 0.0)
    elif args.planner == "goal":
        planner = GoalTrackingAdapter(max_vx=args.max_vx, max_vy=args.max_vy, max_wz=args.max_wz)
    elif args.planner == "fdm":
        planner = FDMPlannerAdapter(
            checkpoint=args.fdm_checkpoint,
            run_dir=args.fdm_run_dir,
            device=args.fdm_device or args.policy_device,
            population_size=args.fdm_population_size,
            mppi_iterations=args.fdm_mppi_iterations,
            mppi_gamma=args.fdm_mppi_gamma,
            mppi_sigma=args.fdm_mppi_sigma,
            mppi_beta=args.fdm_mppi_beta,
            seed=args.fdm_seed,
            replan_interval=args.fdm_replan_interval,
            max_vx=args.max_vx,
            max_vy=args.max_vy,
            max_wz=args.max_wz,
            action_min_vx=args.fdm_action_min_vx,
            action_max_vx=args.fdm_action_max_vx,
            action_max_vy=args.fdm_action_max_vy,
            action_max_wz=args.fdm_action_max_wz,
            gait_min_vx=args.fdm_gait_min_vx,
            gait_max_vx=args.fdm_gait_max_vx,
            gait_max_vy=args.fdm_gait_max_vy,
            gait_max_wz=args.fdm_gait_max_wz,
            tracking_prior_weight=args.fdm_tracking_prior_weight,
            command_progress_weight=args.fdm_command_progress_weight,
            terminal_progress_weight=args.fdm_terminal_progress_weight,
            lateral_command_weight=args.fdm_lateral_command_weight,
            goal_tolerance=args.fdm_goal_tolerance,
            progress_guard_ratio=args.fdm_progress_guard_ratio,
            progress_guard_max_risk=args.fdm_progress_guard_max_risk,
            progress_guard_max_scan_cost=args.fdm_progress_guard_max_scan_cost,
            terminal_position_weight=args.fdm_terminal_position_weight,
            terminal_rot_weight=args.fdm_terminal_rot_weight,
            terminal_heading_to_goal_weight=args.fdm_terminal_heading_to_goal_weight,
            collision_traj_factor=args.fdm_collision_traj_factor,
            collision_high_risk_factor=args.fdm_collision_high_risk_factor,
            collision_threshold=args.fdm_collision_threshold,
            collision_safety_factor=args.fdm_collision_safety_factor,
            collision_num_neighbors=args.fdm_collision_num_neighbors,
            collision_neighbor_spread_weight=args.fdm_collision_neighbor_spread_weight,
            velocity_tracking_weight=args.fdm_velocity_tracking_weight,
            desired_velocity=args.fdm_desired_velocity,
            action_cost_dt=args.fdm_action_cost_dt,
            heading_running_weight=args.fdm_heading_running_weight,
            smooth_vx_weight=args.fdm_smooth_vx_weight,
            smooth_vy_weight=args.fdm_smooth_vy_weight,
            smooth_wz_weight=args.fdm_smooth_wz_weight,
            yaw_rate_change_weight=args.fdm_yaw_rate_change_weight,
            near_obstacle_soft_weight=args.fdm_near_obstacle_soft_weight,
            near_obstacle_hard_weight=args.fdm_near_obstacle_hard_weight,
            near_obstacle_soft_distance=args.fdm_near_obstacle_soft_distance,
            near_obstacle_hard_distance=args.fdm_near_obstacle_hard_distance,
            scan_obstacle_weight=args.fdm_scan_obstacle_weight,
            scan_obstacle_clearance=args.fdm_scan_obstacle_clearance,
            scan_obstacle_height_threshold=args.fdm_scan_obstacle_height_threshold,
            scan_use_footprint=args.fdm_scan_use_footprint,
            scan_footprint_front=args.fdm_scan_footprint_front,
            scan_footprint_back=args.fdm_scan_footprint_back,
            scan_footprint_half_width=args.fdm_scan_footprint_half_width,
            near_obstacle_speed_weight=args.fdm_near_obstacle_speed_weight,
            near_obstacle_slow_distance=args.fdm_near_obstacle_slow_distance,
            near_obstacle_stop_distance=args.fdm_near_obstacle_stop_distance,
            front_obstacle_width=args.fdm_front_obstacle_width,
            front_obstacle_lookahead=args.fdm_front_obstacle_lookahead,
            front_obstacle_min_vx=args.fdm_front_obstacle_min_vx,
            height_scan_offset_x=args.height_scan_offset_x,
            height_scan_offset_y=args.height_scan_offset_y,
            stabilize_command=args.fdm_stabilize_command,
            yaw_command_limit=args.fdm_yaw_command_limit,
            lateral_command_limit=args.fdm_lateral_command_limit,
            yaw_drift_limit=args.fdm_yaw_drift_limit,
        )
    else:
        planner = ZeroPlannerAdapter()
    if args.zero_planner:
        planner = ZeroPlannerAdapter()

    if args.height_scan == "raycast":
        height_scan = RaycastHeightScan(
            shape=Sim2SimConfig.height_scan_shape,
            resolution=Sim2SimConfig.height_scan_resolution,
            x_offset=args.height_scan_offset_x,
            y_offset=args.height_scan_offset_y,
            z_start=args.height_scan_z_start,
        )
    else:
        height_scan = FlatHeightScan(
            shape=Sim2SimConfig.height_scan_shape,
            resolution=Sim2SimConfig.height_scan_resolution,
        )

    try:
        env = MujocoG1Env(
            xml_path=args.g1_xml,
            controller=controller,
            height_scan=height_scan,
            physics_dt=args.physics_dt,
        )
    except MissingMuJoCoError as exc:
        print(f"[SIM2SIM] {exc}")
        return

    env.reset()
    planner.reset()
    print(f"[SIM2SIM] Loaded {args.g1_xml}")
    print(f"[SIM2SIM] nq={env.model.nq} nv={env.model.nv} nu={env.model.nu} dt={env.physics_dt}")
    print(f"[SIM2SIM] height_scan={args.height_scan} shape={height_scan.shape} resolution={height_scan.resolution}")

    goal = tuple(args.goal)
    last_command = LowLevelCommand.zeros()
    debug_columns = FDM_DEBUG_COLUMNS if args.planner == "fdm" and not args.zero_planner else []
    csv_file = None
    csv_writer = None
    if args.log_csv is not None:
        args.log_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_file = args.log_csv.open("w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "step",
                "x",
                "y",
                "z",
                "roll",
                "pitch",
                "yaw",
                "vx_cmd",
                "vy_cmd",
                "wz_cmd",
                "ctrl_abs_max",
                "height_min",
                "height_max",
                "height_mean",
                "height_std",
                "height_obstacle_pixels",
                *debug_columns,
            ]
        )
    elif args.root is not None:
        log_dir = args.root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_log = log_dir / f"sim2sim_{args.planner}_{stamp}.csv"
        csv_file = default_log.open("w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "step",
                "x",
                "y",
                "z",
                "roll",
                "pitch",
                "yaw",
                "vx_cmd",
                "vy_cmd",
                "wz_cmd",
                "ctrl_abs_max",
                "height_min",
                "height_max",
                "height_mean",
                "height_std",
                "height_obstacle_pixels",
                *debug_columns,
            ]
        )
        print(f"[SIM2SIM] Logging CSV to {default_log}")

    for step_idx in range(args.steps):
        height_scan_obs = env.observe_height_scan()
        obs = PlannerObservation(
            start_xy_yaw=env.base_xy_yaw(),
            goal_xy_yaw=np.asarray(goal, dtype=np.float32),
            height_scan=height_scan_obs,
            fdm_state=env.observe_fdm_state(),
            fdm_proprioception=env.observe_fdm_proprioception(last_command),
        )
        command: LowLevelCommand = planner.command(obs)
        env.step(command, decimation=args.control_decimation)
        last_command = command
        pose = env.base_xyz_rpy()
        ctrl_abs_max = float(np.max(np.abs(env.data.ctrl))) if env.model.nu else 0.0
        height_min = float(np.min(height_scan_obs))
        height_max = float(np.max(height_scan_obs))
        height_mean = float(np.mean(height_scan_obs))
        height_std = float(np.std(height_scan_obs))
        height_obstacle_pixels = int(np.count_nonzero(height_scan_obs > args.fdm_scan_obstacle_height_threshold))
        debug_info = planner.debug_info()
        if csv_writer is not None:
            csv_writer.writerow(
                [
                    step_idx,
                    *[float(value) for value in pose],
                    command.vx,
                    command.vy,
                    command.wz,
                    ctrl_abs_max,
                    height_min,
                    height_max,
                    height_mean,
                    height_std,
                    height_obstacle_pixels,
                    *[debug_info.get(column, np.nan) for column in debug_columns],
                ]
            )
            if csv_file is not None:
                csv_file.flush()
        if args.print_every > 0 and step_idx % args.print_every == 0:
            print(
                f"[SIM2SIM] step={step_idx} xyz_rpy={pose.round(3).tolist()} "
                f"cmd={[round(command.vx, 3), round(command.vy, 3), round(command.wz, 3)]} "
                f"ctrl_abs_max={ctrl_abs_max:.3f} "
                f"height=[{height_min:.3f},{height_max:.3f},{height_mean:.3f},{height_std:.3f};obs={height_obstacle_pixels}]"
            )

    if csv_file is not None:
        csv_file.close()

    print("[SIM2SIM] Smoke test completed.")


if __name__ == "__main__":
    main()
