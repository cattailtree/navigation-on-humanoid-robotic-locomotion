from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from config import Sim2SimConfig
from fdm_adapter import FDMPlannerAdapter, GoalTrackingAdapter, PlannerObservation, ZeroPlannerAdapter
from fdm_model_bridge import DEFAULT_RUN_DIR
from height_scan import FlatHeightScan, RaycastHeightScan
from low_level_controller import LowLevelCommand, MissingGaitController, TorchPolicyPDController, ZeroTorqueController
from mujoco_g1_env import MissingMuJoCoError, MujocoG1Env
from scene_builder import generate_fdm_terrain_obstacles, generate_scene_with_obstacles


@dataclass
class EpisodeResult:
    episode: int
    seed: int
    status: str
    success: bool
    steps: int
    sim_time: float
    wall_time: float
    final_x: float
    final_y: float
    final_yaw: float
    goal_distance: float
    goal_x_error: float
    yaw_error: float
    reference_path_length: float
    path_length: float
    spl: float
    reference_time: float
    spt: float
    min_height: float
    max_roll: float
    max_pitch: float
    max_ctrl_abs: float
    max_obstacle_pixels: int
    max_height_fdm_hit_count: int
    max_height_fdm_geom_count: int
    height_fdm_x_min: float
    height_fdm_x_max: float
    height_fdm_y_min: float
    height_fdm_y_max: float
    height_top_geoms_at_max_fdm_hits: str
    max_fdm_risk: float
    max_fdm_scan_cost: float
    illegal_contact_steps: int
    max_illegal_contact_streak: int
    last_illegal_contact_pair: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch MuJoCo sim2sim evaluation for G1 + FDM/MPPI.")
    parser.add_argument("--root", type=Path, default=Sim2SimConfig.root)
    parser.add_argument("--g1-xml", type=Path, default=Sim2SimConfig.g1_xml)
    parser.add_argument("--low-level-policy", type=Path, default=Sim2SimConfig.low_level_policy)
    parser.add_argument("--fdm-run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--fdm-checkpoint", type=Path, default=Sim2SimConfig.fdm_checkpoint)
    parser.add_argument("--episodes", type=int, default=240)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--max-failures", type=int, default=0, help="Stop early after this many non-success episodes. 0 disables.")
    parser.add_argument("--control-decimation", type=int, default=Sim2SimConfig.control_decimation)
    parser.add_argument("--physics-dt", type=float, default=Sim2SimConfig.physics_dt)
    parser.add_argument("--planner", choices=("fdm", "goal", "zero"), default="fdm")
    parser.add_argument(
        "--eval-preset",
        choices=("full", "fast"),
        default="full",
        help="fast lowers MPPI samples/iterations for coarse success-rate sweeps.",
    )
    parser.add_argument(
        "--fdm-terrain-cfg",
        choices=(
            "planner_eval",
            "planner_eval_2d",
            "planner_eval_calib",
            "planner_eval_humanoid",
            "paper_figure",
            "sparse_boxes",
            "humanoid_plan_test",
        ),
        default="planner_eval",
    )
    parser.add_argument("--height-scan", choices=("flat", "raycast"), default="raycast")
    parser.add_argument("--height-scan-z-start", type=float, default=2.0)
    parser.add_argument("--height-scan-offset-x", type=float, default=0.0)
    parser.add_argument("--height-scan-offset-y", type=float, default=0.0)
    parser.add_argument("--goal", type=float, nargs=3, metavar=("X", "Y", "YAW"), default=Sim2SimConfig.goal_xy_yaw)
    parser.add_argument("--success-distance", type=float, default=0.7)
    parser.add_argument("--success-x-distance", type=float, default=0.4)
    parser.add_argument("--success-yaw", type=float, default=math.pi)
    parser.add_argument(
        "--spt-reference-speed",
        type=float,
        default=1.0,
        help="Reference speed used for SPT = success * shortest_time / max(shortest_time, episode_time).",
    )
    parser.add_argument("--fall-height", type=float, default=0.35)
    parser.add_argument("--fall-roll-pitch", type=float, default=1.0)
    parser.add_argument("--ignore-orientation-fall", action="store_true")
    parser.add_argument(
        "--illegal-contact-grace-steps",
        type=int,
        default=50,
        help=(
            "Require this many consecutive outer control steps of non-foot terrain/obstacle contact "
            "before marking collision. 50 steps is about 1s at the default 50Hz control loop."
        ),
    )
    parser.add_argument("--print-every-episode", type=int, default=10)
    parser.add_argument(
        "--exclude-timeout-from-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude timeout episodes from success-rate/SPL/SPT denominators while still logging them.",
    )
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--generated-scene-dir", type=Path, default=None)
    parser.add_argument("--keep-scenes", action="store_true")
    parser.add_argument("--zero-controller", action="store_true")
    parser.add_argument("--policy-device", type=str, default="cpu")
    parser.add_argument("--policy-obs-dim", type=int, default=480)
    parser.add_argument("--policy-action-dim", type=int, default=29)
    parser.add_argument("--policy-history", type=int, default=5)
    parser.add_argument("--policy-inference-decimation", type=int, default=4)
    parser.add_argument("--allow-partial-policy-joints", action="store_true")
    parser.add_argument("--policy-action-clip", type=float, default=10.0)
    parser.add_argument(
        "--g1-obs-layout",
        choices=("auto", "g1_nav_96", "g1_policy_99"),
        default="g1_nav_96",
    )
    parser.add_argument(
        "--obs-axis-transform",
        choices=("identity", "rot_x_pos_90", "rot_x_neg_90", "swap_yz", "swap_yz_neg"),
        default="identity",
    )

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
    parser.add_argument("--fdm-scan-obstacle-relative-to-floor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fdm-scan-floor-percentile", type=float, default=5.0)
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
    parser.add_argument("--fdm-stabilize-command", action="store_true")
    parser.add_argument("--fdm-yaw-command-limit", type=float, default=0.45)
    parser.add_argument("--fdm-lateral-command-limit", type=float, default=0.04)
    parser.add_argument("--fdm-yaw-drift-limit", type=float, default=0.55)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _apply_eval_preset(args)
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "logs").mkdir(parents=True, exist_ok=True)
    scene_dir = args.generated_scene_dir or Path(args.g1_xml).resolve().parent
    scene_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.summary_csv or (
        args.root / "logs" / f"sim2sim_batch_{args.planner}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    _check_inputs(args)
    rows: list[EpisodeResult] = []
    planner = _make_planner(args)
    controller = _make_controller(args)
    start_wall = time.perf_counter()
    for episode in range(args.episodes):
        seed = args.seed_start + episode
        result = run_episode(args, episode, seed, scene_dir, planner, controller)
        rows.append(result)
        if args.print_every_episode > 0 and (
            episode == 0 or (episode + 1) % args.print_every_episode == 0 or episode + 1 == args.episodes
        ):
            print(_format_progress(rows, episode + 1, args.episodes, start_wall, args.exclude_timeout_from_metrics))
        if args.max_failures > 0 and sum(not row.success for row in rows) >= args.max_failures:
            print(f"[SIM2SIM_BATCH] early stop: failures reached {args.max_failures}")
            break

    _write_summary(summary_csv, rows)
    elapsed = time.perf_counter() - start_wall
    print(_format_final(rows, elapsed, args.exclude_timeout_from_metrics))
    print(f"[SIM2SIM_BATCH] wrote summary: {summary_csv}")
    if not args.keep_scenes:
        _cleanup_scenes(scene_dir)


def _apply_eval_preset(args: argparse.Namespace) -> None:
    if args.eval_preset != "fast":
        return
    args.fdm_population_size = min(args.fdm_population_size, 128)
    args.fdm_mppi_iterations = min(args.fdm_mppi_iterations, 3)
    args.fdm_replan_interval = max(args.fdm_replan_interval, 10)


def run_episode(
    args: argparse.Namespace,
    episode: int,
    seed: int,
    scene_dir: Path,
    planner,
    controller,
) -> EpisodeResult:
    xml_path = _make_episode_scene(args, seed, scene_dir)
    height_scan = _make_height_scan(args)

    try:
        env = MujocoG1Env(xml_path=xml_path, controller=controller, height_scan=height_scan, physics_dt=args.physics_dt)
    except MissingMuJoCoError as exc:
        raise RuntimeError(str(exc)) from exc

    env.reset()
    planner.reset()
    goal = np.asarray(args.goal, dtype=np.float32)
    last_command = LowLevelCommand.zeros()
    start_xy = env.base_xy_yaw()[:2].astype(np.float64)
    prev_xy = start_xy.copy()
    reference_path_length = float(np.linalg.norm(goal[:2].astype(np.float64) - start_xy))
    reference_time = reference_path_length / max(float(args.spt_reference_speed), 1e-6)
    path_length = 0.0
    min_height = float("inf")
    max_roll = 0.0
    max_pitch = 0.0
    max_ctrl_abs = 0.0
    max_obstacle_pixels = 0
    max_height_fdm_hit_count = 0
    max_height_fdm_geom_count = 0
    height_fdm_x_min = float("nan")
    height_fdm_x_max = float("nan")
    height_fdm_y_min = float("nan")
    height_fdm_y_max = float("nan")
    height_top_geoms_at_max_fdm_hits = ""
    max_fdm_risk = 0.0
    max_fdm_scan_cost = 0.0
    illegal_contact_steps = 0
    illegal_contact_streak = 0
    max_illegal_contact_streak = 0
    last_illegal_contact_pair = ""
    reached_once = False
    reached_step = -1
    reached_path_length = 0.0
    status = "timeout"
    wall_start = time.perf_counter()
    step_idx = 0

    for step_idx in range(args.steps):
        height_scan_obs = env.observe_height_scan()
        obs = PlannerObservation(
            start_xy_yaw=env.base_xy_yaw(),
            goal_xy_yaw=goal,
            height_scan=height_scan_obs,
            fdm_state=env.observe_fdm_state(),
            fdm_proprioception=env.observe_fdm_proprioception(last_command),
        )
        command = planner.command(obs)
        env.step(command, decimation=args.control_decimation)
        last_command = command
        pose = env.base_xyz_rpy()
        xy = pose[:2].astype(np.float64)
        path_length += float(np.linalg.norm(xy - prev_xy))
        prev_xy = xy

        roll = abs(float(pose[3]))
        pitch = abs(float(pose[4]))
        min_height = min(min_height, float(pose[2]))
        max_roll = max(max_roll, roll)
        max_pitch = max(max_pitch, pitch)
        max_ctrl_abs = max(max_ctrl_abs, float(np.max(np.abs(env.data.ctrl))) if env.model.nu else 0.0)
        obstacle_pixels = _count_obstacle_pixels(
            height_scan_obs,
            threshold=args.fdm_scan_obstacle_height_threshold,
            relative_to_floor=args.fdm_scan_obstacle_relative_to_floor,
            floor_percentile=args.fdm_scan_floor_percentile,
        )
        max_obstacle_pixels = max(max_obstacle_pixels, obstacle_pixels)
        scan_debug = env.height_scan.debug_info() if hasattr(env.height_scan, "debug_info") else {}
        fdm_hit_count = int(scan_debug.get("height_fdm_hit_count", 0) or 0)
        fdm_geom_count = int(scan_debug.get("height_fdm_geom_count", 0) or 0)
        if fdm_hit_count > max_height_fdm_hit_count:
            max_height_fdm_hit_count = fdm_hit_count
            height_top_geoms_at_max_fdm_hits = str(scan_debug.get("height_top_geoms", ""))
        max_height_fdm_geom_count = max(max_height_fdm_geom_count, fdm_geom_count)
        if fdm_hit_count > 0:
            height_fdm_x_min = _nanmin(height_fdm_x_min, float(scan_debug.get("height_fdm_x_min", float("nan"))))
            height_fdm_x_max = _nanmax(height_fdm_x_max, float(scan_debug.get("height_fdm_x_max", float("nan"))))
            height_fdm_y_min = _nanmin(height_fdm_y_min, float(scan_debug.get("height_fdm_y_min", float("nan"))))
            height_fdm_y_max = _nanmax(height_fdm_y_max, float(scan_debug.get("height_fdm_y_max", float("nan"))))
        debug = planner.debug_info()
        risk = float(debug.get("fdm_best_risk_max", 0.0))
        scan_cost = float(debug.get("fdm_cost_scan_obstacle", 0.0))
        max_fdm_risk = max(max_fdm_risk, risk)
        max_fdm_scan_cost = max(max_fdm_scan_cost, scan_cost)
        illegal_pairs = env.illegal_contact_pairs()
        illegal_contact = bool(illegal_pairs)
        if illegal_contact:
            illegal_contact_steps += 1
            illegal_contact_streak += 1
            last_illegal_contact_pair = ";".join(f"{body}:{geom}" for body, geom in illegal_pairs[:4])
        else:
            illegal_contact_streak = 0
        max_illegal_contact_streak = max(max_illegal_contact_streak, illegal_contact_streak)

        distance, x_error, yaw_error = _goal_error(pose, goal)
        goal_reached = (
            distance <= args.success_distance
            and x_error <= args.success_x_distance
            and yaw_error <= args.success_yaw
        )
        if goal_reached and not reached_once:
            reached_once = True
            reached_step = step_idx
            reached_path_length = path_length
        orientation_fall = (roll >= args.fall_roll_pitch or pitch >= args.fall_roll_pitch) and not args.ignore_orientation_fall
        if pose[2] <= args.fall_height or orientation_fall:
            status = "fall"
            break
        if args.illegal_contact_grace_steps > 0 and illegal_contact_streak >= args.illegal_contact_grace_steps:
            status = "collision"
            break
        if goal_reached:
            status = "success"
            break
    else:
        pose = env.base_xyz_rpy()
        distance, x_error, yaw_error = _goal_error(pose, goal)

    if status != "success" and reached_once:
        status = "success"
        step_idx = reached_step
        path_length = reached_path_length
    wall_time = time.perf_counter() - wall_start
    sim_time = (step_idx + 1) * args.control_decimation * float(args.physics_dt)
    pose = env.base_xyz_rpy()
    distance, x_error, yaw_error = _goal_error(pose, goal)
    spl = _success_weighted_ratio(status == "success", reference_path_length, path_length)
    spt = _success_weighted_ratio(status == "success", reference_time, sim_time)
    return EpisodeResult(
        episode=episode,
        seed=seed,
        status=status,
        success=status == "success",
        steps=step_idx + 1,
        sim_time=sim_time,
        wall_time=wall_time,
        final_x=float(pose[0]),
        final_y=float(pose[1]),
        final_yaw=float(pose[5]),
        goal_distance=distance,
        goal_x_error=x_error,
        yaw_error=yaw_error,
        reference_path_length=reference_path_length,
        path_length=path_length,
        spl=spl,
        reference_time=reference_time,
        spt=spt,
        min_height=min_height,
        max_roll=max_roll,
        max_pitch=max_pitch,
        max_ctrl_abs=max_ctrl_abs,
        max_obstacle_pixels=max_obstacle_pixels,
        max_height_fdm_hit_count=max_height_fdm_hit_count,
        max_height_fdm_geom_count=max_height_fdm_geom_count,
        height_fdm_x_min=height_fdm_x_min,
        height_fdm_x_max=height_fdm_x_max,
        height_fdm_y_min=height_fdm_y_min,
        height_fdm_y_max=height_fdm_y_max,
        height_top_geoms_at_max_fdm_hits=height_top_geoms_at_max_fdm_hits,
        max_fdm_risk=max_fdm_risk,
        max_fdm_scan_cost=max_fdm_scan_cost,
        illegal_contact_steps=illegal_contact_steps,
        max_illegal_contact_streak=max_illegal_contact_streak,
        last_illegal_contact_pair=last_illegal_contact_pair,
    )


def _make_episode_scene(args: argparse.Namespace, seed: int, scene_dir: Path) -> Path:
    obstacles = generate_fdm_terrain_obstacles(args.fdm_terrain_cfg, seed=seed)
    output_xml = scene_dir / f"batch_eval_{Path(args.g1_xml).stem}_{args.fdm_terrain_cfg}_seed_{seed}.xml"
    return generate_scene_with_obstacles(Path(args.g1_xml), obstacles, output_xml)


def _make_controller(args: argparse.Namespace):
    if args.zero_controller:
        return ZeroTorqueController()
    if args.low_level_policy is not None and args.low_level_policy.exists():
        return TorchPolicyPDController(
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
    return MissingGaitController()


def _make_planner(args: argparse.Namespace):
    if args.planner == "zero":
        return ZeroPlannerAdapter()
    if args.planner == "goal":
        return GoalTrackingAdapter(max_vx=args.max_vx, max_vy=args.max_vy, max_wz=args.max_wz)
    return FDMPlannerAdapter(
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
        scan_obstacle_relative_to_floor=args.fdm_scan_obstacle_relative_to_floor,
        scan_floor_percentile=args.fdm_scan_floor_percentile,
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


def _make_height_scan(args: argparse.Namespace):
    if args.height_scan == "raycast":
        return RaycastHeightScan(
            shape=Sim2SimConfig.height_scan_shape,
            resolution=Sim2SimConfig.height_scan_resolution,
            x_offset=args.height_scan_offset_x,
            y_offset=args.height_scan_offset_y,
            z_start=args.height_scan_z_start,
        )
    return FlatHeightScan(shape=Sim2SimConfig.height_scan_shape, resolution=Sim2SimConfig.height_scan_resolution)


def _goal_error(pose: np.ndarray, goal: np.ndarray) -> tuple[float, float, float]:
    distance = float(np.linalg.norm(pose[:2].astype(np.float64) - goal[:2].astype(np.float64)))
    x_error = float(abs(float(goal[0]) - float(pose[0])))
    yaw_error = float(abs(math.atan2(math.sin(float(goal[2] - pose[5])), math.cos(float(goal[2] - pose[5])))))
    return distance, x_error, yaw_error


def _count_obstacle_pixels(
    height_scan: np.ndarray,
    *,
    threshold: float,
    relative_to_floor: bool,
    floor_percentile: float,
) -> int:
    height = np.asarray(height_scan, dtype=np.float32)
    if not relative_to_floor:
        return int(np.count_nonzero(height > threshold))
    finite_height = height[np.isfinite(height)]
    if finite_height.size == 0:
        return 0
    floor_height = float(np.percentile(finite_height, floor_percentile))
    return int(np.count_nonzero((height - floor_height) > threshold))


def _nanmin(current: float, value: float) -> float:
    if math.isnan(current):
        return value
    if math.isnan(value):
        return current
    return min(current, value)


def _nanmax(current: float, value: float) -> float:
    if math.isnan(current):
        return value
    if math.isnan(value):
        return current
    return max(current, value)


def _success_weighted_ratio(success: bool, reference: float, actual: float) -> float:
    if not success or reference <= 0.0 or actual <= 0.0:
        return 0.0
    return float(reference / max(reference, actual))


def _write_summary(path: Path, rows: list[EpisodeResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(EpisodeResult.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _metric_rows(rows: list[EpisodeResult], exclude_timeout: bool) -> list[EpisodeResult]:
    if not exclude_timeout:
        return rows
    return [row for row in rows if row.status != "timeout"]


def _format_progress(
    rows: list[EpisodeResult],
    done: int,
    total: int,
    start_wall: float,
    exclude_timeout: bool,
) -> str:
    metric_rows = _metric_rows(rows, exclude_timeout)
    success = sum(row.success for row in metric_rows)
    status_counts = _status_counts(rows)
    mean_dist = float(np.mean([row.goal_distance for row in metric_rows])) if metric_rows else float("nan")
    elapsed = time.perf_counter() - start_wall
    eta = elapsed / max(done, 1) * max(total - done, 0)
    denom = len(metric_rows)
    return (
        f"[SIM2SIM_BATCH] {done}/{total} success={success}/{denom} "
        f"rate={success / max(denom, 1):.3f} mean_dist={mean_dist:.3f} "
        f"elapsed={elapsed / 60.0:.1f}min eta={eta / 60.0:.1f}min status={status_counts}"
    )


def _format_final(rows: list[EpisodeResult], elapsed: float, exclude_timeout: bool) -> str:
    metric_rows = _metric_rows(rows, exclude_timeout)
    success = sum(row.success for row in metric_rows)
    success_rows = [row for row in metric_rows if row.success]
    mean_time = float(np.mean([row.sim_time for row in metric_rows])) if metric_rows else float("nan")
    mean_path = float(np.mean([row.path_length for row in metric_rows])) if metric_rows else float("nan")
    mean_dist = float(np.mean([row.goal_distance for row in metric_rows])) if metric_rows else float("nan")
    mean_spl = float(np.mean([row.spl for row in metric_rows])) if metric_rows else float("nan")
    mean_spt = float(np.mean([row.spt for row in metric_rows])) if metric_rows else float("nan")
    success_spl = float(np.mean([row.spl for row in success_rows])) if success_rows else float("nan")
    success_spt = float(np.mean([row.spt for row in success_rows])) if success_rows else float("nan")
    denom = len(metric_rows)
    timeout_note = "timeouts_excluded" if exclude_timeout else "timeouts_included"
    return (
        "[SIM2SIM_BATCH] completed "
        f"episodes={len(rows)} metric_episodes={denom} {timeout_note} "
        f"success_rate={success / max(denom, 1):.3f} status={_status_counts(rows)} "
        f"mean_sim_time={mean_time:.2f}s "
        f"mean_path={mean_path:.2f}m mean_goal_distance={mean_dist:.3f}m "
        f"SPL_all={mean_spl:.3f} SPL_success={success_spl:.3f} "
        f"SPT_all={mean_spt:.3f} SPT_success={success_spt:.3f} wall_time={elapsed:.1f}s"
    )


def _status_counts(rows: list[EpisodeResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
    return counts


def _check_inputs(args: argparse.Namespace) -> None:
    missing = []
    if not Path(args.g1_xml).exists():
        missing.append(f"G1 MuJoCo XML: {args.g1_xml}")
    if args.planner == "fdm" and (args.fdm_checkpoint is None or not Path(args.fdm_checkpoint).exists()):
        missing.append(f"FDM checkpoint: {args.fdm_checkpoint}")
    if not args.zero_controller and (args.low_level_policy is None or not Path(args.low_level_policy).exists()):
        missing.append(f"G1 low-level policy: {args.low_level_policy}")
    if missing:
        raise FileNotFoundError("Missing inputs:\n  - " + "\n  - ".join(missing))


def _cleanup_scenes(scene_dir: Path) -> None:
    for path in scene_dir.glob("batch_eval_*.xml"):
        try:
            path.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
