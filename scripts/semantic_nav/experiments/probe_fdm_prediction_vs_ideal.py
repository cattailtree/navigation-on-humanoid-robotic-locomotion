from __future__ import annotations

"""Compare FDM rollout against ideal 0.5s SE(2) command integration."""

import argparse
import csv
import os
from math import atan2, cos, degrees, sin
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher

SEMANTIC_NAV_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = SEMANTIC_NAV_ROOT.parent
REPO_ROOT = SEMANTIC_NAV_ROOT.parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
if str(SEMANTIC_NAV_ROOT) not in sys.path:
    sys.path.insert(0, str(SEMANTIC_NAV_ROOT))

import utils.cli_args as cli_args  # isort: skip
from envs.abstract_building_env import DEFAULT_BUILDING_CONFIG  # isort: skip


DEFAULT_RUN_DIR = REPO_ROOT / "logs" / "fdm" / "fdm_se2_prediction_depth" / "May12_14-21-45_fdm_train"
DEFAULT_CHECKPOINT = DEFAULT_RUN_DIR / "model_collection_round_14.pth"
DEFAULT_SUMMARY_CSV = REPO_ROOT / "docs" / "velocity_tracking_probe" / "mppi_command_timestep_5s_deviation.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "docs" / "fdm_prediction_probe"
DEFAULT_WINDOWS: tuple[tuple[str, int, int], ...] = (
    ("A target-start", 200, 450),
    ("B mid-turn", 650, 900),
    ("C final-approach", 950, 1200),
)
FULL_TAPE_WINDOWS: tuple[tuple[str, int, int], ...] = (
    ("A target-start", 8, 18),
    ("B mid-turn", 26, 36),
    ("C final-approach", 38, 48),
)
APPROX_FULL_TAPE: tuple[tuple[float, float, float], ...] = (
    (0.05, 0.02, 0.00),
    (0.15, 0.04, 0.45),
    (0.30, 0.05, 0.55),
    (0.45, 0.06, 0.55),
    (0.55, 0.07, 0.55),
    (0.50, 0.07, 0.45),
    (0.42, 0.07, 0.30),
    (0.35, 0.06, 0.12),
    (0.24, 0.05, -0.05),
    (0.36, 0.06, 0.12),
    (0.30, 0.08, 0.35),
    (0.50, 0.08, 0.55),
    (0.28, 0.07, 0.55),
    (0.22, 0.05, 0.30),
    (0.34, 0.04, -0.05),
    (0.40, 0.03, -0.20),
    (0.32, 0.03, -0.05),
    (0.28, 0.03, 0.08),
    (0.26, 0.04, -0.10),
    (0.32, 0.05, -0.35),
    (0.25, 0.06, -0.58),
    (0.30, 0.07, 0.50),
    (0.42, 0.08, 0.58),
    (0.58, 0.08, 0.58),
    (0.40, 0.07, 0.20),
    (0.25, 0.04, -0.05),
    (0.55, -0.04, -0.10),
    (0.42, -0.06, -0.28),
    (0.36, -0.07, -0.48),
    (0.30, -0.08, -0.62),
    (0.26, -0.08, -0.66),
    (0.24, -0.07, -0.62),
    (0.32, -0.06, -0.58),
    (0.30, -0.05, -0.52),
    (0.28, -0.04, -0.45),
    (0.30, -0.03, -0.35),
    (0.24, 0.02, -0.10),
    (0.40, 0.04, 0.20),
    (0.68, 0.06, 0.50),
    (0.45, 0.06, 0.25),
    (0.24, 0.04, -0.05),
    (0.34, 0.03, -0.18),
    (0.46, 0.03, 0.05),
    (0.40, 0.04, 0.25),
    (0.32, 0.04, 0.38),
    (0.28, 0.04, 0.30),
    (0.35, 0.04, 0.32),
    (0.34, 0.04, 0.35),
    (0.28, 0.04, 0.28),
    (0.22, 0.04, 0.10),
)
APPROX_FULL_RUN_COMMANDS: dict[str, tuple[tuple[float, float, float], ...]] = {
    # Hand-picked from docs/semantic_nav_full_run_diagnostic.png.  The values
    # intentionally capture the trend of each 5s segment rather than the exact
    # overwritten CSV: target-start forward/right turn, mid-run strong left turn,
    # and final approach with mild right turn.
    "A target-start": (
        (0.22, 0.05, -0.05),
        (0.35, 0.06, 0.10),
        (0.28, 0.08, 0.35),
        (0.50, 0.08, 0.55),
        (0.25, 0.07, 0.55),
        (0.18, 0.04, 0.30),
        (0.32, 0.03, -0.05),
        (0.38, 0.02, -0.20),
        (0.30, 0.02, -0.05),
        (0.26, 0.03, 0.08),
    ),
    "B mid-turn": (
        (0.15, 0.00, 0.00),
        (0.22, -0.02, -0.05),
        (0.28, -0.03, -0.10),
        (0.32, -0.04, -0.14),
        (0.30, -0.04, -0.18),
        (0.28, -0.04, -0.20),
        (0.28, -0.03, -0.18),
        (0.28, -0.03, -0.14),
        (0.26, -0.02, -0.10),
        (0.24, -0.02, -0.06),
    ),
    "C final-approach": (
        (0.25, 0.06, -0.05),
        (0.45, 0.08, 0.10),
        (0.65, 0.08, 0.45),
        (0.35, 0.06, 0.20),
        (0.18, 0.04, -0.10),
        (0.30, 0.03, -0.20),
        (0.42, 0.03, 0.05),
        (0.35, 0.04, 0.25),
        (0.28, 0.04, 0.38),
        (0.24, 0.04, 0.30),
    ),
}


parser = argparse.ArgumentParser(description="Probe FDM prediction vs ideal command integration in the Lab elevator scene.")
parser.add_argument(
    "--command-source",
    choices=("approx_full_tape", "approx_full_run", "traj", "traj_full_tape", "snapshot", "snapshot_full", "summary_mean"),
    default="approx_full_tape",
)
parser.add_argument("--traj-csv", type=Path, default=None, help="Full-run traj.csv used to extract 0.5s command windows.")
parser.add_argument("--snapshot-npz", type=Path, default=None, help="Run-time FDM snapshots captured during the semantic navigation run.")
parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV, help="Fallback summary CSV with mean commands.")
parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
parser.add_argument("--building-config", type=Path, default=DEFAULT_BUILDING_CONFIG)
parser.add_argument("--start-pose", type=float, nargs=3, default=(0.0, 0.0, 0.0))
parser.add_argument("--corridor-lobby-elevator-pose", type=float, nargs=3, default=(8.4, 3.0, 3.14159))
parser.add_argument("--blind-arena-center", type=float, nargs=2, default=(5.5, 0.4))
parser.add_argument("--blind-arena-size", type=float, nargs=2, default=(13.0, 9.2))
parser.add_argument("--duration-s", type=float, default=5.0)
parser.add_argument("--command-timestep-s", type=float, default=0.5)
parser.add_argument("--warmup-s", type=float, default=1.0)
parser.add_argument("--actual-prelude-s", type=float, default=1.5, help="Low-speed forward command before recording actual rollout.")
parser.add_argument("--actual-prelude-vx", type=float, default=0.15)
parser.add_argument("--episode-length-s", type=float, default=20.0)
parser.add_argument("--force-exit", action="store_true", help="Exit immediately after outputs to avoid Isaac shutdown hangs.")
cli_args.add_fdm_args(parser, default_num_envs=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.num_envs = 1
args_cli.robot = "g1"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

import fdm.mdp as mdp  # noqa: E402
from fdm.utils.args_cli_utils import cfg_modifier_pre_init, planner_cfg_init  # noqa: E402
from envs.abstract_building_env import load_semantic_graph  # noqa: E402
from executors.fdm_mppi_waypoint_executor import FdmMppiExecutorConfig, FdmMppiWaypointExecutor  # noqa: E402
from executors.lab_robot_adapter import LabRobotAdapter  # noqa: E402
from executors.robot_adapter import VelocityCommand  # noqa: E402
from executors.waypoint_executor import WaypointExecutorConfig  # noqa: E402
from lab_scene.elevator_scene import spawn_blind_search_arena, spawn_corridor_lobby_walls, spawn_elevator_nodes  # noqa: E402
from maps.semantic_graph import Pose2D, SemanticNode  # noqa: E402
from planners.execution_plan import ExecutionStep  # noqa: E402


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args_cli.out_dir.mkdir(parents=True, exist_ok=True)
    status_log = args_cli.out_dir / "fdm_prediction_probe_status.log"
    status_log.unlink(missing_ok=True)
    step_rows: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | int | str]] = []
    full_tape_rows: list[dict[str, float | int | str | bool]] = []
    env = None

    try:
        _status(status_log, "loading_command_windows")
        command_windows, source_label = _load_command_windows()
        _status(status_log, f"command_source={source_label} windows={len(command_windows)}")

        cfg = planner_cfg_init(args_cli)
        cfg = cfg_modifier_pre_init(cfg, args_cli)
        cfg.env_cfg.scene.num_envs = 1
        cfg.env_cfg.episode_length_s = args_cli.episode_length_s
        cfg.env_cfg.scene.terrain.terrain_type = "plane"
        cfg.env_cfg.scene.terrain.terrain_generator = None
        cfg.env_cfg.scene.terrain.random_seed = 0
        cfg.env_cfg.events.reset_base.func = mdp.reset_root_state_center
        cfg.env_cfg.events.reset_base.params = {}

        _status(status_log, "creating_env")
        env = ManagerBasedRLEnv(cfg.env_cfg)
        robot = LabRobotAdapter(env)
        start_pose = Pose2D(args_cli.start_pose[0], args_cli.start_pose[1], args_cli.start_pose[2])
        _spawn_matching_scene(env)

        dummy_step = ExecutionStep(
            kind="walk_to",
            node_id="probe_goal",
            floor="F1",
            pose=Pose2D(start_pose.x + 1.0, start_pose.y, start_pose.yaw),
            description="probe only",
        )
        executor = FdmMppiWaypointExecutor(
            env=env,
            steps=[dummy_step],
            waypoint_cfg=WaypointExecutorConfig(),
            planner_cfg=cfg,
            fdm_cfg=FdmMppiExecutorConfig(
                run_dir=args_cli.run_dir,
                checkpoint=args_cli.checkpoint,
                use_fdm=True,
                population_size=16,
                replan_every=5,
            ),
        )

        model_dt = float(getattr(executor.model.cfg, "command_timestep", args_cli.command_timestep_s))
        if abs(model_dt - args_cli.command_timestep_s) > 1.0e-4:
            print(
                f"[fdm_prediction_probe:warn] requested_dt={args_cli.command_timestep_s:.3f}s "
                f"but model command_timestep={model_dt:.3f}s; using model dt",
                flush=True,
            )
        horizon = int(round(args_cli.duration_s / model_dt))
        if horizon != executor._horizon:
            print(
                f"[fdm_prediction_probe:warn] 5s horizon={horizon} but model horizon={executor._horizon}; "
                f"using min horizon",
                flush=True,
            )
            horizon = min(horizon, executor._horizon)

        if args_cli.command_source == "snapshot_full":
            _status(status_log, "loading full run-time FDM snapshot timeline")
            full_snapshot_rows, source_label = _build_full_snapshot_prediction(
                executor=executor,
                snapshot_path=args_cli.snapshot_npz,
                traj_path=args_cli.traj_csv,
                command_dt=model_dt,
                horizon=horizon,
            )
            full_snapshot_csv = args_cli.out_dir / "fdm_snapshot_full_prediction.csv"
            full_snapshot_plot = args_cli.out_dir / "fdm_snapshot_full_prediction.png"
            _write_csv(full_snapshot_csv, full_snapshot_rows)
            _plot_full_snapshot_prediction(full_snapshot_rows, full_snapshot_plot, source_label)
            print(f"[fdm_prediction_probe] command_source={source_label}", flush=True)
            print(f"[fdm_prediction_probe] wrote full_snapshot={full_snapshot_csv}", flush=True)
            print(f"[fdm_prediction_probe] wrote full_snapshot_plot={full_snapshot_plot}", flush=True)
            _status(status_log, "main_done")
            return

        if args_cli.command_source == "snapshot":
            _status(status_log, "loading run-time FDM snapshots")
            snapshot_windows, source_label = _load_windows_from_snapshot(args_cli.snapshot_npz, args_cli.traj_csv, model_dt, horizon)
            for window in snapshot_windows:
                window_name = str(window["name"])
                commands = list(window["macro_commands"])
                start_pose_window = window["start_pose"]
                fdm_states = _fdm_rollout(
                    executor,
                    {"planner_obs": {}, "fdm_obs_exteroceptive": window["extero_obs"]},
                    start_pose_window,
                    commands,
                    state_history=window["state_history"],
                    proprio_history=window["proprio_history"],
                )
                ideal_states = _ideal_rollout(start_pose_window, commands, model_dt)
                actual_states = _relative_segment_states(
                    start_pose_window,
                    start_pose_window,
                    window["actual_sample_poses"],
                )
                _append_rows(
                    step_rows,
                    summary_rows,
                    window_name,
                    commands,
                    ideal_states,
                    fdm_states,
                    actual_states,
                    actual_reset=False,
                    prelude_reset=False,
                    dt=model_dt,
                )
                _status(status_log, f"snapshot_window_done {window_name}")
        elif args_cli.command_source in ("approx_full_tape", "traj_full_tape"):
            _status(status_log, "running full command tape")
            if args_cli.command_source == "traj_full_tape":
                (
                    step_commands,
                    tape_windows,
                    csv_pose_rows,
                    source_label,
                    csv_lab_dt,
                ) = _load_step_replay_from_traj(args_cli.traj_csv, model_dt, horizon)
                tape_result = _run_csv_step_replay(
                    robot=robot,
                    executor=executor,
                    start_pose=start_pose,
                    commands=step_commands,
                    step_dt=csv_lab_dt,
                    snapshot_rows=[window["row_start"] for window in tape_windows],
                )
                full_tape_rows = _full_tape_rows(step_commands, tape_result, csv_lab_dt)
            else:
                commands = list(APPROX_FULL_TAPE)
                tape_windows = list(FULL_TAPE_WINDOWS)
                tape_result = _run_full_tape(
                    robot=robot,
                    executor=executor,
                    start_pose=start_pose,
                    commands=commands,
                    command_dt=model_dt,
                    horizon=horizon,
                    windows=tape_windows,
                )
                full_tape_rows = _full_tape_rows(commands, tape_result, model_dt)
            for window in tape_windows:
                if isinstance(window, dict):
                    window_name = str(window["name"])
                    row_start = int(window["row_start"])
                    row_end = int(window["row_end"])
                    window_commands = list(window["macro_commands"])
                    window_start_pose = tape_result["poses"][row_start]
                    csv_base = start_pose if row_start == 0 else csv_pose_rows[row_start - 1]
                    actual_sample_poses = [csv_pose_rows[idx] for idx in window["actual_sample_rows"]]
                    actual_states = _relative_segment_states(window_start_pose, csv_base, actual_sample_poses)
                    snapshot_key = row_start
                else:
                    window_name, start_idx, end_idx = window
                    window_commands = list(commands[start_idx:end_idx])
                    window_start_pose = tape_result["poses"][start_idx]
                    actual_states = _relative_segment_states(
                        window_start_pose,
                        window_start_pose,
                        tape_result["poses"][start_idx + 1 : end_idx + 1],
                    )
                    snapshot_key = start_idx
                    row_start = start_idx
                    row_end = end_idx
                fdm_states = _fdm_rollout(
                    executor,
                    tape_result["obs_snapshots"][snapshot_key],
                    window_start_pose,
                    window_commands,
                    state_history=tape_result["state_history_snapshots"][snapshot_key],
                    proprio_history=tape_result["proprio_history_snapshots"][snapshot_key],
                )
                ideal_states = _ideal_rollout(window_start_pose, window_commands, model_dt)
                _append_rows(
                    step_rows,
                    summary_rows,
                    window_name,
                    window_commands,
                    ideal_states,
                    fdm_states,
                    actual_states,
                    actual_reset=any(tape_result["reset_flags"][row_start : row_end + 1]),
                    prelude_reset=any(tape_result["reset_flags"][:row_start]),
                    dt=model_dt,
                )
        else:
            normalized_windows: dict[str, list[tuple[float, float, float]]] = {}
            for window_name, raw_commands in command_windows.items():
                commands = raw_commands[:horizon]
                if len(commands) < horizon:
                    commands = commands + [commands[-1]] * (horizon - len(commands))
                normalized_windows[window_name] = commands

            for window_name, commands in normalized_windows.items():
                _status(status_log, f"window_start {window_name} commands={len(commands)}")
                probe_obs, base_pose, prelude_reset = _run_prelude_and_collect_obs(
                    robot,
                    executor,
                    start_pose,
                    commands,
                )
                fdm_states = _fdm_rollout(executor, probe_obs, start_pose, commands)
                actual_states, actual_reset = _actual_rollout(robot, start_pose, base_pose, commands, model_dt)
                ideal_states = _ideal_rollout(start_pose, commands, model_dt)
                _append_rows(
                    step_rows,
                    summary_rows,
                    window_name,
                    commands,
                    ideal_states,
                    fdm_states,
                    actual_states,
                    actual_reset=actual_reset,
                    prelude_reset=prelude_reset,
                    dt=model_dt,
                )
                _status(status_log, f"window_done {window_name}")

        steps_csv = args_cli.out_dir / "fdm_vs_ideal_steps.csv"
        summary_csv = args_cli.out_dir / "fdm_vs_ideal_summary.csv"
        plot_path = args_cli.out_dir / "fdm_vs_ideal_prediction.png"
        full_tape_csv = args_cli.out_dir / "full_tape_actual.csv"
        full_tape_plot = args_cli.out_dir / "full_tape_actual_overview.png"
        _write_csv(steps_csv, step_rows)
        _write_csv(summary_csv, summary_rows)
        if full_tape_rows:
            _write_csv(full_tape_csv, full_tape_rows)
            _plot_full_tape(full_tape_rows, full_tape_plot)
        _plot_results(step_rows, summary_rows, plot_path, source_label)
        print(f"[fdm_prediction_probe] command_source={source_label}", flush=True)
        print(f"[fdm_prediction_probe] wrote steps={steps_csv}", flush=True)
        print(f"[fdm_prediction_probe] wrote summary={summary_csv}", flush=True)
        print(f"[fdm_prediction_probe] wrote plot={plot_path}", flush=True)
        if full_tape_rows:
            print(f"[fdm_prediction_probe] wrote full_tape={full_tape_csv}", flush=True)
            print(f"[fdm_prediction_probe] wrote full_tape_plot={full_tape_plot}", flush=True)
        _status(status_log, "main_done")
        if args_cli.force_exit:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
    except BaseException as exc:
        _status(status_log, f"exception {type(exc).__name__}: {exc}")
        _status(status_log, traceback.format_exc())
        raise
    finally:
        _status(status_log, "closing")
        if env is not None:
            env.close()
        simulation_app.close()


def _load_command_windows() -> tuple[dict[str, list[tuple[float, float, float]]], str]:
    if args_cli.command_source == "approx_full_tape":
        return {}, "approx full 0.5s command tape from semantic_nav_full_run_diagnostic.png"
    if args_cli.command_source == "traj_full_tape":
        return {}, str(args_cli.traj_csv)
    if args_cli.command_source in ("snapshot", "snapshot_full"):
        return {}, str(args_cli.snapshot_npz)
    if args_cli.command_source == "approx_full_run":
        return {name: list(commands) for name, commands in APPROX_FULL_RUN_COMMANDS.items()}, "approx commands from semantic_nav_full_run_diagnostic.png"
    if args_cli.command_source == "traj":
        if args_cli.traj_csv is None:
            raise ValueError("--command-source traj requires --traj-csv")
        return _load_windows_from_traj(args_cli.traj_csv), str(args_cli.traj_csv)
    if args_cli.command_source != "summary_mean":
        raise ValueError(f"Unknown command source: {args_cli.command_source}")
    return _load_windows_from_summary(args_cli.summary_csv), f"{args_cli.summary_csv} mean-command fallback"


def _load_windows_from_snapshot(
    snapshot_path: Path | None,
    traj_path: Path | None,
    command_dt: float,
    horizon: int,
) -> tuple[list[dict[str, object]], str]:
    if snapshot_path is None:
        raise ValueError("--command-source snapshot requires --snapshot-npz")
    if traj_path is None:
        raise ValueError("--command-source snapshot requires --traj-csv")
    data = np.load(snapshot_path, allow_pickle=True)
    traj_rows = list(csv.DictReader(traj_path.open("r", newline="", encoding="utf-8")))
    if not traj_rows:
        raise ValueError(f"Empty traj csv: {traj_path}")
    lab_dt = _estimate_csv_dt(traj_rows)
    stride = max(1, int(round(command_dt / lab_dt)))
    step_to_snapshot_idx = {int(step): idx for idx, step in enumerate(data["step"].tolist())}
    target_steps = [int(row["step"]) for row in traj_rows if "target" in row.get("mode", "")]
    if target_steps:
        target_start = min(target_steps)
        target_end = max(target_steps)
    else:
        target_start = int(traj_rows[0]["step"])
        target_end = int(traj_rows[-1]["step"])
    window_steps = horizon * stride
    max_start = max(target_start, target_end - window_steps)
    candidate_starts = [
        target_start,
        target_start + max(0, (target_end - target_start - window_steps) // 2),
        max_start,
    ]
    names = ("A target-start", "B target-middle", "C final-approach")
    windows: list[dict[str, object]] = []
    used: set[int] = set()
    for name, start_step in zip(names, candidate_starts):
        start_step = max(0, min(start_step, len(traj_rows) - window_steps - 1))
        while start_step in used and start_step + stride <= len(traj_rows) - window_steps - 1:
            start_step += stride
        used.add(start_step)
        if start_step not in step_to_snapshot_idx:
            available = np.asarray(data["step"], dtype=np.int64)
            nearest = int(available[np.argmin(np.abs(available - start_step))])
            start_step = nearest
        snapshot_idx = step_to_snapshot_idx[start_step]
        macro_rows = [start_step + i * stride for i in range(horizon)]
        actual_rows = [min(start_step + (i + 1) * stride - 1, len(traj_rows) - 1) for i in range(horizon)]
        macro_commands = [
            (
                float(traj_rows[row_idx]["cmd_vx"]),
                float(traj_rows[row_idx]["cmd_vy"]),
                float(traj_rows[row_idx]["cmd_wz"]),
            )
            for row_idx in macro_rows
        ]
        start_pose = Pose2D(
            float(traj_rows[start_step]["x"]),
            float(traj_rows[start_step]["y"]),
            float(traj_rows[start_step]["yaw"]),
        )
        actual_sample_poses = [
            Pose2D(float(traj_rows[row_idx]["x"]), float(traj_rows[row_idx]["y"]), float(traj_rows[row_idx]["yaw"]))
            for row_idx in actual_rows
        ]
        end_step = min(start_step + window_steps - 1, len(traj_rows) - 1)
        windows.append(
            {
                "name": f"{name} steps {start_step}-{end_step}",
                "macro_commands": macro_commands,
                "start_pose": start_pose,
                "actual_sample_poses": actual_sample_poses,
                "state_history": torch.as_tensor(data["state_history"][snapshot_idx : snapshot_idx + 1], dtype=torch.float32),
                "proprio_history": torch.as_tensor(data["proprio_history"][snapshot_idx : snapshot_idx + 1], dtype=torch.float32),
                "extero_obs": torch.as_tensor(data["extero_obs"][snapshot_idx], dtype=torch.float32),
            }
        )
    return windows, f"{snapshot_path} run-time FDM snapshots + {traj_path} future commands"


def _build_full_snapshot_prediction(
    *,
    executor: FdmMppiWaypointExecutor,
    snapshot_path: Path | None,
    traj_path: Path | None,
    command_dt: float,
    horizon: int,
) -> tuple[list[dict[str, float | int | str]], str]:
    if snapshot_path is None:
        raise ValueError("--command-source snapshot_full requires --snapshot-npz")
    if traj_path is None:
        raise ValueError("--command-source snapshot_full requires --traj-csv")
    data = np.load(snapshot_path, allow_pickle=True)
    traj_rows = list(csv.DictReader(traj_path.open("r", newline="", encoding="utf-8")))
    if not traj_rows:
        raise ValueError(f"Empty traj csv: {traj_path}")
    lab_dt = _estimate_csv_dt(traj_rows)
    stride = max(1, int(round(command_dt / lab_dt)))
    step_to_snapshot_idx = {int(step): idx for idx, step in enumerate(data["step"].tolist())}
    target_steps = [int(row["step"]) for row in traj_rows if "target" in row.get("mode", "")]
    target_start = min(target_steps) if target_steps else int(traj_rows[0]["step"])
    target_end = max(target_steps) if target_steps else int(traj_rows[-1]["step"])

    rows: list[dict[str, float | int | str]] = []
    ideal_pose = Pose2D(
        float(traj_rows[target_start]["x"]),
        float(traj_rows[target_start]["y"]),
        float(traj_rows[target_start]["yaw"]),
    )
    max_start = max(target_start, target_end - stride)
    start_steps = list(range(target_start, max_start + 1, stride))

    for idx, start_step in enumerate(start_steps):
        if start_step not in step_to_snapshot_idx:
            continue
        actual_pose = Pose2D(
            float(traj_rows[start_step]["x"]),
            float(traj_rows[start_step]["y"]),
            float(traj_rows[start_step]["yaw"]),
        )
        snapshot_idx = step_to_snapshot_idx[start_step]
        macro_commands: list[tuple[float, float, float]] = []
        for horizon_idx in range(horizon):
            row_idx = min(start_step + horizon_idx * stride, len(traj_rows) - 1)
            macro_commands.append(
                (
                    float(traj_rows[row_idx]["cmd_vx"]),
                    float(traj_rows[row_idx]["cmd_vy"]),
                    float(traj_rows[row_idx]["cmd_wz"]),
                )
            )
        fdm_states = _fdm_rollout(
            executor,
            {"planner_obs": {}, "fdm_obs_exteroceptive": torch.as_tensor(data["extero_obs"][snapshot_idx], dtype=torch.float32)},
            actual_pose,
            macro_commands,
            state_history=torch.as_tensor(data["state_history"][snapshot_idx : snapshot_idx + 1], dtype=torch.float32),
            proprio_history=torch.as_tensor(data["proprio_history"][snapshot_idx : snapshot_idx + 1], dtype=torch.float32),
        )
        one_step_ideal = _ideal_rollout(actual_pose, macro_commands[:1], command_dt)[0]
        next_step = min(start_step + stride, len(traj_rows) - 1)
        actual_next_pose = Pose2D(
            float(traj_rows[next_step]["x"]),
            float(traj_rows[next_step]["y"]),
            float(traj_rows[next_step]["yaw"]),
        )
        command_for_ideal = macro_commands[0]
        ideal_next = _ideal_rollout(ideal_pose, [command_for_ideal], command_dt)[0]
        ideal_pose = Pose2D(float(ideal_next[0]), float(ideal_next[1]), float(ideal_next[2]))

        fdm_next = fdm_states[0]
        rows.append(
            {
                "sample": idx,
                "step": start_step,
                "next_step": next_step,
                "time_s": (start_step - target_start) * lab_dt,
                "mode": str(traj_rows[start_step].get("mode", "")),
                "active": str(traj_rows[start_step].get("active", "")),
                "cmd_vx": macro_commands[0][0],
                "cmd_vy": macro_commands[0][1],
                "cmd_wz": macro_commands[0][2],
                "actual_x": actual_next_pose.x,
                "actual_y": actual_next_pose.y,
                "actual_yaw": actual_next_pose.yaw,
                "ideal_x": ideal_pose.x,
                "ideal_y": ideal_pose.y,
                "ideal_yaw": ideal_pose.yaw,
                "fdm_next_x": float(fdm_next[0]),
                "fdm_next_y": float(fdm_next[1]),
                "fdm_next_yaw": float(fdm_next[2]),
                "one_step_ideal_x": float(one_step_ideal[0]),
                "one_step_ideal_y": float(one_step_ideal[1]),
                "one_step_ideal_yaw": float(one_step_ideal[2]),
                "fdm_minus_actual_next_x": float(fdm_next[0] - float(traj_rows[next_step]["x"])),
                "fdm_minus_actual_next_y": float(fdm_next[1] - float(traj_rows[next_step]["y"])),
                "fdm_minus_actual_next_yaw_deg": degrees(_wrap_to_pi(float(fdm_next[2]) - float(traj_rows[next_step]["yaw"]))),
                "fdm_minus_one_step_ideal_x": float(fdm_next[0] - one_step_ideal[0]),
                "fdm_minus_one_step_ideal_y": float(fdm_next[1] - one_step_ideal[1]),
                "fdm_minus_one_step_ideal_yaw_deg": degrees(_wrap_to_pi(float(fdm_next[2]) - float(one_step_ideal[2]))),
            }
        )

    return rows, f"{snapshot_path} full run-time FDM snapshots + {traj_path} future 0.5s commands"


def _load_full_tape_from_traj(
    path: Path | None,
    command_dt: float,
    horizon: int,
) -> tuple[list[tuple[float, float, float]], list[tuple[str, int, int]], str]:
    if path is None:
        raise ValueError("--command-source traj_full_tape requires --traj-csv")
    rows = list(csv.DictReader(path.open("r", newline="", encoding="utf-8")))
    if not rows:
        raise ValueError(f"Empty traj csv: {path}")
    lab_dt = _estimate_csv_dt(rows)
    stride = max(1, int(round(command_dt / lab_dt)))
    sampled_indices = list(range(0, len(rows), stride))
    commands = [
        (float(rows[idx]["cmd_vx"]), float(rows[idx]["cmd_vy"]), float(rows[idx]["cmd_wz"]))
        for idx in sampled_indices
    ]
    target_steps = [int(row["step"]) for row in rows if "target" in row.get("mode", "")]
    if target_steps:
        target_start_step = min(target_steps)
        target_end_step = max(target_steps)
    else:
        target_start_step = int(rows[0]["step"])
        target_end_step = int(rows[-1]["step"])
    target_start_idx = _nearest_sample_index(sampled_indices, rows, target_start_step)
    target_end_idx = _nearest_sample_index(sampled_indices, rows, target_end_step)
    max_start = max(target_start_idx, target_end_idx - horizon)
    candidate_starts = [
        target_start_idx,
        target_start_idx + max(0, (target_end_idx - target_start_idx - horizon) // 2),
        max_start,
    ]
    names = ("A target-start", "B target-middle", "C final-approach")
    windows: list[tuple[str, int, int]] = []
    used: set[int] = set()
    for name, start_idx in zip(names, candidate_starts):
        start_idx = max(0, min(start_idx, len(commands) - horizon))
        while start_idx in used and start_idx + 1 <= len(commands) - horizon:
            start_idx += 1
        used.add(start_idx)
        windows.append((f"{name} idx {start_idx}-{start_idx + horizon}", start_idx, start_idx + horizon))
    return commands, windows, f"{path} full 0.5s command tape with replayed observations"


def _load_step_replay_from_traj(
    path: Path | None,
    command_dt: float,
    horizon: int,
) -> tuple[
    list[tuple[float, float, float]],
    list[dict[str, object]],
    list[Pose2D],
    str,
    float,
]:
    if path is None:
        raise ValueError("--command-source traj_full_tape requires --traj-csv")
    rows = list(csv.DictReader(path.open("r", newline="", encoding="utf-8")))
    if not rows:
        raise ValueError(f"Empty traj csv: {path}")
    lab_dt = _estimate_csv_dt(rows)
    stride = max(1, int(round(command_dt / lab_dt)))
    step_commands = [
        (float(row["cmd_vx"]), float(row["cmd_vy"]), float(row["cmd_wz"]))
        for row in rows
    ]
    csv_poses = [Pose2D(float(row["x"]), float(row["y"]), float(row["yaw"])) for row in rows]

    target_row_indices = [idx for idx, row in enumerate(rows) if "target" in row.get("mode", "")]
    if target_row_indices:
        target_start = min(target_row_indices)
        target_end = max(target_row_indices)
    else:
        target_start = 0
        target_end = len(rows) - 1
    window_rows = horizon * stride
    max_start = max(target_start, target_end - window_rows)
    candidate_starts = [
        target_start,
        target_start + max(0, (target_end - target_start - window_rows) // 2),
        max_start,
    ]
    names = ("A target-start", "B target-middle", "C final-approach")
    windows: list[dict[str, object]] = []
    used: set[int] = set()
    for name, start in zip(names, candidate_starts):
        start = max(0, min(start, len(rows) - window_rows - 1))
        while start in used and start + stride <= len(rows) - window_rows - 1:
            start += stride
        used.add(start)
        macro_rows = [start + i * stride for i in range(horizon)]
        actual_rows = [min(start + (i + 1) * stride - 1, len(rows) - 1) for i in range(horizon)]
        macro_commands = [step_commands[row_idx] for row_idx in macro_rows]
        end = min(start + window_rows - 1, len(rows) - 1)
        windows.append(
            {
                "name": f"{name} rows {start}-{end}",
                "row_start": start,
                "row_end": end,
                "macro_rows": macro_rows,
                "actual_sample_rows": actual_rows,
                "macro_commands": macro_commands,
            }
        )
    return step_commands, windows, csv_poses, f"{path} per-step replay; FDM uses segment-start 0.5s commands", lab_dt


def _load_csv_pose_rows(path: Path | None, command_dt: float) -> list[Pose2D]:
    if path is None:
        raise ValueError("--traj-csv is required for CSV pose rows")
    rows = list(csv.DictReader(path.open("r", newline="", encoding="utf-8")))
    lab_dt = _estimate_csv_dt(rows)
    stride = max(1, int(round(command_dt / lab_dt)))
    sampled_indices = list(range(0, len(rows), stride))
    return [
        Pose2D(float(rows[idx]["x"]), float(rows[idx]["y"]), float(rows[idx]["yaw"]))
        for idx in sampled_indices
    ]


def _estimate_csv_dt(rows: list[dict[str, str]]) -> float:
    if len(rows) < 3:
        return 0.02
    steps = [int(row["step"]) for row in rows[: min(50, len(rows))]]
    deltas = [b - a for a, b in zip(steps, steps[1:]) if b > a]
    if not deltas:
        return 0.02
    # CSV is logged once per Lab step in these experiments.
    return 0.02 * float(np.median(deltas))


def _nearest_sample_index(sampled_indices: list[int], rows: list[dict[str, str]], target_step: int) -> int:
    best_idx = 0
    best_delta = float("inf")
    for sample_idx, row_idx in enumerate(sampled_indices):
        delta = abs(int(rows[row_idx]["step"]) - target_step)
        if delta < best_delta:
            best_idx = sample_idx
            best_delta = delta
    return best_idx


def _load_windows_from_traj(path: Path) -> dict[str, list[tuple[float, float, float]]]:
    rows = list(csv.DictReader(path.open("r", newline="", encoding="utf-8")))
    by_step = {int(row["step"]): row for row in rows}
    stride = max(1, int(round(args_cli.command_timestep_s / 0.02)))
    windows: dict[str, list[tuple[float, float, float]]] = {}
    for name, start, end in DEFAULT_WINDOWS:
        commands: list[tuple[float, float, float]] = []
        for step in range(start, end, stride):
            row = by_step.get(step)
            if row is None:
                continue
            commands.append((float(row["cmd_vx"]), float(row["cmd_vy"]), float(row["cmd_wz"])))
        if not commands:
            raise ValueError(f"No commands found for window {name} in {path}")
        windows[name] = commands
    return windows


def _load_windows_from_summary(path: Path) -> dict[str, list[tuple[float, float, float]]]:
    rows = list(csv.DictReader(path.open("r", newline="", encoding="utf-8")))
    horizon = max(1, int(round(args_cli.duration_s / args_cli.command_timestep_s)))
    windows: dict[str, list[tuple[float, float, float]]] = {}
    for row in rows:
        name = row["window"]
        command = (float(row["mean_cmd_vx"]), float(row["mean_cmd_vy"]), float(row["mean_cmd_wz"]))
        windows[name] = [command] * horizon
    return windows


def _spawn_matching_scene(env) -> None:
    graph = load_semantic_graph(args_cli.building_config)
    if "elevator_f1" in graph.nodes:
        old_node = graph.nodes["elevator_f1"]
        pose = Pose2D(*args_cli.corridor_lobby_elevator_pose)
        graph.update_node(
            SemanticNode(
                node_id=old_node.node_id,
                floor=old_node.floor,
                kind=old_node.kind,
                pose=pose,
                label=old_node.label,
                attrs=old_node.attrs,
            )
        )
    elevator_poses = {
        node.node_id: node.pose
        for node in graph.nodes.values()
        if node.kind == "elevator_lobby"
    }
    if elevator_poses:
        spawn_elevator_nodes(
            origin=env.scene.env_origins[0],
            node_poses=elevator_poses,
            device=env.device,
            collision=False,
        )
    spawn_blind_search_arena(
        origin=env.scene.env_origins[0],
        center=Pose2D(args_cli.blind_arena_center[0], args_cli.blind_arena_center[1], 0.0),
        size=(args_cli.blind_arena_size[0], args_cli.blind_arena_size[1]),
    )
    spawn_corridor_lobby_walls(origin=env.scene.env_origins[0])
    env.sim.render()


def _reset_and_collect_probe_obs(robot: LabRobotAdapter, executor: FdmMppiWaypointExecutor, start_pose: Pose2D) -> dict:
    with torch.inference_mode():
        robot.reset(start_pose)
    executor.reset()
    zero = VelocityCommand(0.0, 0.0, 0.0)
    warmup_steps = max(1, int(round(args_cli.warmup_s / robot.env.step_dt)))
    obs = executor._collect_observations()
    for _ in range(warmup_steps):
        executor._update_obs_buffers(
            state=obs["fdm_state"].clone(),
            proprio=obs["fdm_obs_proprioception"].clone(),
        )
        robot.step_velocity(zero)
        obs = executor._collect_observations()
    executor._update_obs_buffers(
        state=obs["fdm_state"].clone(),
        proprio=obs["fdm_obs_proprioception"].clone(),
    )
    return obs


def _run_prelude_and_collect_obs(
    robot: LabRobotAdapter,
    executor: FdmMppiWaypointExecutor,
    start_pose: Pose2D,
    commands: list[tuple[float, float, float]],
) -> tuple[dict, Pose2D, bool]:
    with torch.inference_mode():
        robot.reset(start_pose)
    executor.reset()
    first_vx, first_vy, _ = commands[0]
    target_vx = max(0.05, min(0.35, abs(first_vx)))
    target_vy = max(-0.04, min(0.04, first_vy))
    prelude_steps = max(1, int(round(args_cli.actual_prelude_s / robot.env.step_dt)))
    reset = False
    obs = executor._collect_observations()
    for step_idx in range(prelude_steps):
        phase = min(1.0, (step_idx + 1) / max(1, prelude_steps))
        command = VelocityCommand(target_vx * phase, target_vy * phase, 0.0)
        executor._update_obs_buffers(
            state=obs["fdm_state"].clone(),
            proprio=obs["fdm_obs_proprioception"].clone(),
        )
        robot.step_velocity(command)
        obs = executor._collect_observations()
        reset = robot.consume_reset_event() or reset
    executor._update_obs_buffers(
        state=obs["fdm_state"].clone(),
        proprio=obs["fdm_obs_proprioception"].clone(),
    )
    return obs, robot.pose(), reset


def _fdm_rollout(
    executor: FdmMppiWaypointExecutor,
    obs: dict,
    start_pose: Pose2D,
    commands: list[tuple[float, float, float]],
    *,
    state_history: torch.Tensor | None = None,
    proprio_history: torch.Tensor | None = None,
) -> tuple[np.ndarray, bool]:
    device = executor.device
    actions = torch.tensor(commands, device=device, dtype=torch.float32).view(1, 1, len(commands), 3)
    planner_obs = obs["planner_obs"]
    planner_obs["start"] = torch.tensor([[start_pose.x, start_pose.y, start_pose.yaw]], device=device, dtype=torch.float32)
    planner_obs["goal"] = torch.tensor([[start_pose.x, start_pose.y, start_pose.yaw]], device=device, dtype=torch.float32)
    planner_obs["resample_population"] = torch.zeros(executor.env.num_envs, device=device, dtype=torch.bool)
    planner_obs["states"] = (executor._state_history if state_history is None else state_history).clone()
    planner_obs["proprio_obs"] = (executor._proprio_obs_history if proprio_history is None else proprio_history).clone()
    planner_obs["extero_obs"] = obs["fdm_obs_exteroceptive"].clone()
    executor.planner.obs = planner_obs
    executor.planner.env_ids = executor._env_ids
    with torch.inference_mode():
        states = executor.planner.b_obj_func_N_step(
            actions,
            only_rollout=True,
            control_mode="fdm",
            env_ids=executor._env_ids,
        )
    return states[0, 0].detach().cpu().numpy()


def _run_full_tape(
    *,
    robot: LabRobotAdapter,
    executor: FdmMppiWaypointExecutor,
    start_pose: Pose2D,
    commands: list[tuple[float, float, float]],
    command_dt: float,
    horizon: int,
    windows: list[tuple[str, int, int]],
) -> dict[str, list]:
    with torch.inference_mode():
        robot.reset(start_pose)
    executor.reset()
    sample_steps = max(1, int(round(command_dt / robot.env.step_dt)))
    poses: list[Pose2D] = [robot.pose()]
    reset_flags: list[bool] = []
    obs_snapshots: dict[int, dict] = {}
    state_history_snapshots: dict[int, torch.Tensor] = {}
    proprio_history_snapshots: dict[int, torch.Tensor] = {}
    snapshot_indices = {start for _, start, end in windows}
    snapshot_indices.update(end for _, start, end in windows)
    obs = executor._collect_observations()

    for command_idx, (cmd_vx, cmd_vy, cmd_wz) in enumerate(commands):
        if command_idx in snapshot_indices:
            obs_snapshots[command_idx] = _clone_obs(obs)
            state_history_snapshots[command_idx] = executor._state_history.clone()
            proprio_history_snapshots[command_idx] = executor._proprio_obs_history.clone()

        command = VelocityCommand(cmd_vx, cmd_vy, cmd_wz)
        interval_reset = False
        for _ in range(sample_steps):
            executor._update_obs_buffers(
                state=obs["fdm_state"].clone(),
                proprio=obs["fdm_obs_proprioception"].clone(),
            )
            robot.step_velocity(command)
            obs = executor._collect_observations()
            interval_reset = robot.consume_reset_event() or interval_reset
        reset_flags.append(interval_reset)
        poses.append(robot.pose())

    final_index = len(commands)
    if final_index in snapshot_indices:
        obs_snapshots[final_index] = _clone_obs(obs)
        state_history_snapshots[final_index] = executor._state_history.clone()
        proprio_history_snapshots[final_index] = executor._proprio_obs_history.clone()
    return {
        "poses": poses,
        "reset_flags": reset_flags,
        "obs_snapshots": obs_snapshots,
        "state_history_snapshots": state_history_snapshots,
        "proprio_history_snapshots": proprio_history_snapshots,
    }


def _run_csv_step_replay(
    *,
    robot: LabRobotAdapter,
    executor: FdmMppiWaypointExecutor,
    start_pose: Pose2D,
    commands: list[tuple[float, float, float]],
    step_dt: float,
    snapshot_rows: list[int],
) -> dict[str, list]:
    with torch.inference_mode():
        robot.reset(start_pose)
    executor.reset()
    poses: list[Pose2D] = [robot.pose()]
    reset_flags: list[bool] = []
    obs_snapshots: dict[int, dict] = {}
    state_history_snapshots: dict[int, torch.Tensor] = {}
    proprio_history_snapshots: dict[int, torch.Tensor] = {}
    snapshot_indices = set(snapshot_rows)
    obs = executor._collect_observations()

    for row_idx, (cmd_vx, cmd_vy, cmd_wz) in enumerate(commands):
        if row_idx in snapshot_indices:
            obs_snapshots[row_idx] = _clone_obs(obs)
            state_history_snapshots[row_idx] = executor._state_history.clone()
            proprio_history_snapshots[row_idx] = executor._proprio_obs_history.clone()

        executor._update_obs_buffers(
            state=obs["fdm_state"].clone(),
            proprio=obs["fdm_obs_proprioception"].clone(),
        )
        robot.step_velocity(VelocityCommand(cmd_vx, cmd_vy, cmd_wz))
        obs = executor._collect_observations()
        reset_flags.append(robot.consume_reset_event())
        poses.append(robot.pose())

    return {
        "poses": poses,
        "reset_flags": reset_flags,
        "obs_snapshots": obs_snapshots,
        "state_history_snapshots": state_history_snapshots,
        "proprio_history_snapshots": proprio_history_snapshots,
    }


def _clone_obs(obs: dict) -> dict:
    result = {}
    for key, value in obs.items():
        if isinstance(value, dict):
            result[key] = _clone_obs(value)
        elif torch.is_tensor(value):
            result[key] = value.clone()
        else:
            result[key] = value
    return result


def _relative_segment_states(
    start_pose: Pose2D,
    base_pose: Pose2D,
    poses: list[Pose2D],
) -> np.ndarray:
    return np.asarray([_align_pose_delta(start_pose, base_pose, pose) for pose in poses], dtype=np.float64)


def _full_tape_rows(
    commands: list[tuple[float, float, float]],
    tape_result: dict[str, list],
    command_dt: float,
) -> list[dict[str, float | int | bool]]:
    rows: list[dict[str, float | int | bool]] = []
    poses = tape_result["poses"]
    reset_flags = tape_result["reset_flags"]
    for idx, (cmd_vx, cmd_vy, cmd_wz) in enumerate(commands):
        pose = poses[idx + 1]
        rows.append(
            {
                "idx": idx,
                "time_s": (idx + 1) * command_dt,
                "cmd_vx": cmd_vx,
                "cmd_vy": cmd_vy,
                "cmd_wz": cmd_wz,
                "x": pose.x,
                "y": pose.y,
                "yaw": pose.yaw,
                "reset": bool(reset_flags[idx]),
            }
        )
    return rows


def _plot_full_tape(rows: list[dict[str, float | int | bool]], path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    idx = np.array([int(row["idx"]) for row in rows])
    x = np.array([float(row["x"]) for row in rows])
    y = np.array([float(row["y"]) for row in rows])
    yaw = np.array([float(row["yaw"]) for row in rows])
    cmd_vx = np.array([float(row["cmd_vx"]) for row in rows])
    cmd_vy = np.array([float(row["cmd_vy"]) for row in rows])
    cmd_wz = np.array([float(row["cmd_wz"]) for row in rows])
    reset_idx = [int(row["idx"]) for row in rows if bool(row["reset"])]

    axes[0].plot(x, y, "-o", markersize=3, color="#1b9e77", label="actual full tape")
    for name, start, end in FULL_TAPE_WINDOWS:
        seg_x = x[start:end]
        seg_y = y[start:end]
        if len(seg_x):
            axes[0].plot(seg_x, seg_y, linewidth=4, alpha=0.45, label=name)
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].grid(True, alpha=0.25)
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].legend(loc="best")

    axes[1].plot(idx, cmd_vx, label="cmd_vx", color="#4daf4a")
    axes[1].plot(idx, cmd_vy, label="cmd_vy", color="#984ea3")
    axes[1].plot(idx, cmd_wz, label="cmd_wz", color="#ff7f00")
    axes[1].plot(idx, yaw, label="actual yaw", color="#377eb8", alpha=0.7)
    for reset in reset_idx:
        axes[1].axvline(reset, color="#d62728", linewidth=1.5, alpha=0.7)
    for _, start, end in FULL_TAPE_WINDOWS:
        axes[1].axvspan(start, end, color="#eeeeee", alpha=0.45)
    axes[1].grid(True, alpha=0.25)
    axes[1].set_xlabel("0.5s command index")
    axes[1].set_ylabel("m/s or rad/s or rad")
    axes[1].legend(loc="best")
    fig.suptitle("Full 0.5s Command Tape Actual Execution")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _actual_rollout(
    robot: LabRobotAdapter,
    start_pose: Pose2D,
    base_pose: Pose2D,
    commands: list[tuple[float, float, float]],
    command_dt: float,
) -> tuple[np.ndarray, bool]:
    sample_steps = max(1, int(round(command_dt / robot.env.step_dt)))
    states: list[tuple[float, float, float]] = []
    reset = False
    for cmd_vx, cmd_vy, cmd_wz in commands:
        command = VelocityCommand(cmd_vx, cmd_vy, cmd_wz)
        for _ in range(sample_steps):
            robot.step_velocity(command)
        pose = robot.pose()
        states.append(_align_pose_delta(start_pose, base_pose, pose))
        if robot.consume_reset_event():
            reset = True
            break
    while len(states) < len(commands):
        states.append(states[-1])
    return np.asarray(states, dtype=np.float64), reset


def _align_pose_delta(start_pose: Pose2D, base_pose: Pose2D, pose: Pose2D) -> tuple[float, float, float]:
    dx = pose.x - base_pose.x
    dy = pose.y - base_pose.y
    cb = cos(base_pose.yaw)
    sb = sin(base_pose.yaw)
    dx_body = cb * dx + sb * dy
    dy_body = -sb * dx + cb * dy
    c0 = cos(start_pose.yaw)
    s0 = sin(start_pose.yaw)
    x = start_pose.x + c0 * dx_body - s0 * dy_body
    y = start_pose.y + s0 * dx_body + c0 * dy_body
    yaw = _wrap_to_pi(start_pose.yaw + _wrap_to_pi(pose.yaw - base_pose.yaw))
    return x, y, yaw


def _ideal_rollout(start_pose: Pose2D, commands: list[tuple[float, float, float]], dt: float) -> np.ndarray:
    local_x = 0.0
    local_y = 0.0
    local_yaw = 0.0
    states: list[tuple[float, float, float]] = []
    for vx, vy, wz in commands:
        c_prev = cos(local_yaw)
        s_prev = sin(local_yaw)
        local_x += c_prev * vx * dt - s_prev * vy * dt
        local_y += s_prev * vx * dt + c_prev * vy * dt
        local_yaw = _wrap_to_pi(local_yaw + wz * dt)
        c0 = cos(start_pose.yaw)
        s0 = sin(start_pose.yaw)
        world_x = start_pose.x + c0 * local_x - s0 * local_y
        world_y = start_pose.y + s0 * local_x + c0 * local_y
        states.append((world_x, world_y, _wrap_to_pi(start_pose.yaw + local_yaw)))
    return np.asarray(states, dtype=np.float64)


def _append_rows(
    step_rows: list[dict[str, float | int | str]],
    summary_rows: list[dict[str, float | int | str]],
    window_name: str,
    commands: list[tuple[float, float, float]],
    ideal: np.ndarray,
    fdm: np.ndarray,
    actual: np.ndarray,
    *,
    actual_reset: bool,
    prelude_reset: bool,
    dt: float,
) -> None:
    for idx, ((cmd_vx, cmd_vy, cmd_wz), ideal_state, fdm_state, actual_state) in enumerate(
        zip(commands, ideal, fdm, actual),
        start=1,
    ):
        fdm_yaw_err = _wrap_to_pi(float(fdm_state[2]) - float(ideal_state[2]))
        actual_yaw_err = _wrap_to_pi(float(actual_state[2]) - float(ideal_state[2]))
        fdm_actual_yaw_err = _wrap_to_pi(float(fdm_state[2]) - float(actual_state[2]))
        step_rows.append(
            {
                "window": window_name,
                "sample": idx,
                "time_s": idx * dt,
                "cmd_vx": cmd_vx,
                "cmd_vy": cmd_vy,
                "cmd_wz": cmd_wz,
                "ideal_x": float(ideal_state[0]),
                "ideal_y": float(ideal_state[1]),
                "ideal_yaw": float(ideal_state[2]),
                "fdm_x": float(fdm_state[0]),
                "fdm_y": float(fdm_state[1]),
                "fdm_yaw": float(fdm_state[2]),
                "actual_x": float(actual_state[0]),
                "actual_y": float(actual_state[1]),
                "actual_yaw": float(actual_state[2]),
                "fdm_minus_ideal_x": float(fdm_state[0] - ideal_state[0]),
                "fdm_minus_ideal_y": float(fdm_state[1] - ideal_state[1]),
                "fdm_minus_ideal_yaw": fdm_yaw_err,
                "fdm_minus_ideal_yaw_deg": degrees(fdm_yaw_err),
                "actual_minus_ideal_x": float(actual_state[0] - ideal_state[0]),
                "actual_minus_ideal_y": float(actual_state[1] - ideal_state[1]),
                "actual_minus_ideal_yaw": actual_yaw_err,
                "actual_minus_ideal_yaw_deg": degrees(actual_yaw_err),
                "fdm_minus_actual_x": float(fdm_state[0] - actual_state[0]),
                "fdm_minus_actual_y": float(fdm_state[1] - actual_state[1]),
                "fdm_minus_actual_yaw": fdm_actual_yaw_err,
                "fdm_minus_actual_yaw_deg": degrees(fdm_actual_yaw_err),
                "prelude_reset": prelude_reset,
                "actual_reset": actual_reset,
            }
        )
    final_ideal = ideal[-1]
    final_fdm = fdm[-1]
    final_actual = actual[-1]
    final_yaw_err = _wrap_to_pi(float(final_fdm[2]) - float(final_ideal[2]))
    final_actual_yaw_err = _wrap_to_pi(float(final_actual[2]) - float(final_ideal[2]))
    final_fdm_actual_yaw_err = _wrap_to_pi(float(final_fdm[2]) - float(final_actual[2]))
    summary_rows.append(
        {
            "window": window_name,
            "samples": len(commands),
            "duration_s": len(commands) * dt,
            "mean_cmd_vx": float(np.mean([cmd[0] for cmd in commands])),
            "mean_cmd_vy": float(np.mean([cmd[1] for cmd in commands])),
            "mean_cmd_wz": float(np.mean([cmd[2] for cmd in commands])),
            "ideal_final_x": float(final_ideal[0]),
            "ideal_final_y": float(final_ideal[1]),
            "ideal_final_yaw_deg": degrees(float(final_ideal[2])),
            "fdm_final_x": float(final_fdm[0]),
            "fdm_final_y": float(final_fdm[1]),
            "fdm_final_yaw_deg": degrees(float(final_fdm[2])),
            "actual_final_x": float(final_actual[0]),
            "actual_final_y": float(final_actual[1]),
            "actual_final_yaw_deg": degrees(float(final_actual[2])),
            "fdm_minus_ideal_x": float(final_fdm[0] - final_ideal[0]),
            "fdm_minus_ideal_y": float(final_fdm[1] - final_ideal[1]),
            "fdm_minus_ideal_yaw_deg": degrees(final_yaw_err),
            "actual_minus_ideal_x": float(final_actual[0] - final_ideal[0]),
            "actual_minus_ideal_y": float(final_actual[1] - final_ideal[1]),
            "actual_minus_ideal_yaw_deg": degrees(final_actual_yaw_err),
            "fdm_minus_actual_x": float(final_fdm[0] - final_actual[0]),
            "fdm_minus_actual_y": float(final_fdm[1] - final_actual[1]),
            "fdm_minus_actual_yaw_deg": degrees(final_fdm_actual_yaw_err),
            "prelude_reset": prelude_reset,
            "actual_reset": actual_reset,
        }
    )


def _plot_results(
    step_rows: list[dict[str, float | int | str]],
    summary_rows: list[dict[str, float | int | str]],
    path: Path,
    source_label: str,
) -> None:
    windows = [str(row["window"]) for row in summary_rows]
    colors = {"ideal": "#444444", "fdm": "#d95f02", "actual": "#1b9e77"}
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0])

    for col, window in enumerate(windows):
        ax = fig.add_subplot(gs[0, col])
        rows = [row for row in step_rows if row["window"] == window]
        ideal_x = [float(row["ideal_x"]) for row in rows]
        ideal_y = [float(row["ideal_y"]) for row in rows]
        fdm_x = [float(row["fdm_x"]) for row in rows]
        fdm_y = [float(row["fdm_y"]) for row in rows]
        actual_x = [float(row["actual_x"]) for row in rows]
        actual_y = [float(row["actual_y"]) for row in rows]
        ax.plot(ideal_x, ideal_y, "-o", color=colors["ideal"], markersize=3, label="ideal SE(2)")
        ax.plot(fdm_x, fdm_y, "-o", color=colors["fdm"], markersize=3, label="FDM prediction")
        ax.plot(actual_x, actual_y, "-o", color=colors["actual"], markersize=3, label="actual Lab")
        ax.scatter([ideal_x[-1]], [ideal_y[-1]], color=colors["ideal"], s=55)
        ax.scatter([fdm_x[-1]], [fdm_y[-1]], color=colors["fdm"], s=55)
        ax.scatter([actual_x[-1]], [actual_y[-1]], color=colors["actual"], s=55)
        ax.set_title(window)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("x [m]")
        if col == 0:
            ax.set_ylabel("y [m]")
            ax.legend(loc="best")

    ax_pos = fig.add_subplot(gs[1, :2])
    y = np.arange(len(summary_rows))
    err_x = [float(row["fdm_minus_ideal_x"]) for row in summary_rows]
    err_y = [float(row["fdm_minus_ideal_y"]) for row in summary_rows]
    actual_err_x = [float(row["actual_minus_ideal_x"]) for row in summary_rows]
    actual_err_y = [float(row["actual_minus_ideal_y"]) for row in summary_rows]
    height = 0.18
    ax_pos.barh(y - height * 1.5, err_x, height, label="FDM x", color="#377eb8")
    ax_pos.barh(y - height * 0.5, err_y, height, label="FDM y", color="#4daf4a")
    ax_pos.barh(y + height * 0.5, actual_err_x, height, label="actual x", color="#80b1d3")
    ax_pos.barh(y + height * 1.5, actual_err_y, height, label="actual y", color="#b3de69")
    ax_pos.axvline(0.0, color="black", linewidth=1)
    ax_pos.set_yticks(y, windows)
    ax_pos.set_xlabel("FDM - ideal final position [m]")
    ax_pos.grid(True, axis="x", alpha=0.25)
    ax_pos.legend()

    ax_yaw = fig.add_subplot(gs[1, 2])
    yaw_err = [float(row["fdm_minus_ideal_yaw_deg"]) for row in summary_rows]
    actual_yaw_err = [float(row["actual_minus_ideal_yaw_deg"]) for row in summary_rows]
    ax_yaw.barh(y - 0.15, yaw_err, 0.28, color="#984ea3", label="FDM")
    ax_yaw.barh(y + 0.15, actual_yaw_err, 0.28, color="#cab2d6", label="actual")
    ax_yaw.axvline(0.0, color="black", linewidth=1)
    ax_yaw.set_yticks(y, windows)
    ax_yaw.set_xlabel("FDM - ideal final yaw [deg]")
    ax_yaw.grid(True, axis="x", alpha=0.25)
    ax_yaw.legend()

    fig.suptitle("Ideal vs FDM Prediction vs Actual Lab Execution\n" + source_label, fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_full_snapshot_prediction(
    rows: list[dict[str, float | int | str]],
    path: Path,
    source_label: str,
) -> None:
    if not rows:
        raise ValueError("No full snapshot rows to plot")
    step = np.asarray([int(row["step"]) for row in rows])
    actual_x = np.asarray([float(row["actual_x"]) for row in rows])
    actual_y = np.asarray([float(row["actual_y"]) for row in rows])
    ideal_x = np.asarray([float(row["one_step_ideal_x"]) for row in rows])
    ideal_y = np.asarray([float(row["one_step_ideal_y"]) for row in rows])
    fdm_x = np.asarray([float(row["fdm_next_x"]) for row in rows])
    fdm_y = np.asarray([float(row["fdm_next_y"]) for row in rows])
    cmd_vx = np.asarray([float(row["cmd_vx"]) for row in rows])
    cmd_vy = np.asarray([float(row["cmd_vy"]) for row in rows])
    cmd_wz = np.asarray([float(row["cmd_wz"]) for row in rows])
    fdm_actual_x = np.asarray([float(row["fdm_minus_actual_next_x"]) for row in rows])
    fdm_actual_y = np.asarray([float(row["fdm_minus_actual_next_y"]) for row in rows])
    fdm_actual_yaw = np.asarray([float(row["fdm_minus_actual_next_yaw_deg"]) for row in rows])
    fdm_ideal_x = np.asarray([float(row["fdm_minus_one_step_ideal_x"]) for row in rows])
    fdm_ideal_y = np.asarray([float(row["fdm_minus_one_step_ideal_y"]) for row in rows])
    fdm_ideal_yaw = np.asarray([float(row["fdm_minus_one_step_ideal_yaw_deg"]) for row in rows])

    active = [str(row["active"]) for row in rows]
    change_indices = [0]
    for idx in range(1, len(active)):
        if active[idx] != active[idx - 1]:
            change_indices.append(idx)
    change_indices.append(len(active) - 1)

    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.45, 1.0, 1.0])
    ax_map = fig.add_subplot(gs[0, :])
    ax_cmd = fig.add_subplot(gs[1, 0])
    ax_err = fig.add_subplot(gs[1, 1])
    ax_yaw = fig.add_subplot(gs[2, 0])
    ax_norm = fig.add_subplot(gs[2, 1])

    ax_map.plot(ideal_x, ideal_y, color="#444444", linewidth=2.0, label="ideal one-step from each snapshot")
    ax_map.plot(fdm_x, fdm_y, color="#d95f02", linewidth=2.0, marker="o", markersize=2.8, label="FDM one-step from each snapshot")
    ax_map.plot(actual_x, actual_y, color="#1b9e77", linewidth=2.2, label="actual next pose, sampled every 0.5s")
    ax_map.scatter([actual_x[0]], [actual_y[0]], s=90, color="#1565c0", label=f"target-nav start step {step[0]}")
    ax_map.scatter([actual_x[-1]], [actual_y[-1]], s=90, color="#d62728", label=f"end step {step[-1]}")
    for idx in change_indices[1:-1]:
        label = active[idx].split(":")[-1]
        ax_map.scatter([actual_x[idx]], [actual_y[idx]], s=55, color="#ffbf00", edgecolor="black", zorder=5)
        ax_map.text(actual_x[idx], actual_y[idx] + 0.08, label, fontsize=8, color="#7a5200")
    ax_map.set_aspect("equal", adjustable="datalim")
    ax_map.grid(True, alpha=0.25)
    ax_map.set_xlabel("x [m]")
    ax_map.set_ylabel("y [m]")
    ax_map.legend(loc="best")
    ax_map.set_title("Full Snapshot Prediction Timeline")

    ax_cmd.plot(step, cmd_vx, label="cmd_vx", color="#4daf4a")
    ax_cmd.plot(step, cmd_vy, label="cmd_vy", color="#984ea3")
    ax_cmd.plot(step, cmd_wz, label="cmd_wz", color="#ff7f00")
    ax_cmd.axhline(0.0, color="black", linewidth=0.8)
    ax_cmd.grid(True, alpha=0.25)
    ax_cmd.set_xlabel("CSV step")
    ax_cmd.set_ylabel("m/s or rad/s")
    ax_cmd.legend(loc="best")

    ax_err.plot(step, fdm_actual_x, label="FDM - actual next x", color="#377eb8")
    ax_err.plot(step, fdm_actual_y, label="FDM - actual next y", color="#4daf4a")
    ax_err.plot(step, fdm_ideal_x, "--", label="FDM - ideal next x", color="#80b1d3")
    ax_err.plot(step, fdm_ideal_y, "--", label="FDM - ideal next y", color="#b3de69")
    ax_err.axhline(0.0, color="black", linewidth=0.8)
    ax_err.grid(True, alpha=0.25)
    ax_err.set_xlabel("CSV step")
    ax_err.set_ylabel("one-step error [m]")
    ax_err.legend(loc="best")

    ax_yaw.plot(step, fdm_actual_yaw, label="FDM - actual next yaw", color="#e41a1c")
    ax_yaw.plot(step, fdm_ideal_yaw, "--", label="FDM - ideal next yaw", color="#984ea3")
    ax_yaw.axhline(0.0, color="black", linewidth=0.8)
    ax_yaw.grid(True, alpha=0.25)
    ax_yaw.set_xlabel("CSV step")
    ax_yaw.set_ylabel("one-step yaw error [deg]")
    ax_yaw.legend(loc="best")

    actual_norm = np.sqrt(fdm_actual_x**2 + fdm_actual_y**2)
    ideal_norm = np.sqrt(fdm_ideal_x**2 + fdm_ideal_y**2)
    ax_norm.plot(step, actual_norm, label="|FDM - actual next xy|", color="#1b9e77")
    ax_norm.plot(step, ideal_norm, label="|FDM - ideal next xy|", color="#d95f02")
    ax_norm.grid(True, alpha=0.25)
    ax_norm.set_xlabel("CSV step")
    ax_norm.set_ylabel("one-step XY error norm [m]")
    ax_norm.legend(loc="best")

    fig.suptitle("Full One-Step Ideal vs FDM Snapshot Prediction vs Actual Lab Execution\n" + source_label, fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _wrap_to_pi(angle: float) -> float:
    return atan2(sin(angle), cos(angle))


def _status(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(message + "\n")
    print(f"[fdm_prediction_probe] {message}", flush=True)


if __name__ == "__main__":
    main()
