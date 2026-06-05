from __future__ import annotations

"""Probe Lab G1 velocity tracking for fixed high-level commands."""

import argparse
import csv
import os
from math import cos, sin
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher

SEMANTIC_NAV_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = SEMANTIC_NAV_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
if str(SEMANTIC_NAV_ROOT) not in sys.path:
    sys.path.insert(0, str(SEMANTIC_NAV_ROOT))

import utils.cli_args as cli_args  # isort: skip


DEFAULT_COMMANDS: tuple[tuple[float, float, float], ...] = (
    (1.0, 0.1, 0.66),
    (0.5, 0.1, 0.66),
    (1.0, 0.1, 0.33),
    (0.5, 0.1, 0.33),
    (1.0, 0.1, 0.0),
    (0.5, 0.1, 0.0),
    (1.0, 0.0, 0.66),
    (1.0, 0.0, 0.33),
)


parser = argparse.ArgumentParser(description="Measure Lab G1 fixed velocity-command tracking.")
parser.add_argument("--duration-s", type=float, default=5.0)
parser.add_argument("--sample-period-s", type=float, default=0.5)
parser.add_argument("--settle-s", type=float, default=0.0, help="Optional zero-command settling after reset.")
parser.add_argument("--out-dir", type=Path, default=Path(r"D:\semantic_nav_run\velocity_tracking_probe"))
parser.add_argument("--start-pose", type=float, nargs=3, default=(0.0, 0.0, 0.0))
parser.add_argument("--episode-length-s", type=float, default=20.0)
parser.add_argument("--commands", type=float, nargs="*", default=None, help="Flat vx vy wz triples. Defaults to the requested set.")
parser.add_argument("--low-level-policy-file", type=Path, default=None, help="Override the G1 low-level policy path.")
parser.add_argument("--low-level-policy-mode", choices=("single", "dwaq"), default="single")
parser.add_argument("--low-level-obs-dim", type=int, default=None, help="Single-frame obs dim for multi-input policies.")
parser.add_argument("--low-level-obs-history", type=int, default=5)
parser.add_argument("--dwaq-align-deploy-robot", action="store_true", help="Use DWAQ deploy default pose and PD gains.")
parser.add_argument("--dwaq-clip-deploy-command", action="store_true", help="Clip commands to the original DWAQ deploy range.")
parser.add_argument("--dwaq-gait-phase-layout", choices=("deploy", "train"), default="deploy")
parser.add_argument("--warp-cache-path", type=Path, default=None, help="Optional per-run Warp kernel cache path.")
parser.add_argument("--force-exit", action="store_true", help="Exit immediately after writing outputs to avoid Isaac shutdown hangs.")
cli_args.add_fdm_args(parser, default_num_envs=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.warp_cache_path is not None:
    args_cli.warp_cache_path.mkdir(parents=True, exist_ok=True)
    os.environ["WARP_CACHE_PATH"] = str(args_cli.warp_cache_path)

args_cli.num_envs = 1
args_cli.robot = "g1"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.terrains as terrain_gen  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import ObservationTermCfg as ObsTerm  # noqa: E402

import fdm.mdp as mdp  # noqa: E402
from fdm.env_cfg.robot_cfg_g1 import G1_29DOF_JOINT_NAMES  # noqa: E402
from fdm.utils.args_cli_utils import cfg_modifier_pre_init, planner_cfg_init  # noqa: E402
from executors.lab_robot_adapter import LabRobotAdapter  # noqa: E402
from executors.robot_adapter import VelocityCommand  # noqa: E402
from maps.semantic_graph import Pose2D  # noqa: E402


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    commands = _parse_commands(args_cli.commands)
    args_cli.out_dir.mkdir(parents=True, exist_ok=True)
    samples_csv = args_cli.out_dir / "velocity_tracking_samples.csv"
    summary_csv = args_cli.out_dir / "velocity_tracking_summary.csv"
    plot_path = args_cli.out_dir / "velocity_tracking_error.png"
    status_log = args_cli.out_dir / "probe_status.log"
    status_log.unlink(missing_ok=True)
    _status(status_log, "main_start")

    env = None
    try:
        _status(status_log, "building_cfg")
        cfg = planner_cfg_init(args_cli)
        cfg = cfg_modifier_pre_init(cfg, args_cli)
        if args_cli.low_level_policy_file is not None:
            cfg.env_cfg.actions.velocity_cmd.low_level_policy_file = str(args_cli.low_level_policy_file)
        if args_cli.low_level_policy_mode == "dwaq" and args_cli.dwaq_align_deploy_robot:
            _align_dwaq_deploy_robot(cfg.env_cfg.scene.robot)
        cfg.env_cfg.actions.velocity_cmd.low_level_policy_mode = args_cli.low_level_policy_mode
        cfg.env_cfg.actions.velocity_cmd.low_level_obs_dim = args_cli.low_level_obs_dim
        cfg.env_cfg.actions.velocity_cmd.low_level_obs_history = args_cli.low_level_obs_history
        if args_cli.low_level_policy_mode == "dwaq":
            if hasattr(cfg.env_cfg.observations.policy, "joint_vel"):
                cfg.env_cfg.observations.policy.joint_vel.scale = 1.0
            if hasattr(cfg.env_cfg.observations.policy, "joint_vel_rel"):
                cfg.env_cfg.observations.policy.joint_vel_rel.scale = 1.0
            if args_cli.dwaq_clip_deploy_command:
                cfg.env_cfg.actions.velocity_cmd.clip_mode = "minmax"
                cfg.env_cfg.actions.velocity_cmd.clip = [(-0.4, 0.7), (-0.4, 0.4), (-1.57, 1.57)]
            cfg.env_cfg.actions.velocity_cmd.low_level_obs_term_dims = [3, 3, 3, 29, 29, 29, 4]
            cfg.env_cfg.observations.policy.gait_phase = ObsTerm(
                func=mdp.dwaq_gait_phase,
                params={"period": 0.8, "offset": 0.5, "layout": args_cli.dwaq_gait_phase_layout},
            )
        cfg.env_cfg.scene.num_envs = 1
        cfg.env_cfg.episode_length_s = args_cli.episode_length_s
        cfg.env_cfg.scene.terrain.terrain_type = "generator"
        cfg.env_cfg.scene.terrain.terrain_generator = _flat_terrain_cfg()
        if hasattr(cfg.env_cfg.scene.terrain, "groundplane"):
            cfg.env_cfg.scene.terrain.groundplane = False
        cfg.env_cfg.scene.terrain.random_seed = 0
        cfg.env_cfg.events.reset_base.func = mdp.reset_root_state_center
        cfg.env_cfg.events.reset_base.params = {}
        _status(status_log, "creating_env")
        env = ManagerBasedRLEnv(cfg.env_cfg)
        _status(status_log, "env_created")
        robot = LabRobotAdapter(env)

        step_dt = float(env.step_dt)
        sample_steps = max(1, int(round(args_cli.sample_period_s / step_dt)))
        sample_dt = sample_steps * step_dt
        sample_count = max(1, int(round(args_cli.duration_s / sample_dt)))
        settle_steps = max(0, int(round(args_cli.settle_s / step_dt)))
        print(
            f"[velocity_probe] step_dt={step_dt:.5f}s sample_steps={sample_steps} "
            f"sample_dt={sample_dt:.3f}s samples={sample_count}",
            flush=True,
        )
        _status(
            status_log,
            f"timing step_dt={step_dt:.5f} sample_steps={sample_steps} sample_dt={sample_dt:.3f} samples={sample_count}",
        )

        sample_rows: list[dict[str, float | int | str | bool]] = []
        summary_rows: list[dict[str, float | int | str]] = []
        start_pose = Pose2D(args_cli.start_pose[0], args_cli.start_pose[1], args_cli.start_pose[2])
        for command_idx, (cmd_vx, cmd_vy, cmd_wz) in enumerate(commands):
            _status(status_log, f"command_start idx={command_idx} cmd=({cmd_vx},{cmd_vy},{cmd_wz})")
            command = VelocityCommand(cmd_vx, cmd_vy, cmd_wz)
            robot.reset(start_pose)
            _status(status_log, f"command_reset idx={command_idx}")
            for _ in range(settle_steps):
                robot.step_velocity(VelocityCommand(0.0, 0.0, 0.0))

            command_rows: list[dict[str, float | int | str | bool]] = []
            for sample_idx in range(sample_count):
                pose0 = robot.pose()
                for _ in range(sample_steps):
                    robot.step_velocity(command)
                pose1 = robot.pose()
                actual_vx, actual_vy, actual_wz = _interval_body_velocity(pose0, pose1, sample_dt)
                reset = robot.consume_reset_event()
                row = {
                    "command_idx": command_idx,
                    "sample_idx": sample_idx,
                    "time_s": (sample_idx + 1) * sample_dt,
                    "cmd_vx": cmd_vx,
                    "cmd_vy": cmd_vy,
                    "cmd_wz": cmd_wz,
                    "actual_vx": actual_vx,
                    "actual_vy": actual_vy,
                    "actual_wz": actual_wz,
                    "err_vx": actual_vx - cmd_vx,
                    "err_vy": actual_vy - cmd_vy,
                    "err_wz": actual_wz - cmd_wz,
                    "start_x": pose0.x,
                    "start_y": pose0.y,
                    "start_yaw": pose0.yaw,
                    "end_x": pose1.x,
                    "end_y": pose1.y,
                    "end_yaw": pose1.yaw,
                    "reset": reset,
                    "reset_reason": robot.last_reset_reason(),
                }
                sample_rows.append(row)
                command_rows.append(row)
                print(
                    f"[velocity_probe] cmd={command_idx} sample={sample_idx + 1}/{sample_count} "
                    f"cmd=({cmd_vx:.2f},{cmd_vy:.2f},{cmd_wz:.2f}) "
                    f"actual=({actual_vx:.3f},{actual_vy:.3f},{actual_wz:.3f}) "
                    f"err=({row['err_vx']:.3f},{row['err_vy']:.3f},{row['err_wz']:.3f}) reset={reset}",
                    flush=True,
                )
                if reset:
                    _status(status_log, f"command_reset_event idx={command_idx} sample={sample_idx} reason={robot.last_reset_reason()}")
                    break

            summary_rows.append(_summarize_command(command_idx, command_rows))
            _status(status_log, f"command_done idx={command_idx} samples={len(command_rows)}")

        _status(status_log, f"writing_outputs samples={len(sample_rows)} summaries={len(summary_rows)}")
        _write_csv(samples_csv, sample_rows)
        _write_csv(summary_csv, summary_rows)
        _plot_results(sample_rows, summary_rows, plot_path)
        print(f"[velocity_probe] wrote samples={samples_csv}", flush=True)
        print(f"[velocity_probe] wrote summary={summary_csv}", flush=True)
        print(f"[velocity_probe] wrote plot={plot_path}", flush=True)
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


def _parse_commands(values: list[float] | None) -> tuple[tuple[float, float, float], ...]:
    if values is None:
        return DEFAULT_COMMANDS
    if len(values) % 3 != 0:
        raise ValueError("--commands must be a flat list of vx vy wz triples")
    return tuple((float(values[idx]), float(values[idx + 1]), float(values[idx + 2])) for idx in range(0, len(values), 3))


def _flat_terrain_cfg():
    return terrain_gen.TerrainGeneratorCfg(
        size=(10.0, 10.0),
        border_width=1.0,
        border_height=0.0,
        num_rows=1,
        num_cols=1,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=False,
        sub_terrains={
            "flat_reference": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=1.0,
                noise_range=(0.0, 0.0),
                noise_step=0.005,
                border_width=0.25,
                horizontal_scale=0.1,
                vertical_scale=0.005,
                downsampled_scale=0.1,
            )
        },
    )


DWAQ_DEPLOY_DEFAULT_JOINT_POS = (
    -0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.42, 0.42, 0.35, 0.35, -0.23, -0.23, 0.18, -0.18, 0.0, 0.0,
    0.0, 0.0, 0.87, 0.87, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
)


def _align_dwaq_deploy_robot(robot_cfg) -> None:
    """Match the original G1DWAQ_Lab deploy pose and PD gains."""
    robot_cfg.init_state.joint_pos = {
        joint_name: joint_pos for joint_name, joint_pos in zip(G1_29DOF_JOINT_NAMES, DWAQ_DEPLOY_DEFAULT_JOINT_POS)
    }
    robot_cfg.actuators = {
        "dwaq_deploy_legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                ".*waist.*",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
                ".*waist_yaw_joint": 88.0,
                ".*waist_roll_joint": 35.0,
                ".*waist_pitch_joint": 35.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
                ".*waist_yaw_joint": 32.0,
                ".*waist_roll_joint": 30.0,
                ".*waist_pitch_joint": 30.0,
            },
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
                ".*waist.*": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
                ".*waist.*": 5.0,
            },
            armature=0.01,
        ),
        "dwaq_deploy_feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim={
                ".*_ankle_pitch_joint": 35.0,
                ".*_ankle_roll_joint": 35.0,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": 30.0,
                ".*_ankle_roll_joint": 30.0,
            },
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
        "dwaq_deploy_shoulders": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
            },
            stiffness=100.0,
            damping=2.0,
            armature=0.01,
        ),
        "dwaq_deploy_arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
            },
            velocity_limit_sim={
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
            },
            stiffness=50.0,
            damping=2.0,
            armature=0.01,
        ),
        "dwaq_deploy_wrist": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_wrist_.*",
            ],
            effort_limit_sim={
                ".*_wrist_yaw_joint": 5.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_wrist_yaw_joint": 22.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
            },
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
        ),
    }


def _interval_body_velocity(pose0: Pose2D, pose1: Pose2D, dt: float) -> tuple[float, float, float]:
    dx = pose1.x - pose0.x
    dy = pose1.y - pose0.y
    c = cos(pose0.yaw)
    s = sin(pose0.yaw)
    vx_body = (c * dx + s * dy) / dt
    vy_body = (-s * dx + c * dy) / dt
    wz = _wrap_to_pi(pose1.yaw - pose0.yaw) / dt
    return vx_body, vy_body, wz


def _summarize_command(command_idx: int, rows: list[dict[str, float | int | str | bool]]) -> dict[str, float | int | str]:
    if not rows:
        return {"command_idx": command_idx, "samples": 0}
    result: dict[str, float | int | str] = {
        "command_idx": command_idx,
        "samples": len(rows),
        "cmd_vx": float(rows[0]["cmd_vx"]),
        "cmd_vy": float(rows[0]["cmd_vy"]),
        "cmd_wz": float(rows[0]["cmd_wz"]),
        "reset_count": sum(1 for row in rows if row["reset"]),
    }
    for key in ("vx", "vy", "wz"):
        actual = np.array([float(row[f"actual_{key}"]) for row in rows], dtype=np.float64)
        err = np.array([float(row[f"err_{key}"]) for row in rows], dtype=np.float64)
        result[f"mean_actual_{key}"] = float(actual.mean())
        result[f"mean_err_{key}"] = float(err.mean())
        result[f"mae_{key}"] = float(np.abs(err).mean())
        result[f"rmse_{key}"] = float(np.sqrt(np.square(err).mean()))
    return result


def _write_csv(path: Path, rows: list[dict[str, float | int | str | bool]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _status(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", errors="replace") as status_file:
        status_file.write(message + "\n")
        status_file.flush()


def _plot_results(
    sample_rows: list[dict[str, float | int | str | bool]],
    summary_rows: list[dict[str, float | int | str]],
    path: Path,
) -> None:
    if not sample_rows:
        return
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), dpi=150, sharex=False)
    labels = [f"{int(row['command_idx'])}:({row['cmd_vx']:.1f},{row['cmd_vy']:.1f},{row['cmd_wz']:.2f})" for row in summary_rows]
    x = np.arange(len(summary_rows))
    for axis_idx, key in enumerate(("vx", "vy", "wz")):
        ax_actual = axes[axis_idx, 0]
        ax_err = axes[axis_idx, 1]
        cmd = np.array([float(row[f"cmd_{key}"]) for row in summary_rows])
        actual = np.array([float(row[f"mean_actual_{key}"]) for row in summary_rows])
        mae = np.array([float(row[f"mae_{key}"]) for row in summary_rows])
        ax_actual.bar(x - 0.18, cmd, width=0.36, label="command")
        ax_actual.bar(x + 0.18, actual, width=0.36, label="actual mean")
        ax_actual.set_ylabel(key)
        ax_actual.grid(axis="y", alpha=0.25)
        ax_actual.legend()
        ax_err.bar(x, mae, color="#d55e00")
        ax_err.set_ylabel(f"{key} MAE")
        ax_err.grid(axis="y", alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
    axes[0, 0].set_title("Command vs Actual Mean")
    axes[0, 1].set_title("Mean Absolute Error")
    fig.suptitle("Lab G1 Velocity Tracking Probe, 5s per command, sampled every 0.5s")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _wrap_to_pi(angle: float) -> float:
    while angle > 3.141592653589793:
        angle -= 6.283185307179586
    while angle < -3.141592653589793:
        angle += 6.283185307179586
    return angle


if __name__ == "__main__":
    main()
