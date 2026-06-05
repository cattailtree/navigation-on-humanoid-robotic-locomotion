from __future__ import annotations

"""Short preset launcher for the Lab elevator task with SE(2) A* and pure MPPI."""

import argparse
from pathlib import Path
import subprocess
import sys


TASK_SCRIPT = Path(__file__).with_name("run_lab_elevator_task.py")
DEFAULT_OUTPUT_DIR = Path(r"D:\semantic_nav_run")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the IsaacLab elevator-search preset with pure MPPI. Unknown args are forwarded."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for camera frames and traj.csv.")
    parser.add_argument("--show-command", action="store_true", help="Print the expanded command before launching.")
    parser.add_argument("--dry-run", action="store_true", help="Print the expanded command and exit without launching IsaacLab.")
    args, passthrough = parser.parse_known_args()

    trajectory_csv = args.output_dir / "traj.csv"
    cmd = [
        sys.executable,
        str(TASK_SCRIPT),
        "--headless",
        "--enable_cameras",
        "--steps",
        "5000",
        "--episode-length-s",
        "90",
        "--print-every",
        "50",
        "--detector",
        "apexnav_gdino",
        "--blind-find-elevator",
        "--blind-floor",
        "F1",
        "--adaptive-exploration",
        "--spawn-blind-search-arena",
        "--blind-arena-center",
        "5.5",
        "0.4",
        "--blind-arena-size",
        "13.0",
        "9.2",
        "--spawn-corridor-lobby",
        "--spawn-center-pillar",
        "--grid-planner",
        "se2",
        "--se2-yaw-bins",
        "16",
        "--se2-step-distance",
        "0.6",
        "--se2-output-min-spacing",
        "4.0",
        "--se2-output-yaw-threshold",
        "0.55",
        "--perception-min-score",
        "0.55",
        "--perception-every",
        "50",
        "--motion-detection-image-dir",
        str(args.output_dir),
        "--trajectory-csv",
        str(trajectory_csv),
        "--local-executor",
        "mppi",
        "--fdm-mppi-population",
        "512",
        "--fdm-mppi-replan-every",
        "8",
        "--fdm-mppi-lookahead",
        "4.0",
        "--fdm-mppi-pass-tolerance",
        "0.5",
        "--fdm-mppi-progress-margin",
        "0.25",
        "--fdm-mppi-final-tolerance",
        "0.5",
        "--fdm-mppi-min-forward-carrot",
        "1.2",
        "--fdm-mppi-final-waypoint-handoff-distance",
        "1.0",
        "--fdm-mppi-no-face-subgoal",
        "--fdm-mppi-final-approach-distance",
        "2.0",
        "--fdm-mppi-final-approach-tolerance",
        "0.5",
        "--fdm-mppi-min-vx",
        "-0.1",
        "--fdm-mppi-max-vx",
        "1.0",
        "--fdm-mppi-max-vy",
        "0.3",
        "--fdm-mppi-max-wz",
        "0.2",
    ]
    cmd.extend(passthrough)

    if args.show_command or args.dry_run:
        print("[semantic_nav:preset] expanded command:", flush=True)
        print(" ".join(f'"{part}"' if " " in part else part for part in cmd), flush=True)
    if args.dry_run:
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
