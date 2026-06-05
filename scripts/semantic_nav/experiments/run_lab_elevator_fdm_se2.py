from __future__ import annotations

"""Short preset launcher for the Lab elevator task with SE(2) A* and FDM-MPPI."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
TASK_SCRIPT = Path(__file__).with_name("run_lab_elevator_task.py")
SEMANTIC_NAV_ROOT = TASK_SCRIPT.parents[1]
SCRIPTS_ROOT = SEMANTIC_NAV_ROOT.parent
DEFAULT_RUN_DIR = REPO_ROOT / "logs" / "fdm" / "fdm_se2_prediction_depth" / "May12_14-21-45_fdm_train"
DEFAULT_CHECKPOINT = DEFAULT_RUN_DIR / "model_collection_round_14.pth"
DEFAULT_OUTPUT_DIR = Path(r"D:\semantic_nav_run")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the IsaacLab elevator-search preset. Unknown arguments are forwarded "
            "to run_lab_elevator_task.py, so later flags override this preset."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR, help="FDM training run directory.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="FDM checkpoint path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for camera frames and traj.csv.")
    parser.add_argument("--goal", default="take the elevator to the basement target room", help="Natural-language task for the LLM parser.")
    parser.add_argument("--target", default="auto", help="Target semantic node id, or auto to let the LLM/parser choose.")
    parser.add_argument("--task-parser", default="openai_compatible", choices=("rule", "llm_http", "openai_compatible"))
    parser.add_argument("--llm-endpoint", default="http://127.0.0.1:11434/v1", help="OpenAI-compatible local LLM endpoint.")
    parser.add_argument("--llm-model", default="qwen2.5:3b", help="Local LLM model name.")
    parser.add_argument("--llm-timeout-s", type=float, default=20.0)
    parser.add_argument("--log-llm", action="store_true", help="Print raw LLM parser responses.")
    parser.add_argument("--success-radius", type=float, default=1.0, help="Final target radius in meters counted as success.")
    parser.add_argument("--record-every", type=int, default=10, help="Capture one recording frame every N sim steps.")
    parser.add_argument("--record-resolution", type=int, nargs=2, default=(640, 480), help="Recording frame width height.")
    parser.add_argument("--record-top-center", type=float, nargs=2, default=(4.8, 1.1), help="Top-down camera center in local XY.")
    parser.add_argument("--record-top-height", type=float, default=13.0, help="Top-down camera height.")
    parser.add_argument("--record", action="store_true", help="Enable robot-view and top-down recording. Disabled by default to keep memory low.")
    parser.add_argument("--no-record", action="store_true", help="Deprecated alias; recording is disabled unless --record is passed.")
    parser.add_argument("--no-video", action="store_true", help="Skip MP4 generation after the run.")
    parser.add_argument("--show-command", action="store_true", help="Print the expanded command before launching.")
    parser.add_argument("--dry-run", action="store_true", help="Print the expanded command and exit without launching IsaacLab.")
    args, passthrough = parser.parse_known_args()

    trajectory_csv = args.output_dir / "traj.csv"
    result_json = args.output_dir / "result.json"
    fdm_snapshot_out = args.output_dir / "fdm_snapshots.npz"
    run_log = args.output_dir / "run.log"
    launch_goal, launch_target, launch_task_parser = _preparse_task_before_isaac(args)
    do_record = args.record and not args.no_record

    cmd = [
        sys.executable,
        str(TASK_SCRIPT),
        "--steps",
        "5000",
        "--episode-length-s",
        "90",
        "--print-every",
        "50",
        "--detector",
        "apexnav_gdino",
        "--perception-endpoint",
        "http://127.0.0.1:12181/gdino",
        "--goal",
        launch_goal,
        "--target",
        launch_target,
        "--task-parser",
        launch_task_parser,
        "--llm-endpoint",
        args.llm_endpoint,
        "--llm-model",
        args.llm_model,
        "--llm-timeout-s",
        str(args.llm_timeout_s),
        "--blind-find-elevator",
        "--blind-floor",
        "F1",
        "--adaptive-exploration",
        "--spawn-blind-search-arena",
        "--start-pose-override",
        "0.0",
        "0.0",
        "0.0",
        "--blind-arena-center",
        "5.5",
        "0.4",
        "--blind-arena-size",
        "13.0",
        "9.2",
        "--spawn-corridor-lobby",
        "--spawn-center-pillar",
        "--corridor-lobby-elevator-pose",
        "8.4",
        "3.0",
        "3.14159",
        "--grid-planner",
        "se2",
        "--se2-yaw-bins",
        "16",
        "--se2-step-distance",
        "0.6",
        "--se2-output-min-spacing",
        "1.2",
        "--se2-output-yaw-threshold",
        "0.55",
        "--perception-min-score",
        "0.5",
        "--perception-every",
        "50",
        "--localize-detection-with-depth",
        "--depth-localization-approach-distance",
        str(args.success_radius),
        "--motion-detection-image-dir",
        str(args.output_dir),
        "--trajectory-csv",
        str(trajectory_csv),
        "--fdm-snapshot-out",
        str(fdm_snapshot_out),
        "--result-json",
        str(result_json),
        "--local-executor",
        "fdm_mppi",
        "--fdm-run-dir",
        str(args.run_dir),
        "--fdm-checkpoint",
        str(args.checkpoint),
        "--fdm-mppi-population",
        "512",
        "--fdm-mppi-replan-every",
        "25",
        "--fdm-mppi-lookahead",
        "1.6",
        "--fdm-mppi-pass-tolerance",
        "0.5",
        "--fdm-mppi-progress-margin",
        "0.25",
        "--fdm-mppi-final-tolerance",
        str(args.success_radius),
        "--fdm-mppi-min-forward-carrot",
        "1.2",
        "--fdm-mppi-final-waypoint-handoff-distance",
        "1.0",
        "--fdm-mppi-disable-collision-cost-goal-radius",
        "2.0",
        "--fdm-mppi-disable-mppi-risk-cost-goal-radius",
        "2.0",
        "--fdm-mppi-no-face-subgoal",
        "--fdm-mppi-final-approach-distance",
        "2.0",
        "--fdm-mppi-final-approach-tolerance",
        str(args.success_radius),
    ]
    cmd[2:2] = ["--headless"]
    if do_record:
        cmd.extend(
            [
                "--record-run-dir",
                str(args.output_dir),
                "--record-every",
                str(args.record_every),
                "--record-resolution",
                str(args.record_resolution[0]),
                str(args.record_resolution[1]),
                "--record-top-center",
                str(args.record_top_center[0]),
                str(args.record_top_center[1]),
                "--record-top-height",
                str(args.record_top_height),
            ]
        )
    if args.log_llm:
        cmd.append("--log-llm")
    cmd.extend(passthrough)

    if args.show_command or args.dry_run:
        print("[semantic_nav:preset] expanded command:", flush=True)
        print(" ".join(f'"{part}"' if " " in part else part for part in cmd), flush=True)
    if args.dry_run:
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if do_record:
        _clean_recording_outputs(args.output_dir)
    result_json.unlink(missing_ok=True)
    with run_log.open("w", encoding="utf-8", errors="replace") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
    result_success = _read_result_success(result_json)
    parent_summary = f"[semantic_nav:preset] child_return_code={return_code} result_success={result_success} log={run_log}"
    print(parent_summary, flush=True)
    with run_log.open("a", encoding="utf-8", errors="replace") as log_file:
        log_file.write(parent_summary + "\n")
    if do_record and not args.no_video:
        _compose_recordings(args.output_dir, success=result_success)
    return return_code


def _clean_recording_outputs(output_dir: Path) -> None:
    for frame_dir_name in ("robot_view_frames", "topdown_frames"):
        frame_dir = output_dir / frame_dir_name
        if frame_dir.exists():
            for frame_path in _frame_files(frame_dir):
                frame_path.unlink(missing_ok=True)
        else:
            frame_dir.mkdir(parents=True, exist_ok=True)
    for pattern in (
        "robot_first_person_*.mp4",
        "topdown_*.mp4",
        "combined_first_person_topdown_*.mp4",
        "blind_*.jpg",
    ):
        for path in output_dir.glob(pattern):
            path.unlink(missing_ok=True)


def _read_result_success(result_json: Path) -> bool:
    if not result_json.exists():
        print(f"[semantic_nav:preset] missing result json: {result_json}", flush=True)
        return False
    try:
        data = json.loads(result_json.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[semantic_nav:preset] failed to read result json: {exc}", flush=True)
        return False
    return bool(data.get("success", False))


def _compose_recordings(output_dir: Path, *, success: bool) -> None:
    try:
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        import imageio.v2 as imageio
    except Exception as exc:
        print(f"[semantic_nav:record] video compose skipped: {exc}", flush=True)
        return

    fps = 20
    suffix = "success" if success else "partial"
    views = [
        ("robot_view_frames", output_dir / f"robot_first_person_{suffix}.mp4", "Robot First-Person View"),
        ("topdown_frames", output_dir / f"topdown_{suffix}.mp4", "Top-Down View"),
    ]
    try:
        title_font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 26)
        small_font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 18)
    except Exception:
        title_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    for frame_dir_name, out_path, title in views:
        frames = _frame_files(output_dir / frame_dir_name)
        if not frames:
            print(f"[semantic_nav:record] no frames in {output_dir / frame_dir_name}", flush=True)
            continue
        with imageio.get_writer(
            str(out_path),
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=1,
            pixelformat="yuv420p",
        ) as writer:
            last = None
            for path in frames:
                image = Image.open(path).convert("RGB")
                width, _ = image.size
                draw = ImageDraw.Draw(image, "RGBA")
                draw.rectangle((0, 0, width, 54), fill=(0, 0, 0, 100))
                draw.text((18, 12), title, font=title_font, fill=(255, 255, 255, 255))
                step = path.stem.split("_step_")[-1]
                draw.text((width - 165, 17), f"step {step}", font=small_font, fill=(255, 255, 255, 230))
                last = np.asarray(image)
                writer.append_data(last)
            if last is not None:
                for _ in range(fps * 2):
                    writer.append_data(last)
        print(f"[semantic_nav:record] wrote {out_path} frames={len(frames)}", flush=True)

    robot_frames = _frame_files(output_dir / "robot_view_frames")
    top_frames = _frame_files(output_dir / "topdown_frames")
    count = min(len(robot_frames), len(top_frames))
    if count == 0:
        return
    combined_path = output_dir / f"combined_first_person_topdown_{suffix}.mp4"
    with imageio.get_writer(
        str(combined_path),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
        pixelformat="yuv420p",
    ) as writer:
        last = None
        for idx in range(count):
            left = Image.open(robot_frames[idx]).convert("RGB").resize((640, 480), Image.LANCZOS)
            right = Image.open(top_frames[idx]).convert("RGB").resize((640, 480), Image.LANCZOS)
            image = Image.new("RGB", (1280, 520), (20, 24, 30))
            image.paste(left, (0, 40))
            image.paste(right, (640, 40))
            draw = ImageDraw.Draw(image, "RGBA")
            draw.rectangle((0, 0, 1280, 40), fill=(13, 36, 74, 255))
            draw.text((20, 7), "Robot First-Person View", font=small_font, fill=(255, 255, 255, 255))
            draw.text((660, 7), "Top-Down View", font=small_font, fill=(255, 255, 255, 255))
            step = robot_frames[idx].stem.split("_step_")[-1]
            draw.text((1160, 7), f"step {step}", font=small_font, fill=(255, 255, 255, 230))
            last = np.asarray(image)
            writer.append_data(last)
        if last is not None:
            for _ in range(fps * 2):
                writer.append_data(last)
    print(f"[semantic_nav:record] wrote {combined_path} frames={count}", flush=True)


def _frame_files(frame_dir: Path) -> list[Path]:
    return sorted([*frame_dir.glob("*.jpg"), *frame_dir.glob("*.png")])


def _preparse_task_before_isaac(args: argparse.Namespace) -> tuple[str, str, str]:
    if args.task_parser != "openai_compatible":
        return args.goal, args.target, args.task_parser
    if args.target.strip().lower() not in {"", "auto", "llm", "none", "null"}:
        return args.goal, args.target, "rule"

    if str(SCRIPTS_ROOT) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_ROOT))
    if str(SEMANTIC_NAV_ROOT) not in sys.path:
        sys.path.insert(0, str(SEMANTIC_NAV_ROOT))

    from envs.abstract_building_env import DEFAULT_BUILDING_CONFIG, load_semantic_graph
    from llm.factory import make_task_parser, release_task_parser_resources_from_args

    graph = load_semantic_graph(DEFAULT_BUILDING_CONFIG)
    parser = make_task_parser(
        args.task_parser,
        endpoint=args.llm_endpoint,
        model=args.llm_model,
        timeout_s=args.llm_timeout_s,
        log_raw=args.log_llm,
    )
    print("[semantic_nav:preset] pre-parsing task with local LLM before Isaac starts", flush=True)
    try:
        parsed = parser.parse(args.goal, current_floor="F1", graph=graph, start_node_id="start_f1")
    except Exception as exc:
        print(
            "[semantic_nav:preset] LLM pre-parse failed; falling back to rule parser "
            f"({type(exc).__name__}: {exc})",
            flush=True,
        )
        release_task_parser_resources_from_args(args)
        return args.goal, args.target, "rule"
    target = parsed.target_node_id or args.target
    print(
        f"[semantic_nav:preset] llm intent={parsed.goal.intent} target_floor={parsed.goal.target_floor} target_node={target}",
        flush=True,
    )
    release_task_parser_resources_from_args(args)
    return args.goal, target, "rule"


if __name__ == "__main__":
    raise SystemExit(main())
