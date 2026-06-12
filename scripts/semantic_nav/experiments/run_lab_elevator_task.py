from __future__ import annotations

"""Run the single-elevator semantic task in Isaac Lab.

Launches Isaac Sim before importing Lab/FDM modules, matching the pattern used by plan.py.
"""

import argparse
import base64
import csv
from dataclasses import dataclass
import json
from math import atan2, cos, hypot, sin
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np

from isaaclab.app import AppLauncher

SEMANTIC_NAV_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = SEMANTIC_NAV_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
if str(SEMANTIC_NAV_ROOT) not in sys.path:
    sys.path.insert(0, str(SEMANTIC_NAV_ROOT))

import utils.cli_args as cli_args  # isort: skip
from envs.abstract_building_env import DEFAULT_BUILDING_CONFIG  # isort: skip
from llm.factory import add_task_parser_args, make_task_parser_from_args, normalize_target_node_id, release_task_parser_resources_from_args  # isort: skip


LAB_TERRAIN_CHOICES = (
    "plane",
    "generator",
    "fdm_train",
    "fdm_train_no_terrain",
    "fdm_eval",
    "fdm_eval_no_terrain",
    "fdm_rough",
    "fdm_rough_no_terrain",
    "planner_eval",
    "planner_eval_no_terrain",
    "planner_eval_2d",
    "planner_eval_2d_no_terrain",
    "paper_figure",
    "paper_figure_no_terrain",
)
FDM_FLOOR_TERRAIN_NAMES = (
    "flat_reference",
    "grass_reference",
    "snow_reference",
    "mud_reference",
    "rough_floor_reference",
)


parser = argparse.ArgumentParser(description="Run the semantic single-elevator task in Isaac Lab.")
parser.add_argument("--building-config", type=Path, default=DEFAULT_BUILDING_CONFIG)
parser.add_argument("--start", default="start_f1")
parser.add_argument("--start-pose-override", type=float, nargs=3, default=None, help="Override start node local x y yaw.")
parser.add_argument("--goal", default="go downstairs to target room")
parser.add_argument("--target", default="target_room_b1")
add_task_parser_args(parser)
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--episode-length-s", type=float, default=90.0, help="Lab episode length for long semantic navigation runs.")
parser.add_argument("--print-every", type=int, default=50)
parser.add_argument("--xy-tolerance", type=float, default=0.45)
parser.add_argument("--max-vx", type=float, default=0.45)
parser.add_argument("--max-vy", type=float, default=0.08)
parser.add_argument("--max-wz", type=float, default=0.66)
parser.add_argument("--local-executor", choices=("waypoint", "fdm_mppi", "mppi"), default="waypoint")
parser.add_argument("--fdm-run-dir", type=Path, default=None, help="FDM run directory containing params/config.yaml.")
parser.add_argument("--fdm-checkpoint", type=Path, default=None, help="FDM model checkpoint used by --local-executor fdm_mppi.")
parser.add_argument("--fdm-mppi-population", type=int, default=512)
parser.add_argument("--fdm-mppi-replan-every", type=int, default=5)
parser.add_argument("--fdm-mppi-lookahead", type=float, default=2.0)
parser.add_argument("--fdm-mppi-pass-tolerance", type=float, default=0.75)
parser.add_argument("--fdm-mppi-progress-margin", type=float, default=0.25)
parser.add_argument("--fdm-mppi-final-tolerance", type=float, default=1.15)
parser.add_argument("--fdm-mppi-no-face-subgoal", action="store_true")
parser.add_argument("--fdm-mppi-min-forward-carrot", type=float, default=1.0)
parser.add_argument("--fdm-mppi-final-approach-distance", type=float, default=2.0)
parser.add_argument("--fdm-mppi-final-approach-tolerance", type=float, default=0.55)
parser.add_argument("--fdm-mppi-final-approach-yaw-tolerance", type=float, default=0.35)
parser.add_argument("--fdm-mppi-final-turn-wz", type=float, default=0.45)
parser.add_argument("--fdm-mppi-final-waypoint-handoff-distance", type=float, default=1.0)
parser.add_argument("--fdm-mppi-disable-collision-cost-goal-radius", type=float, default=0.0)
parser.add_argument("--fdm-mppi-disable-mppi-risk-cost-goal-radius", type=float, default=0.0)
parser.add_argument("--fdm-mppi-subgoal-yaw-gate", type=float, default=0.75)
parser.add_argument("--fdm-mppi-subgoal-yaw-tolerance", type=float, default=0.25)
parser.add_argument("--fdm-mppi-subgoal-turn-max-steps", type=int, default=60)
parser.add_argument("--fdm-mppi-min-vx", type=float, default=-0.1)
parser.add_argument("--fdm-mppi-max-vx", type=float, default=1.0)
parser.add_argument("--fdm-mppi-max-vy", type=float, default=0.3)
parser.add_argument("--fdm-mppi-max-wz", type=float, default=0.2)
parser.add_argument("--low-level-policy-file", type=Path, default=None, help="Override the G1 low-level gait policy path.")
parser.add_argument("--low-level-policy-mode", choices=("single", "dwaq"), default="single")
parser.add_argument("--low-level-obs-dim", type=int, default=None, help="Single-frame obs dim for multi-input low-level policies.")
parser.add_argument("--low-level-obs-history", type=int, default=5)
parser.add_argument("--dwaq-clip-deploy-command", action="store_true", help="Clip high-level commands to the original DWAQ deploy range.")
parser.add_argument("--dwaq-gait-phase-layout", choices=("deploy", "train"), default="deploy")
parser.add_argument("--humanoid-final-approach-distance", type=float, default=1.2)
parser.add_argument("--humanoid-final-approach-lateral-offset", type=float, default=0.0)
parser.add_argument("--trajectory-csv", type=Path, default=None, help="Optional CSV path for per-step semantic navigation trajectory logs.")
parser.add_argument("--fdm-snapshot-out", type=Path, default=None, help="Optional NPZ path for per-step FDM input snapshots.")
parser.add_argument(
    "--lab-terrain",
    choices=LAB_TERRAIN_CHOICES,
    default="plane",
    help=(
        "Terrain source for the Lab navigation scene. fdm_train includes the terrain_cfg "
        "flat/grass/snow/mud/rough floor mix used for FDM-style training."
    ),
)
parser.add_argument(
    "--detector",
    choices=("apexnav", "graph", "dummy_client", "http_client", "apexnav_gdino", "apexnav_yolov7"),
    default="apexnav",
)
parser.add_argument("--perception-endpoint", default=None, help="HTTP endpoint for external YOLO/GroundingDINO detections.")
parser.add_argument("--perception-min-score", type=float, default=0.75, help="Minimum score for accepting external perception detections.")
parser.add_argument("--image", type=Path, default=None, help="Debug image sent to an external ApexNav perception server.")
parser.add_argument("--log-detections", action="store_true", help="Print external perception detections and semantic node mapping.")
parser.add_argument("--spawn-lab-elevator", action="store_true", help="Spawn a minimal elevator-looking visual target in Isaac Lab.")
parser.add_argument("--use-lab-camera", action="store_true", help="Capture a Lab camera image and send it to the external detector.")
parser.add_argument("--lab-camera-image-out", type=Path, default=None, help="Optional path for the Lab camera debug image.")
parser.add_argument("--detect-during-motion", action="store_true", help="Run external perception periodically inside the execution loop.")
parser.add_argument("--perception-every", type=int, default=50, help="Motion-time perception interval in sim steps.")
parser.add_argument(
    "--localize-detection-with-depth",
    action="store_true",
    help="Use the accepted detector bbox and robot-view depth image to set the target pose.",
)
parser.add_argument(
    "--depth-localization-approach-distance",
    type=float,
    default=1.0,
    help="Distance to stop in front of the depth-localized detection point.",
)
parser.add_argument(
    "--depth-localization-max-node-distance",
    type=float,
    default=2.0,
    help="Reject a depth-localized hit if it is this far from the matched semantic node in XY; <=0 disables the check.",
)
parser.add_argument("--motion-detection-floor", default="B1", help="Floor on which the Lab visual target becomes visible.")
parser.add_argument("--motion-detection-node", default="elevator_b1", help="Semantic node used for motion-time Lab camera detection.")
parser.add_argument("--motion-detection-image-dir", type=Path, default=None, help="Optional directory for motion-time camera images.")
parser.add_argument("--stop-on-detected-node", default=None, help="Stop successfully once motion perception confirms this semantic node.")
parser.add_argument("--record-run-dir", type=Path, default=None, help="Optional directory for robot-view and top-down run recording.")
parser.add_argument(
    "--record-viewport",
    action="store_true",
    help="Record by capturing the active viewport instead of IsaacLab Camera sensor frames.",
)
parser.add_argument("--record-every", type=int, default=10, help="Capture one recording frame every N sim steps.")
parser.add_argument("--record-resolution", type=int, nargs=2, default=(640, 480), help="Recording frame width height.")
parser.add_argument("--record-top-center", type=float, nargs=2, default=(4.8, 0.4), help="Top-down camera center in local XY.")
parser.add_argument("--record-top-height", type=float, default=12.0, help="Top-down camera height.")
parser.add_argument("--result-json", type=Path, default=None, help="Optional path for run success/failure summary.")
parser.add_argument("--blind-find-elevator", action="store_true", help="Blindly explore until perception detects an elevator, then A* to it.")
parser.add_argument("--blind-find-object", action="store_true", help="Blindly explore until perception detects the requested semantic object, then A* to it.")
parser.add_argument("--search-node-id", default=None, help="Semantic graph node id to search for during blind object search.")
parser.add_argument("--search-kind", default=None, help="Semantic node kind accepted during blind object search; defaults to elevator_lobby for elevator mode.")
parser.add_argument("--search-label", default=None, help="Target object label used for detector prompts, for example fridge or refrigerator.")
parser.add_argument(
    "--search-prompts",
    default=None,
    help="Dot/comma separated detector prompts for blind object search. Defaults are inferred from search label or elevator mode.",
)
parser.add_argument("--blind-floor", default="F1", help="Floor used for blind elevator search.")
parser.add_argument("--blind-vx", type=float, default=0.35, help="Forward velocity during blind search.")
parser.add_argument("--blind-wz", type=float, default=0.08, help="Yaw velocity during blind search.")
parser.add_argument(
    "--blind-detection-confirmations",
    type=int,
    default=1,
    help="Number of object detections required before blind search switches to target navigation.",
)
parser.add_argument("--adaptive-exploration", action="store_true", help="Use frontier-like arena coverage before target detection.")
parser.add_argument("--exploration-spacing", type=float, default=1.6, help="Spacing between adaptive exploration sweeps.")
parser.add_argument("--spawn-blind-search-arena", action="store_true", help="Spawn a bounded arena for blind-search experiments.")
parser.add_argument("--grid-planner", choices=("xy", "se2"), default="xy")
parser.add_argument("--se2-yaw-bins", type=int, default=16)
parser.add_argument("--se2-step-distance", type=float, default=0.6)
parser.add_argument("--se2-output-min-spacing", type=float, default=0.55)
parser.add_argument("--se2-output-yaw-threshold", type=float, default=None)
parser.add_argument("--blind-arena-center", type=float, nargs=2, default=(4.0, 0.6), help="Blind arena center in local XY.")
parser.add_argument("--blind-arena-size", type=float, nargs=2, default=(9.5, 5.2), help="Blind arena size in local XY.")
parser.add_argument("--near-field-elevator", action="store_true", help="Place the blind-search elevator where it appears after a few forward steps.")
parser.add_argument("--near-field-elevator-pose", type=float, nargs=3, default=(3.2, 0.9, 3.14159), help="Local x y yaw for the near-field F1 elevator.")
parser.add_argument("--spawn-corridor-lobby", action="store_true", help="Spawn a corridor that opens into a wider elevator lobby.")
parser.add_argument(
    "--corridor-lobby-elevator-pose",
    type=float,
    nargs=3,
    default=(8.4, 3.0, 3.14159),
    help="Local x y yaw for elevator_f1 in the corridor-lobby scene.",
)
parser.add_argument("--spawn-occluder-wall", action="store_true", help="Spawn an internal wall so the elevator is not initially visible.")
parser.add_argument("--occluder-wall-center", type=float, nargs=2, default=(2.5, 1.55), help="Occluder wall center in local XY.")
parser.add_argument("--occluder-wall-size", type=float, nargs=2, default=(0.18, 1.0), help="Occluder wall size in local XY.")
parser.add_argument("--occluder-wall-height", type=float, default=2.4, help="Occluder wall height in meters.")
parser.add_argument("--spawn-center-pillar", action="store_true", help="Spawn a square pillar obstacle near the scene center.")
parser.add_argument("--center-pillar-center", type=float, nargs=2, default=None, help="Pillar center in local XY. Defaults to the blind arena center.")
parser.add_argument("--center-pillar-size", type=float, nargs=2, default=(0.65, 0.65), help="Pillar footprint size in local XY.")
parser.add_argument("--center-pillar-height", type=float, default=2.4, help="Pillar height in meters.")
parser.add_argument("--spawn-planner-eval-obstacles", action="store_true", help="Add terrain_cfg planner-eval style boxes/pillars/gates to the elevator-search scene.")
parser.add_argument("--planner-eval-obstacle-profile", choices=("light", "slalom", "dense"), default="slalom")
parser.add_argument("--exploration-wall-margin", type=float, default=1.6, help="Distance kept from arena walls by adaptive exploration viewpoints.")
cli_args.add_fdm_args(parser, default_num_envs=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.num_envs = 1
args_cli.robot = "g1"
if args_cli.use_lab_camera or args_cli.detect_during_motion or args_cli.blind_find_elevator or args_cli.blind_find_object or args_cli.localize_detection_with_depth or (
    args_cli.record_run_dir is not None and not args_cli.record_viewport
):
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import ObservationTermCfg as ObsTerm  # noqa: E402

import fdm.env_cfg.terrain_cfg as fdm_terrain_cfg  # noqa: E402
import fdm.mdp as mdp  # noqa: E402
from fdm.utils.args_cli_utils import cfg_modifier_pre_init, planner_cfg_init  # noqa: E402

from envs.abstract_building_env import load_semantic_graph  # noqa: E402
from executors.fdm_mppi_waypoint_executor import FdmMppiExecutorConfig, FdmMppiWaypointExecutor  # noqa: E402
from executors.lab_robot_adapter import LabRobotAdapter  # noqa: E402
from executors.semantic_execution_loop import PerceptionHookResult, run_semantic_execution_loop  # noqa: E402
from executors.waypoint_executor import WaypointExecutor, WaypointExecutorConfig  # noqa: E402
from executors.robot_adapter import ExecutionLoopResult, VelocityCommand  # noqa: E402
from maps.semantic_graph import Pose2D, SemanticNode  # noqa: E402
from lab_scene.elevator_scene import (  # noqa: E402
    LabRunRecorder,
    LabRunRecorderConfig,
    ViewportRunRecorder,
    capture_node_camera_b64,
    capture_robot_view_camera_b64,
    capture_robot_view_camera_observation,
    spawn_blind_search_arena,
    spawn_corridor_lobby_walls,
    spawn_elevator_nodes,
    spawn_minimal_elevator_scene,
    spawn_planner_eval_obstacles,
    spawn_rect_wall,
)
from perception.factory import make_semantic_detector  # noqa: E402
from planners.adaptive_exploration import AdaptiveExplorationStrategy, ExplorationViewpoint  # noqa: E402
from planners.astar_planner import SemanticAStarPlanner  # noqa: E402
from planners.execution_plan import ExecutionStep  # noqa: E402
from planners.execution_plan import build_execution_plan  # noqa: E402
from planners.grid_astar import GridAStarConfig, GridBounds, OccupancyGridAStar  # noqa: E402
from planners.humanoid_se2_astar import HumanoidSE2AStar, HumanoidSE2AStarConfig  # noqa: E402
from planners.semantic_task_planner import SemanticTaskPlanner  # noqa: E402


def _nearest_node_on_floor(graph, pose, floor: str) -> str | None:
    best_node_id = None
    best_dist = float("inf")
    for node in graph.nodes.values():
        if node.floor != floor:
            continue
        dx = node.pose.x - pose.x
        dy = node.pose.y - pose.y
        dist = dx * dx + dy * dy
        if dist < best_dist:
            best_node_id = node.node_id
            best_dist = dist
    return best_node_id


def _write_result_json(path: Path | None, result: ExecutionLoopResult, *, mode: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "success": result.success,
        "mode": mode,
        "steps": result.steps,
        "reason": result.reason,
        "final_pose": {
            "x": result.final_pose.x,
            "y": result.final_pose.y,
            "yaw": result.final_pose.yaw,
        },
        "final_step": result.final_step.node_id if result.final_step is not None else None,
        "confirmed_nodes": list(result.confirmed_nodes),
        "perception_events": list(result.perception_events[-20:]),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


@dataclass(frozen=True)
class DepthLocalizedTarget:
    hit_xyz: tuple[float, float, float]
    approach_pose: Pose2D
    depth_m: float
    pixel_uv: tuple[float, float]


class FdmSnapshotRecorder:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.rows: list[dict[str, Any]] = []
        self.state_history: list[Any] = []
        self.proprio_history: list[Any] = []
        self.extero_obs: list[Any] = []
        self.last_plan: list[Any] = []

    def record(
        self,
        *,
        step_idx: int,
        mode: str,
        active: str,
        pose: Pose2D,
        command: VelocityCommand,
        executor: Any,
    ) -> None:
        if self.path is None or executor is None or not hasattr(executor, "fdm_snapshot"):
            return
        snapshot = executor.fdm_snapshot()
        if snapshot is None:
            return
        self.rows.append(
            {
                "step": int(step_idx),
                "mode": mode,
                "active": active,
                "x": pose.x,
                "y": pose.y,
                "yaw": pose.yaw,
                "cmd_vx": command.vx,
                "cmd_vy": command.vy,
                "cmd_wz": command.wz,
            }
        )
        self.state_history.append(snapshot["state_history"].detach().cpu().numpy())
        self.proprio_history.append(snapshot["proprio_history"].detach().cpu().numpy())
        self.extero_obs.append(snapshot["extero_obs"].detach().cpu().numpy())
        self.last_plan.append(snapshot["last_plan"].detach().cpu().numpy())

    def close(self) -> None:
        if self.path is None or not self.rows:
            return
        import numpy as np

        self.path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "step": [row["step"] for row in self.rows],
            "mode": [row["mode"] for row in self.rows],
            "active": [row["active"] for row in self.rows],
            "x": [row["x"] for row in self.rows],
            "y": [row["y"] for row in self.rows],
            "yaw": [row["yaw"] for row in self.rows],
            "cmd_vx": [row["cmd_vx"] for row in self.rows],
            "cmd_vy": [row["cmd_vy"] for row in self.rows],
            "cmd_wz": [row["cmd_wz"] for row in self.rows],
        }
        np.savez_compressed(
            self.path,
            step=np.asarray(meta["step"], dtype=np.int64),
            mode=np.asarray(meta["mode"], dtype=object),
            active=np.asarray(meta["active"], dtype=object),
            x=np.asarray(meta["x"], dtype=np.float32),
            y=np.asarray(meta["y"], dtype=np.float32),
            yaw=np.asarray(meta["yaw"], dtype=np.float32),
            cmd_vx=np.asarray(meta["cmd_vx"], dtype=np.float32),
            cmd_vy=np.asarray(meta["cmd_vy"], dtype=np.float32),
            cmd_wz=np.asarray(meta["cmd_wz"], dtype=np.float32),
            state_history=np.concatenate(self.state_history, axis=0),
            proprio_history=np.concatenate(self.proprio_history, axis=0),
            extero_obs=np.stack(self.extero_obs, axis=0),
            last_plan=np.concatenate(self.last_plan, axis=0),
        )
        print(f"[semantic_nav:fdm_snapshot] wrote {len(self.rows)} snapshots to {self.path}", flush=True)


def _depth_localize_detection_target(
    *,
    observation,
    detection,
    robot_pose: Pose2D,
    approach_distance: float,
) -> DepthLocalizedTarget | None:
    bbox = getattr(detection, "bbox", None)
    if bbox is None or observation is None or observation.depth is None:
        return None
    if observation.intrinsics is None or observation.camera_eye is None or observation.camera_target is None:
        return None

    depth = np.asarray(observation.depth, dtype=np.float32)
    if depth.ndim != 2:
        return None
    height, width = depth.shape
    x1 = int(np.clip(bbox.x1 * width, 0, width - 1))
    x2 = int(np.clip(bbox.x2 * width, x1 + 1, width))
    y1 = int(np.clip(bbox.y1 * height, 0, height - 1))
    y2 = int(np.clip(bbox.y2 * height, y1 + 1, height))
    if x2 <= x1 or y2 <= y1:
        return None

    roi_x1 = int(round(x1 + 0.30 * (x2 - x1)))
    roi_x2 = int(round(x1 + 0.70 * (x2 - x1)))
    roi_y1 = int(round(y1 + 0.35 * (y2 - y1)))
    roi_y2 = int(round(y1 + 0.75 * (y2 - y1)))
    roi_x1 = max(x1, min(roi_x1, x2 - 1))
    roi_x2 = max(roi_x1 + 1, min(roi_x2, x2))
    roi_y1 = max(y1, min(roi_y1, y2 - 1))
    roi_y2 = max(roi_y1 + 1, min(roi_y2, y2))
    roi_depth = depth[roi_y1:roi_y2, roi_x1:roi_x2]
    valid = np.isfinite(roi_depth) & (roi_depth > 0.15) & (roi_depth < 30.0)
    if not np.any(valid):
        roi_depth = depth[y1:y2, x1:x2]
        valid = np.isfinite(roi_depth) & (roi_depth > 0.15) & (roi_depth < 30.0)
        if not np.any(valid):
            return None
    depth_m = float(np.median(roi_depth[valid]))
    pixel_u = float((roi_x1 + roi_x2 - 1) * 0.5)
    pixel_v = float((roi_y1 + roi_y2 - 1) * 0.5)

    u_i = int(np.clip(round(pixel_u), 0, width - 1))
    v_i = int(np.clip(round(pixel_v), 0, height - 1))
    intrinsics = np.asarray(observation.intrinsics, dtype=np.float32)
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    if abs(fx) < 1e-6 or abs(fy) < 1e-6:
        return None
    eye = np.asarray(observation.camera_eye, dtype=np.float32)
    target = np.asarray(observation.camera_target, dtype=np.float32)
    forward = target - eye
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm < 1e-6:
        return None
    forward = forward / forward_norm
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    right_norm = float(np.linalg.norm(right))
    if right_norm < 1e-6:
        return None
    right = right / right_norm
    up = np.cross(right, forward)
    up_norm = float(np.linalg.norm(up))
    if up_norm < 1e-6:
        return None
    up = up / up_norm
    hit = eye + forward * depth_m + right * ((u_i - cx) / fx * depth_m) - up * ((v_i - cy) / fy * depth_m)
    if observation.world_origin is not None:
        hit = hit - np.asarray(observation.world_origin, dtype=np.float32)
    hit_x = float(hit[0])
    hit_y = float(hit[1])
    hit_z = float(hit[2])
    if not (0.1 <= hit_z <= 3.0):
        return None

    dx = robot_pose.x - hit_x
    dy = robot_pose.y - hit_y
    distance = hypot(dx, dy)
    if distance < 1e-3:
        return None
    stop_distance = max(0.0, approach_distance)
    approach_x = hit_x + dx / distance * stop_distance
    approach_y = hit_y + dy / distance * stop_distance
    approach_yaw = atan2(hit_y - approach_y, hit_x - approach_x)
    return DepthLocalizedTarget(
        hit_xyz=(hit_x, hit_y, hit_z),
        approach_pose=Pose2D(approach_x, approach_y, approach_yaw),
        depth_m=depth_m,
        pixel_uv=(pixel_u, pixel_v),
    )


def _build_walk_steps_from_path(graph, node_ids: list[str]) -> list[ExecutionStep]:
    steps: list[ExecutionStep] = []
    for node_id in node_ids[1:]:
        node = graph.nodes[node_id]
        steps.append(
            ExecutionStep(
                kind="walk_to",
                node_id=node.node_id,
                floor=node.floor,
                pose=node.pose,
                description=f"walk to {node.node_id}",
            )
        )
    return steps


def _build_walk_steps_from_poses(
    poses: list[Pose2D],
    *,
    floor: str,
    target_node_id: str,
    preserve_yaw: bool = False,
) -> list[ExecutionStep]:
    if not preserve_yaw:
        poses = _humanoidize_grid_path(
            poses,
            final_approach_distance=args_cli.humanoid_final_approach_distance,
            final_lateral_offset=args_cli.humanoid_final_approach_lateral_offset,
        )
    steps: list[ExecutionStep] = []
    for idx, pose in enumerate(poses[1:], start=1):
        node_id = target_node_id if idx == len(poses) - 1 else f"grid_wp_{idx:03d}"
        steps.append(
            ExecutionStep(
                kind="walk_to",
                node_id=node_id,
                floor=floor,
                pose=pose,
                description=f"grid walk to {node_id}",
            )
        )
    return steps


def _humanoidize_grid_path(
    poses: list[Pose2D],
    *,
    final_approach_distance: float,
    final_lateral_offset: float,
) -> list[Pose2D]:
    if len(poses) < 2:
        return poses

    path = list(poses)
    if final_approach_distance > 0.0 and len(path) >= 2:
        final_pose = path[-1]
        prev_pose = path[-2]
        final_segment_length = hypot(final_pose.x - prev_pose.x, final_pose.y - prev_pose.y)
        approach_yaw = atan2(final_pose.y - prev_pose.y, final_pose.x - prev_pose.x)
        if final_segment_length > final_approach_distance + 0.35:
            approach_pose = Pose2D(
                final_pose.x - final_approach_distance * cos(approach_yaw) - final_lateral_offset * sin(approach_yaw),
                final_pose.y - final_approach_distance * sin(approach_yaw) + final_lateral_offset * cos(approach_yaw),
                approach_yaw,
            )
            path.insert(-1, approach_pose)

    result: list[Pose2D] = []
    for idx, pose in enumerate(path):
        if idx + 1 < len(path):
            next_pose = path[idx + 1]
            yaw = atan2(next_pose.y - pose.y, next_pose.x - pose.x)
        elif idx > 0:
            prev_pose = path[idx - 1]
            yaw = atan2(pose.y - prev_pose.y, pose.x - prev_pose.x)
        else:
            yaw = pose.yaw
        result.append(Pose2D(pose.x, pose.y, yaw))
    return result


def _split_search_prompts(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    parts = value.replace(".", ",").split(",")
    return tuple(part.strip() for part in parts if part.strip())


def _search_terms_from_label(label: str | None) -> tuple[str, ...]:
    if label is None:
        return ()
    terms = _split_search_prompts(label)
    if not terms:
        terms = (label.strip(),) if label.strip() else ()
    extras: list[str] = []
    for term in terms:
        normalized = term.lower()
        if normalized == "fridge":
            extras.append("refrigerator")
        elif normalized == "refrigerator":
            extras.append("fridge")
    return tuple(dict.fromkeys([*terms, *extras]))


def _blind_search_prompts(graph, target_node_id: str | None) -> tuple[str, ...]:
    explicit = _split_search_prompts(args_cli.search_prompts)
    if explicit:
        return explicit
    label_terms = _search_terms_from_label(args_cli.search_label)
    if label_terms:
        return label_terms
    if args_cli.blind_find_object and target_node_id is not None and target_node_id in graph.nodes:
        node = graph.nodes[target_node_id]
        node_terms = _split_search_prompts(str(node.attrs.get("detection_label", "")))
        if node_terms:
            return node_terms
        if node.label:
            return _search_terms_from_label(node.label)
    return ("elevator", "lift", "elevator door", "elevator sign")


def _node_search_terms(node: SemanticNode) -> set[str]:
    values = [
        node.node_id,
        node.label,
        node.kind,
        str(node.attrs.get("detection_label", "")),
        str(node.attrs.get("semantic_hint", "")),
    ]
    terms: set[str] = set()
    for value in values:
        stripped = value.strip().lower()
        if stripped:
            terms.add(stripped)
        for token in stripped.replace("_", " ").replace("-", " ").split():
            if len(token) > 2:
                terms.add(token)
    return terms


def _matches_blind_search_target(node: SemanticNode, *, floor: str) -> bool:
    if node.floor != floor:
        return False
    if args_cli.search_node_id:
        return node.node_id == args_cli.search_node_id
    search_kind = args_cli.search_kind
    if search_kind is None and not args_cli.blind_find_object:
        search_kind = "elevator_lobby"
    if search_kind is not None and node.kind != search_kind:
        return False
    label_terms = _search_terms_from_label(args_cli.search_label)
    if label_terms:
        node_terms = _node_search_terms(node)
        for term in label_terms:
            normalized = term.lower()
            if normalized in node_terms:
                return True
            if any(normalized in node_term or node_term in normalized for node_term in node_terms):
                return True
        return False
    return search_kind is not None


def _make_executor(
    *,
    env,
    steps: list[ExecutionStep],
    cfg: WaypointExecutorConfig,
    planner_cfg,
) -> WaypointExecutor | FdmMppiWaypointExecutor:
    if args_cli.local_executor not in ("fdm_mppi", "mppi"):
        return WaypointExecutor(steps, cfg)
    use_fdm = args_cli.local_executor == "fdm_mppi"
    if use_fdm and (args_cli.fdm_run_dir is None or args_cli.fdm_checkpoint is None):
        print("[semantic_nav:fdm_mppi] missing --fdm-run-dir/--fdm-checkpoint; falling back to waypoint executor", flush=True)
        return WaypointExecutor(steps, cfg)
    print(
        f"[semantic_nav:{args_cli.local_executor}] creating executor steps={len(steps)} "
        f"population={args_cli.fdm_mppi_population} replan_every={args_cli.fdm_mppi_replan_every}",
        flush=True,
    )
    return FdmMppiWaypointExecutor(
        env=env,
        steps=steps,
        waypoint_cfg=cfg,
        planner_cfg=planner_cfg,
        fdm_cfg=FdmMppiExecutorConfig(
            run_dir=args_cli.fdm_run_dir if args_cli.fdm_run_dir is not None else Path("."),
            checkpoint=args_cli.fdm_checkpoint if args_cli.fdm_checkpoint is not None else Path("."),
            use_fdm=use_fdm,
            population_size=args_cli.fdm_mppi_population,
            replan_every=args_cli.fdm_mppi_replan_every,
            lookahead_distance=args_cli.fdm_mppi_lookahead,
            pass_tolerance=args_cli.fdm_mppi_pass_tolerance,
            progress_margin=args_cli.fdm_mppi_progress_margin,
            final_tolerance=args_cli.fdm_mppi_final_tolerance,
            face_subgoal=not args_cli.fdm_mppi_no_face_subgoal,
            min_forward_carrot=args_cli.fdm_mppi_min_forward_carrot,
            final_approach_distance=args_cli.fdm_mppi_final_approach_distance,
            final_approach_tolerance=args_cli.fdm_mppi_final_approach_tolerance,
            final_approach_yaw_tolerance=args_cli.fdm_mppi_final_approach_yaw_tolerance,
            final_turn_wz=args_cli.fdm_mppi_final_turn_wz,
            final_waypoint_handoff_distance=args_cli.fdm_mppi_final_waypoint_handoff_distance,
            disable_collision_cost_goal_radius=args_cli.fdm_mppi_disable_collision_cost_goal_radius,
            disable_mppi_risk_cost_goal_radius=args_cli.fdm_mppi_disable_mppi_risk_cost_goal_radius,
            subgoal_yaw_gate=args_cli.fdm_mppi_subgoal_yaw_gate,
            subgoal_yaw_tolerance=args_cli.fdm_mppi_subgoal_yaw_tolerance,
            subgoal_turn_max_steps=args_cli.fdm_mppi_subgoal_turn_max_steps,
            min_vx=args_cli.fdm_mppi_min_vx,
            max_vx=args_cli.fdm_mppi_max_vx,
            max_vy=args_cli.fdm_mppi_max_vy,
            max_wz=args_cli.fdm_mppi_max_wz,
        ),
    )


def run_blind_find_then_astar(
    *,
    graph,
    robot: LabRobotAdapter,
    planner_cfg,
    detector_kind: str,
    perception_endpoint: str | None,
    max_steps: int,
    print_every: int,
    perception_every: int,
    image_dir: Path | None,
    floor: str,
    cfg: WaypointExecutorConfig,
    grid_planner: OccupancyGridAStar | None = None,
    se2_planner: HumanoidSE2AStar | None = None,
    exploration_strategy: AdaptiveExplorationStrategy | None = None,
    recorder: LabRunRecorder | None = None,
    target_node_id: str | None = None,
) -> ExecutionLoopResult:
    astar = SemanticAStarPlanner(graph)
    search_prompts = _blind_search_prompts(graph, target_node_id)
    executor: WaypointExecutor | FdmMppiWaypointExecutor | None = None
    executor_mode = "blind"
    perception_events: list[str] = []
    confirmed_nodes: set[str] = set()
    detection_confirm_counts: dict[str, int] = {}
    csv_file = None
    csv_writer = None
    fdm_snapshot_recorder = FdmSnapshotRecorder(args_cli.fdm_snapshot_out)
    if args_cli.trajectory_csv is not None:
        args_cli.trajectory_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_file = args_cli.trajectory_csv.open("w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "step",
                "floor",
                "mode",
                "active",
                "x",
                "y",
                "yaw",
                "cmd_vx",
                "cmd_vy",
                "cmd_wz",
                "event",
                "illegal_contact",
                "target_name",
                "target_x",
                "target_y",
                "target_yaw",
                "target_dist",
                "target_yaw_err",
                "subgoal_turning",
                "subgoal_turn_steps",
            ],
        )
        csv_writer.writeheader()

    try:
        for step_idx in range(max_steps):
            pose = robot.pose()
            event = ""
            command = VelocityCommand(0.0, 0.0, 0.0)

            if (
                executor_mode != "target_grid_astar"
                and executor_mode != "target_semantic_astar"
                and perception_every > 0
                and step_idx > 0
                and step_idx % perception_every == 0
            ):
                image_path = None
                if image_dir is not None:
                    image_dir.mkdir(parents=True, exist_ok=True)
                    image_path = image_dir / f"blind_{step_idx:05d}_{floor}.jpg"
                camera_observation = None
                if args_cli.localize_detection_with_depth:
                    camera_observation = capture_robot_view_camera_observation(
                        env=robot.env,
                        robot_pose=pose,
                        image_path=image_path,
                    )
                    image_b64 = camera_observation.image_jpeg_b64
                else:
                    image_b64 = capture_robot_view_camera_b64(env=robot.env, robot_pose=pose, image_path=image_path)
                runtime_detector = make_semantic_detector(
                    detector_kind,
                    graph=graph,
                    perception_endpoint=perception_endpoint,
                    image_jpeg_b64=image_b64,
                    log_detections=True,
                    min_score=args_cli.perception_min_score,
                    prompts=search_prompts,
                )
                current_node_id = _nearest_node_on_floor(graph, pose, floor)
                try:
                    detections = runtime_detector.detect(graph, current_node_id=current_node_id)
                except Exception as exc:
                    event = f"blind_perception_failed_floor={floor} camera=robot_view error={type(exc).__name__}: {exc}"
                    perception_events.append(f"step={step_idx} {event}")
                    detections = []
                target_detections = [
                    detection
                    for detection in detections
                    if (
                        (args_cli.blind_find_object and detection.node_id is None)
                        or (
                            detection.node_id is not None
                            and _matches_blind_search_target(graph.nodes[detection.node_id], floor=floor)
                        )
                    )
                ]
                selected_nodes = tuple(detection.node_id or f"open:{detection.label}" for detection in target_detections)
                selected = ",".join(selected_nodes) or "none"
                event = f"blind_perception_floor={floor} camera=robot_view prompts={','.join(search_prompts)} selected={selected}"
                perception_events.append(f"step={step_idx} {event}")
                if target_detections and current_node_id is not None:
                    def _detection_priority(detection) -> tuple[int, float]:
                        label = detection.label.lower()
                        if "door" in label:
                            return (0, -detection.score)
                        if label.strip() == "elevator" or label.strip() == "lift":
                            return (1, -detection.score)
                        return (2, -detection.score)

                    selected_detection = sorted(target_detections, key=_detection_priority)[0]
                    target_node_id = selected_detection.node_id or f"detected_{selected_detection.label.lower().replace(' ', '_').replace('-', '_')}"
                    required_confirmations = max(1, args_cli.blind_detection_confirmations)
                    detection_confirm_counts[target_node_id] = detection_confirm_counts.get(target_node_id, 0) + 1
                    confirmation_count = detection_confirm_counts[target_node_id]
                    if confirmation_count < required_confirmations:
                        event = (
                            f"{event}; pending_detection={target_node_id} "
                            f"score={selected_detection.score:.3f} confirm={confirmation_count}/{required_confirmations}"
                        )
                        perception_events[-1] = f"step={step_idx} {event}"
                    else:
                        confirmed_nodes.add(target_node_id)
                        graph_target_pose = graph.nodes[target_node_id].pose if selected_detection.node_id is not None else None
                        target_pose = graph_target_pose or pose
                        localized_target = _depth_localize_detection_target(
                            observation=camera_observation,
                            detection=selected_detection,
                            robot_pose=pose,
                            approach_distance=args_cli.depth_localization_approach_distance,
                        )
                        depth_target_rejected = False
                        if (
                            localized_target is not None
                            and graph_target_pose is not None
                            and args_cli.depth_localization_max_node_distance > 0.0
                            and hypot(
                                localized_target.hit_xyz[0] - graph_target_pose.x,
                                localized_target.hit_xyz[1] - graph_target_pose.y,
                            )
                            > args_cli.depth_localization_max_node_distance
                        ):
                            hit_x, hit_y, hit_z = localized_target.hit_xyz
                            hit_distance = hypot(hit_x - graph_target_pose.x, hit_y - graph_target_pose.y)
                            print(
                                "[semantic_nav:perception] depth_rejected "
                                f"node={target_node_id} hit_distance={hit_distance:.2f}m "
                                f"limit={args_cli.depth_localization_max_node_distance:.2f}m "
                                f"hit=({hit_x:.2f},{hit_y:.2f},{hit_z:.2f})",
                                flush=True,
                            )
                            event = (
                                f"{event}; depth_target_rejected={target_node_id} "
                                f"hit_distance={hit_distance:.2f}"
                            )
                            localized_target = None
                            depth_target_rejected = True
                        if localized_target is not None:
                            target_pose = localized_target.approach_pose
                            hit_x, hit_y, hit_z = localized_target.hit_xyz
                            print(
                                "[semantic_nav:perception] depth_localized "
                                f"node={target_node_id} depth={localized_target.depth_m:.2f}m "
                                f"pixel=({localized_target.pixel_uv[0]:.1f},{localized_target.pixel_uv[1]:.1f}) "
                                f"hit=({hit_x:.2f},{hit_y:.2f},{hit_z:.2f}) "
                                f"approach=({target_pose.x:.2f},{target_pose.y:.2f},{target_pose.yaw:.2f})",
                                flush=True,
                            )
                            event = (
                                f"{event}; depth_target={target_node_id} "
                                f"hit=({hit_x:.2f},{hit_y:.2f},{hit_z:.2f}) "
                                f"approach=({target_pose.x:.2f},{target_pose.y:.2f},{target_pose.yaw:.2f})"
                            )
                        elif args_cli.localize_detection_with_depth and not depth_target_rejected:
                            event = f"{event}; depth_target_failed={target_node_id}"
                        if selected_detection.node_id is None and localized_target is None:
                            event = f"{event}; open_target_no_depth={target_node_id}"
                            perception_events[-1] = f"step={step_idx} {event}"
                        elif grid_planner is not None:
                            planner = se2_planner if se2_planner is not None else grid_planner
                            grid_path = planner.plan(pose, target_pose)
                            if grid_path:
                                executor = _make_executor(
                                    env=robot.env,
                                    steps=_build_walk_steps_from_poses(
                                        grid_path,
                                        floor=floor,
                                        target_node_id=target_node_id,
                                        preserve_yaw=se2_planner is not None,
                                    ),
                                    cfg=cfg,
                                    planner_cfg=planner_cfg,
                                )
                                executor_mode = "target_grid_astar"
                                path_text = "->".join(f"({wp.x:.1f},{wp.y:.1f})" for wp in grid_path)
                                print(f"[semantic_nav:astar] selected_path={path_text}", flush=True)
                                event = f"{event}; grid_astar_target={target_node_id} path={path_text}"
                            else:
                                event = f"{event}; grid_astar_failed target={target_node_id}"
                        if executor is None:
                            if selected_detection.node_id is not None:
                                path = astar.plan_to_any(
                                    start=current_node_id,
                                    goal_node_ids=[target_node_id],
                                    edge_filter=lambda edge: edge.kind == "walk",
                                )
                                if not path.is_empty:
                                    executor = _make_executor(
                                        env=robot.env,
                                        steps=_build_walk_steps_from_path(graph, path.node_ids),
                                        cfg=cfg,
                                        planner_cfg=planner_cfg,
                                    )
                                    executor_mode = "target_semantic_astar"
                                    event = f"{event}; astar_start={current_node_id} astar_target={target_node_id} path={'->'.join(path.node_ids)}"
                                else:
                                    event = f"{event}; astar_failed start={current_node_id} target={target_node_id}"
                            else:
                                event = f"{event}; open_target_requires_grid_planner={target_node_id}"
                        perception_events[-1] = f"step={step_idx} {event}"

            if executor is None:
                if exploration_strategy is not None and grid_planner is not None:
                    viewpoint = exploration_strategy.next_viewpoint(robot.pose())
                    if viewpoint is not None:
                        grid_path = grid_planner.plan(robot.pose(), viewpoint.pose)
                        if grid_path:
                            executor = _make_executor(
                                env=robot.env,
                                steps=_build_walk_steps_from_poses(grid_path, floor=floor, target_node_id=viewpoint.name),
                                cfg=cfg,
                                planner_cfg=planner_cfg,
                            )
                            executor_mode = "explore_grid_astar"
                            path_text = "->".join(f"({wp.x:.1f},{wp.y:.1f})" for wp in grid_path)
                            event = f"{event}; explore_target={viewpoint.name} path={path_text}" if event else f"explore_target={viewpoint.name} path={path_text}"
                        else:
                            exploration_strategy.mark_reached()
                    else:
                        event = f"{event}; no_exploration_viewpoint" if event else "no_exploration_viewpoint"
                if executor is None:
                    command = VelocityCommand(args_cli.blind_vx, 0.0, args_cli.blind_wz)
                    robot.step_velocity(command)
                else:
                    command, status = executor.update(robot.pose())
                    robot.step_velocity(command)
                    if status.done and executor_mode == "explore_grid_astar" and exploration_strategy is not None:
                        exploration_strategy.mark_reached()
                        executor = None
                        executor_mode = "blind"
                active_name = executor_mode if executor is not None else "blind_search"
                done = False
            else:
                command, status = executor.update(robot.pose())
                robot.step_velocity(command)
                active_step = status.active_step
                active_name = f"{executor_mode}:{active_step.node_id if active_step is not None else 'done'}"
                done = status.done
                if status.done and executor_mode == "explore_grid_astar" and exploration_strategy is not None:
                    exploration_strategy.mark_reached()
                    executor = None
                    executor_mode = "blind"
                    done = False
                if status.event:
                    event = f"{event}; {status.event}" if event else status.event

            pose = robot.pose()
            illegal_contact = robot.illegal_contact()
            if robot.consume_reset_event():
                if executor is not None and hasattr(executor, "notify_env_reset"):
                    executor.notify_env_reset()
                executor = None
                executor_mode = "blind"
                active_name = "blind_search"
                reset_reason = robot.last_reset_reason() if hasattr(robot, "last_reset_reason") else ""
                reset_event = "env_reset_detected_restarting_search"
                if reset_reason:
                    reset_event = f"{reset_event} {reset_reason}"
                event = f"{event}; {reset_event}" if event else reset_event
            if csv_writer is not None:
                debug_info = executor.debug_info() if executor is not None and hasattr(executor, "debug_info") else {}
                csv_writer.writerow(
                    {
                        "step": step_idx,
                        "floor": floor,
                        "mode": executor_mode,
                        "active": active_name,
                        "x": pose.x,
                        "y": pose.y,
                        "yaw": pose.yaw,
                        "cmd_vx": command.vx,
                        "cmd_vy": command.vy,
                        "cmd_wz": command.wz,
                        "event": event,
                        "illegal_contact": illegal_contact,
                        "target_name": debug_info.get("target_name", ""),
                        "target_x": debug_info.get("target_x", ""),
                        "target_y": debug_info.get("target_y", ""),
                        "target_yaw": debug_info.get("target_yaw", ""),
                        "target_dist": debug_info.get("target_dist", ""),
                        "target_yaw_err": debug_info.get("target_yaw_err", ""),
                        "subgoal_turning": debug_info.get("subgoal_turning", ""),
                        "subgoal_turn_steps": debug_info.get("subgoal_turn_steps", ""),
                    }
                )
                csv_file.flush()
            fdm_snapshot_recorder.record(
                step_idx=step_idx,
                mode=executor_mode,
                active=active_name,
                pose=pose,
                command=command,
                executor=executor,
            )
            if print_every > 0 and (step_idx % print_every == 0 or event):
                print(
                    f"[semantic_nav:blind] step={step_idx} floor={floor} active={active_name} "
                    f"pose=({pose.x:.3f}, {pose.y:.3f}, {pose.yaw:.3f}) event={event or '-'} "
                    f"illegal_contact={illegal_contact}",
                    flush=True,
                )
            if recorder is not None:
                recorder.capture(step_idx=step_idx, robot_pose=pose)
            if done:
                return ExecutionLoopResult(
                    success=True,
                    steps=step_idx,
                    final_pose=pose,
                    final_step=executor.current_step() if executor is not None else None,
                    reason="reached detected elevator by A*",
                    perception_events=tuple(perception_events),
                    confirmed_nodes=tuple(sorted(confirmed_nodes)),
                )

        return ExecutionLoopResult(
            success=False,
            steps=max_steps,
            final_pose=robot.pose(),
            final_step=executor.current_step() if executor is not None else None,
            reason="timeout",
            perception_events=tuple(perception_events),
            confirmed_nodes=tuple(sorted(confirmed_nodes)),
        )
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        if csv_file is not None:
            csv_file.close()
        fdm_snapshot_recorder.close()


def _apply_low_level_gait_overrides(cfg) -> None:
    action_cfg = getattr(getattr(cfg.env_cfg, "actions", None), "velocity_cmd", None)
    if action_cfg is None:
        return
    dwaq_term_dims = [3, 3, 3, 29, 29, 29, 4]
    if args_cli.low_level_policy_file is not None:
        action_cfg.low_level_policy_file = str(args_cli.low_level_policy_file)
    action_cfg.low_level_policy_mode = args_cli.low_level_policy_mode
    action_cfg.low_level_obs_dim = args_cli.low_level_obs_dim
    action_cfg.low_level_obs_history = args_cli.low_level_obs_history
    if args_cli.low_level_policy_mode != "dwaq":
        return

    policy_obs = getattr(cfg.env_cfg.observations, "policy", None)
    if policy_obs is not None:
        if hasattr(policy_obs, "joint_vel"):
            policy_obs.joint_vel.scale = 1.0
        if hasattr(policy_obs, "joint_vel_rel"):
            policy_obs.joint_vel_rel.scale = 1.0
        policy_obs.gait_phase = ObsTerm(
            func=mdp.dwaq_gait_phase,
            params={"period": 0.8, "offset": 0.5, "layout": args_cli.dwaq_gait_phase_layout},
        )
    action_cfg.low_level_obs_term_dims = dwaq_term_dims
    if action_cfg.low_level_obs_dim is None:
        action_cfg.low_level_obs_dim = sum(dwaq_term_dims)
    if args_cli.dwaq_clip_deploy_command:
        action_cfg.clip_mode = "minmax"
        action_cfg.clip = [(-0.4, 0.7), (-0.4, 0.4), (-1.57, 1.57)]


def _apply_lab_terrain(cfg) -> None:
    terrain = cfg.env_cfg.scene.terrain
    selected = args_cli.lab_terrain
    print(f"[semantic_nav:lab] applying terrain={selected}", flush=True)
    if selected == "plane":
        terrain.terrain_type = "plane"
        terrain.terrain_generator = None
        terrain.random_seed = 0
        print("[semantic_nav:lab] terrain=plane", flush=True)
        return

    terrain_generators = {
        "generator": fdm_terrain_cfg.PAPER_FIGURE_TERRAIN_CFG,
        "paper_figure": fdm_terrain_cfg.PAPER_FIGURE_TERRAIN_CFG,
        "fdm_train": _fdm_train_floor_only_terrain_generator(),
        "fdm_train_no_terrain": fdm_terrain_cfg.FDM_TERRAINS_NO_TERRAIN_CFG,
        "fdm_eval": fdm_terrain_cfg.FDM_EVAL_EXTEROCEPTIVE_TERRAINS_CFG,
        "fdm_eval_no_terrain": fdm_terrain_cfg.FDM_EVAL_EXTEROCEPTIVE_TERRAINS_NO_TERRAIN_CFG,
        "fdm_rough": fdm_terrain_cfg.FDM_ROUGH_TERRAINS_CFG,
        "fdm_rough_no_terrain": fdm_terrain_cfg.FDM_ROUGH_TERRAINS_NO_TERRAIN_CFG,
        "planner_eval": fdm_terrain_cfg.PLANNER_EVAL_CFG,
        "planner_eval_no_terrain": fdm_terrain_cfg.PLANNER_EVAL_NO_TERRAIN_CFG,
        "planner_eval_2d": fdm_terrain_cfg.PLANNER_EVAL_2D_CFG,
        "planner_eval_2d_no_terrain": fdm_terrain_cfg.PLANNER_EVAL_2D_NO_TERRAIN_CFG,
        "paper_figure_no_terrain": fdm_terrain_cfg.PAPER_FIGURE_NO_TERRAIN_CFG,
    }
    terrain.terrain_type = "generator"
    terrain.terrain_generator = terrain_generators[selected]
    terrain.random_seed = 0
    if hasattr(terrain, "groundplane"):
        terrain.groundplane = False
    sub_terrains = getattr(terrain.terrain_generator, "sub_terrains", {})
    print(
        f"[semantic_nav:lab] terrain={selected} generator_sub_terrains={','.join(sub_terrains.keys())}",
        flush=True,
    )


def _fdm_train_floor_only_terrain_generator():
    generator = fdm_terrain_cfg._terrain_generator(
        "train",
        border_width=1.0,
        num_rows=8,
        num_cols=8,
    )
    generator.sub_terrains = {
        name: generator.sub_terrains[name]
        for name in FDM_FLOOR_TERRAIN_NAMES
        if name in generator.sub_terrains
    }
    total_proportion = sum(float(getattr(sub_cfg, "proportion", 0.0)) for sub_cfg in generator.sub_terrains.values())
    if total_proportion > 0.0:
        for sub_cfg in generator.sub_terrains.values():
            sub_cfg.proportion = float(sub_cfg.proportion) / total_proportion
    return generator


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    env = None

    try:
        release_task_parser_resources_from_args(args_cli)
        cfg = planner_cfg_init(args_cli)
        cfg = cfg_modifier_pre_init(cfg, args_cli)
        _apply_low_level_gait_overrides(cfg)
        cfg.env_cfg.scene.num_envs = 1
        cfg.env_cfg.episode_length_s = args_cli.episode_length_s
        command_cfg = getattr(getattr(cfg.env_cfg, "commands", None), "command", None)
        if hasattr(command_cfg, "debug_vis"):
            command_cfg.debug_vis = False
        _apply_lab_terrain(cfg)
        cfg.env_cfg.events.reset_base.func = mdp.reset_root_state_center
        cfg.env_cfg.events.reset_base.params = {}

        env = ManagerBasedRLEnv(cfg.env_cfg)

        graph = load_semantic_graph(args_cli.building_config)
        if (args_cli.near_field_elevator or args_cli.spawn_corridor_lobby) and "elevator_f1" in graph.nodes:
            old_node = graph.nodes["elevator_f1"]
            pose_values = args_cli.corridor_lobby_elevator_pose if args_cli.spawn_corridor_lobby else args_cli.near_field_elevator_pose
            updated_pose = Pose2D(
                pose_values[0],
                pose_values[1],
                pose_values[2],
            )
            graph.update_node(
                SemanticNode(
                    node_id=old_node.node_id,
                    floor=old_node.floor,
                    kind=old_node.kind,
                    pose=updated_pose,
                    label=old_node.label,
                    attrs=old_node.attrs,
                )
            )
            scene_label = "corridor-lobby" if args_cli.spawn_corridor_lobby else "near-field"
            print(
                f"[semantic_nav:lab] {scene_label} elevator_f1 pose=({updated_pose.x:.2f},{updated_pose.y:.2f},{updated_pose.yaw:.2f})",
                flush=True,
            )
        if args_cli.start_pose_override is not None:
            old_node = graph.nodes[args_cli.start]
            pose_values = args_cli.start_pose_override
            updated_pose = Pose2D(pose_values[0], pose_values[1], pose_values[2])
            graph.update_node(
                SemanticNode(
                    node_id=old_node.node_id,
                    floor=old_node.floor,
                    kind=old_node.kind,
                    pose=updated_pose,
                    label=old_node.label,
                    attrs=old_node.attrs,
                )
            )
            print(
                f"[semantic_nav:lab] start {args_cli.start} pose override=({updated_pose.x:.2f},{updated_pose.y:.2f},{updated_pose.yaw:.2f})",
                flush=True,
            )
        start_node = graph.nodes[args_cli.start]
        elevator_node = graph.nodes.get("elevator_f1")
        if args_cli.detect_during_motion or args_cli.blind_find_elevator:
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
                    collision=not args_cli.spawn_corridor_lobby,
                )
                env.sim.render()
                print(
                    f"[semantic_nav:lab] spawned static elevator visual nodes={','.join(sorted(elevator_poses))}",
                    flush=True,
                )
        corridor_lobby_obstacles: list[tuple[float, float, float, float]] = []
        if args_cli.spawn_blind_search_arena:
            spawn_blind_search_arena(
                origin=env.scene.env_origins[0],
                center=Pose2D(args_cli.blind_arena_center[0], args_cli.blind_arena_center[1], 0.0),
                size=(args_cli.blind_arena_size[0], args_cli.blind_arena_size[1]),
            )
            env.sim.render()
            print(
                f"[semantic_nav:lab] spawned blind search arena center={tuple(args_cli.blind_arena_center)} "
                f"size={tuple(args_cli.blind_arena_size)}",
                flush=True,
            )
        if args_cli.spawn_corridor_lobby:
            corridor_lobby_obstacles = spawn_corridor_lobby_walls(origin=env.scene.env_origins[0])
            env.sim.render()
            print(
                "[semantic_nav:lab] spawned corridor-lobby walls "
                f"obstacles={len(corridor_lobby_obstacles)}",
                flush=True,
            )
        if args_cli.spawn_occluder_wall:
            spawn_rect_wall(
                origin=env.scene.env_origins[0],
                name="initial_elevator_occluder",
                center=Pose2D(args_cli.occluder_wall_center[0], args_cli.occluder_wall_center[1], 0.0),
                size=(args_cli.occluder_wall_size[0], args_cli.occluder_wall_size[1]),
                height=args_cli.occluder_wall_height,
            )
            env.sim.render()
            print(
                f"[semantic_nav:lab] spawned occluder wall center={tuple(args_cli.occluder_wall_center)} "
                f"size={tuple(args_cli.occluder_wall_size)} height={args_cli.occluder_wall_height}",
                flush=True,
            )
        center_pillar_center = (
            tuple(args_cli.center_pillar_center)
            if args_cli.center_pillar_center is not None
            else tuple(args_cli.blind_arena_center)
        )
        if args_cli.spawn_center_pillar:
            spawn_rect_wall(
                origin=env.scene.env_origins[0],
                name="center_pillar",
                center=Pose2D(center_pillar_center[0], center_pillar_center[1], 0.0),
                size=(args_cli.center_pillar_size[0], args_cli.center_pillar_size[1]),
                height=args_cli.center_pillar_height,
            )
            env.sim.render()
            print(
                f"[semantic_nav:lab] spawned center pillar center={center_pillar_center} "
                f"size={tuple(args_cli.center_pillar_size)} height={args_cli.center_pillar_height}",
                flush=True,
            )
        if args_cli.spawn_planner_eval_obstacles:
            planner_eval_obstacles = spawn_planner_eval_obstacles(
                origin=env.scene.env_origins[0],
                profile=args_cli.planner_eval_obstacle_profile,
            )
            corridor_lobby_obstacles.extend(planner_eval_obstacles)
            env.sim.render()
            print(
                "[semantic_nav:lab] spawned planner-eval style obstacles "
                f"profile={args_cli.planner_eval_obstacle_profile} obstacles={len(planner_eval_obstacles)}",
                flush=True,
            )
        spawn_preplan_elevator = (args_cli.spawn_lab_elevator or args_cli.use_lab_camera) and not args_cli.detect_during_motion
        if elevator_node is not None and spawn_preplan_elevator:
            spawn_minimal_elevator_scene(
                origin=env.scene.env_origins[0],
                elevator_pose=elevator_node.pose,
                device=env.device,
            )
            env.sim.render()
            print("[semantic_nav:lab] spawned minimal elevator visual scene", flush=True)
        image_jpeg_b64 = None
        if args_cli.image is not None:
            image_jpeg_b64 = base64.b64encode(args_cli.image.read_bytes()).decode("utf-8")
        if args_cli.use_lab_camera:
            if elevator_node is None:
                raise RuntimeError("use_lab_camera requires an elevator_f1 node in the semantic graph")
            image_jpeg_b64 = capture_node_camera_b64(
                env=env,
                node_pose=elevator_node.pose,
                image_path=args_cli.lab_camera_image_out,
            )
        uses_external_detector = args_cli.detector in ("http_client", "apexnav_gdino", "apexnav_yolov7")
        if uses_external_detector:
            print(
                f"[semantic_nav:lab] detector={args_cli.detector} endpoint={args_cli.perception_endpoint or 'default'} "
                f"image={str(args_cli.image) if args_cli.image is not None else 'lab_camera' if args_cli.use_lab_camera else 'None'} "
                f"log_detections={args_cli.log_detections}",
                flush=True,
            )
        planning_detector_kind = args_cli.detector
        if args_cli.detect_during_motion and image_jpeg_b64 is None and uses_external_detector:
            planning_detector_kind = "graph"
            print(
                "[semantic_nav:lab] using graph detector for initial A* plan; external detector runs during motion",
                flush=True,
            )
        print("[semantic_nav:debug] creating LabRobotAdapter", flush=True)
        robot = LabRobotAdapter(env)
        print("[semantic_nav:debug] resetting robot to start pose", flush=True)
        robot.reset(start_node.pose)
        print("[semantic_nav:debug] robot reset complete", flush=True)
        recorder = None
        if args_cli.record_run_dir is not None:
            recorder_cls = ViewportRunRecorder if args_cli.record_viewport else LabRunRecorder
            recorder = recorder_cls(
                env=env,
                cfg=LabRunRecorderConfig(
                    out_dir=args_cli.record_run_dir,
                    every=args_cli.record_every,
                    resolution=(args_cli.record_resolution[0], args_cli.record_resolution[1]),
                    top_center=(args_cli.record_top_center[0], args_cli.record_top_center[1]),
                    top_height=args_cli.record_top_height,
                ),
            )
            print(f"[semantic_nav:record] frame_dir={args_cli.record_run_dir}", flush=True)
        executor_cfg = WaypointExecutorConfig(
            xy_tolerance=args_cli.xy_tolerance,
            max_vx=args_cli.max_vx,
            max_vy=args_cli.max_vy,
            max_wz=args_cli.max_wz,
        )
        grid_planner = None
        se2_planner = None
        exploration_strategy = None
        if args_cli.spawn_blind_search_arena:
            grid_bounds = GridBounds(
                center_x=args_cli.blind_arena_center[0],
                center_y=args_cli.blind_arena_center[1],
                size_x=args_cli.blind_arena_size[0],
                size_y=args_cli.blind_arena_size[1],
            )
            grid_planner = OccupancyGridAStar(
                GridAStarConfig(
                    bounds=grid_bounds
                )
            )
            if args_cli.grid_planner == "se2":
                se2_planner = HumanoidSE2AStar(
                    HumanoidSE2AStarConfig(
                        grid=grid_planner,
                        yaw_bins=args_cli.se2_yaw_bins,
                        step_distance=args_cli.se2_step_distance,
                        output_min_spacing=args_cli.se2_output_min_spacing,
                        output_yaw_threshold=args_cli.se2_output_yaw_threshold,
                    )
                )
            if args_cli.spawn_occluder_wall:
                grid_planner.add_rect_obstacle(
                    center_x=args_cli.occluder_wall_center[0],
                    center_y=args_cli.occluder_wall_center[1],
                    size_x=args_cli.occluder_wall_size[0],
                    size_y=args_cli.occluder_wall_size[1],
                )
            if args_cli.spawn_center_pillar:
                grid_planner.add_rect_obstacle(
                    center_x=center_pillar_center[0],
                    center_y=center_pillar_center[1],
                    size_x=args_cli.center_pillar_size[0],
                    size_y=args_cli.center_pillar_size[1],
                )
            for center_x, center_y, size_x, size_y in corridor_lobby_obstacles:
                grid_planner.add_rect_obstacle(
                    center_x=center_x,
                    center_y=center_y,
                    size_x=size_x,
                    size_y=size_y,
                )
            if args_cli.adaptive_exploration:
                prefix_viewpoints = None
                if args_cli.spawn_corridor_lobby:
                    prefix_viewpoints = [
                        ExplorationViewpoint(Pose2D(2.5, 0.0, 0.0), "corridor_mid"),
                        ExplorationViewpoint(Pose2D(5.6, 0.0, 0.0), "corridor_exit"),
                        ExplorationViewpoint(Pose2D(7.4, 1.4, 0.0), "lobby_entry"),
                    ]
                exploration_strategy = AdaptiveExplorationStrategy(
                    grid_bounds,
                    floor=args_cli.blind_floor,
                    spacing=args_cli.exploration_spacing,
                    wall_margin=args_cli.exploration_wall_margin,
                    prefix_viewpoints=prefix_viewpoints,
                )
        if args_cli.blind_find_elevator or args_cli.blind_find_object:
            print("[semantic_nav:debug] blind branch: creating task parser", flush=True)
            task_parser = make_task_parser_from_args(args_cli)
            print("[semantic_nav:debug] blind branch: parsing task", flush=True)
            target_node_id = args_cli.search_node_id or normalize_target_node_id(args_cli.target)
            parsed_task = task_parser.parse(
                args_cli.goal,
                current_floor=start_node.floor,
                graph=graph,
                start_node_id=args_cli.start,
            )
            print("[semantic_nav:debug] blind branch: task parsed", flush=True)
            if target_node_id is None:
                target_node_id = parsed_task.target_node_id
            if args_cli.blind_find_object:
                if args_cli.search_label is None and parsed_task.search_label is not None:
                    args_cli.search_label = parsed_task.search_label
                if args_cli.search_prompts is None and parsed_task.search_prompts:
                    args_cli.search_prompts = ".".join(parsed_task.search_prompts)
            print("[semantic_nav:lab] task:", parsed_task.goal.raw_text, flush=True)
            print("[semantic_nav:lab] task parser:", args_cli.task_parser, flush=True)
            print("[semantic_nav:lab] parsed intent:", parsed_task.goal.intent, flush=True)
            print("[semantic_nav:lab] target floor:", parsed_task.goal.target_floor, flush=True)
            print("[semantic_nav:lab] target node:", target_node_id or "auto", flush=True)
            result = run_blind_find_then_astar(
                graph=graph,
                robot=robot,
                planner_cfg=cfg,
                detector_kind=args_cli.detector,
                perception_endpoint=args_cli.perception_endpoint,
                max_steps=args_cli.steps,
                print_every=args_cli.print_every,
                perception_every=args_cli.perception_every,
                image_dir=args_cli.motion_detection_image_dir,
                floor=args_cli.blind_floor,
                cfg=executor_cfg,
                grid_planner=grid_planner,
                se2_planner=se2_planner,
                exploration_strategy=exploration_strategy,
                recorder=recorder,
                target_node_id=target_node_id,
            )
            print(f"[semantic_nav:lab] done success={result.success} steps={result.steps} reason={result.reason}")
            _write_result_json(args_cli.result_json, result, mode="blind_find_object" if args_cli.blind_find_object else "blind_find_elevator")
            if result.confirmed_nodes:
                print(f"[semantic_nav:lab] confirmed nodes: {','.join(result.confirmed_nodes)}")
            if result.perception_events:
                print("[semantic_nav:lab] perception events:")
                for event in result.perception_events[-10:]:
                    print(f"  {event}")
            return

        detector = make_semantic_detector(
            planning_detector_kind,
            graph=graph,
            perception_endpoint=args_cli.perception_endpoint,
            image_jpeg_b64=image_jpeg_b64,
            log_detections=args_cli.log_detections or (uses_external_detector and planning_detector_kind == args_cli.detector),
            min_score=args_cli.perception_min_score,
        )
        task_parser = make_task_parser_from_args(args_cli)
        target_node_id = normalize_target_node_id(args_cli.target)
        task_plan = SemanticTaskPlanner(graph, detector=detector, goal_parser=task_parser).plan(
            args_cli.start,
            args_cli.goal,
            target_node_id,
        )
        execution_steps = build_execution_plan(graph, task_plan)

        executor = _make_executor(
            env=env,
            steps=execution_steps,
            cfg=executor_cfg,
            planner_cfg=cfg,
        )

        print("[semantic_nav:lab] task:", task_plan.goal.raw_text)
        print("[semantic_nav:lab] task parser:", args_cli.task_parser)
        if task_plan.elevator_plan is not None:
            print("[semantic_nav:lab] selected elevator:", task_plan.elevator_plan.elevator_node_id)
        print("[semantic_nav:lab] execution steps:")
        for idx, step in enumerate(execution_steps):
            if step.kind == "floor_transition":
                print(f"  {idx:02d}. FLOOR_TRANSITION {step.node_id} -> {step.dst_node_id}")
            else:
                print(f"  {idx:02d}. WALK_TO {step.node_id} floor={step.floor} pose=({step.pose.x:.2f},{step.pose.y:.2f},{step.pose.yaw:.2f})")

        motion_perception_hook = None
        if args_cli.detect_during_motion:
            def motion_perception_hook(step_idx: int, active_floor: str) -> PerceptionHookResult | None:
                if active_floor != args_cli.motion_detection_floor:
                    return None
                image_path = None
                if args_cli.motion_detection_image_dir is not None:
                    args_cli.motion_detection_image_dir.mkdir(parents=True, exist_ok=True)
                    image_path = args_cli.motion_detection_image_dir / f"motion_{step_idx:05d}_{active_floor}.jpg"
                image_b64 = capture_robot_view_camera_b64(env=env, robot_pose=robot.pose(), image_path=image_path)
                runtime_detector = make_semantic_detector(
                    args_cli.detector,
                    graph=graph,
                    perception_endpoint=args_cli.perception_endpoint,
                    image_jpeg_b64=image_b64,
                    log_detections=True,
                    min_score=args_cli.perception_min_score,
                )
                current_node_id = _nearest_node_on_floor(graph, robot.pose(), active_floor)
                detections = runtime_detector.detect(graph, current_node_id=current_node_id)
                selected_nodes = tuple(detection.node_id for detection in detections)
                selected = ",".join(selected_nodes) or "none"
                return PerceptionHookResult(
                    event=f"motion_perception_floor={active_floor} camera=robot_view selected={selected}",
                    selected_node_ids=selected_nodes,
                )

        result = run_semantic_execution_loop(
            graph=graph,
            robot=robot,
            executor=executor,
            max_steps=args_cli.steps,
            print_every=args_cli.print_every,
            active_floor=start_node.floor,
            perception_hook=motion_perception_hook,
            perception_every=args_cli.perception_every if args_cli.detect_during_motion else 0,
            stop_on_detected_node=args_cli.stop_on_detected_node,
            step_hook=(lambda step_idx, pose: recorder.capture(step_idx=step_idx, robot_pose=pose)) if recorder is not None else None,
        )
        print(f"[semantic_nav:lab] done success={result.success} steps={result.steps} reason={result.reason}")
        _write_result_json(args_cli.result_json, result, mode="semantic_execution")
        if result.confirmed_nodes:
            print(f"[semantic_nav:lab] confirmed nodes: {','.join(result.confirmed_nodes)}")
        if result.perception_events:
            print("[semantic_nav:lab] perception events:")
            for event in result.perception_events[-10:]:
                print(f"  {event}")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
