from __future__ import annotations

import argparse
import sys
from pathlib import Path


SEMANTIC_NAV_ROOT = Path(__file__).resolve().parents[1]
if str(SEMANTIC_NAV_ROOT) not in sys.path:
    sys.path.insert(0, str(SEMANTIC_NAV_ROOT))

from envs.abstract_building_env import DEFAULT_BUILDING_CONFIG, load_semantic_graph
from executors.waypoint_executor import WaypointExecutor, advance_abstract_pose
from llm.factory import add_task_parser_args, make_task_parser_from_args, normalize_target_node_id
from perception.factory import make_semantic_detector
from planners.execution_plan import build_execution_plan
from planners.semantic_task_planner import SemanticTaskPlanner


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test semantic elevator steps with an abstract waypoint follower.")
    parser.add_argument("--building-config", type=Path, default=DEFAULT_BUILDING_CONFIG)
    parser.add_argument("--start", default="start_f1", help="Start semantic node id.")
    parser.add_argument("--goal", default="go downstairs to target room", help="Natural-language style task string.")
    parser.add_argument("--target", default="target_room_b1", help="Target semantic node id after floor transition.")
    add_task_parser_args(parser)
    parser.add_argument(
        "--detector",
        choices=("apexnav", "graph", "dummy_client", "http_client", "apexnav_gdino", "apexnav_yolov7"),
        default="apexnav",
    )
    parser.add_argument("--perception-endpoint", default=None, help="HTTP endpoint for external YOLO/GroundingDINO detections.")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()

    graph = load_semantic_graph(args.building_config)
    start_node = graph.nodes[args.start]
    detector = make_semantic_detector(args.detector, graph=graph, perception_endpoint=args.perception_endpoint)
    task_parser = make_task_parser_from_args(args)
    target_node_id = normalize_target_node_id(args.target)
    task_plan = SemanticTaskPlanner(graph, detector=detector, goal_parser=task_parser).plan(
        args.start,
        args.goal,
        target_node_id,
    )
    execution_steps = build_execution_plan(graph, task_plan)

    robot_pose = start_node.pose
    executor = WaypointExecutor(execution_steps)
    active_floor = start_node.floor

    print("[semantic_nav] abstract rollout start")
    print("[semantic_nav] task parser:", args.task_parser)
    print("[semantic_nav] start:", args.start, f"floor={active_floor}", f"pose=({robot_pose.x:.2f}, {robot_pose.y:.2f}, {robot_pose.yaw:.2f})")

    for step_idx in range(args.max_steps):
        command, status = executor.update(robot_pose)
        if status.event:
            print(f"[semantic_nav] step={step_idx:04d} event={status.event}")
            if status.event.startswith("floor transition") and status.active_step is not None:
                dst = status.active_step.dst_node_id
                if dst is not None:
                    dst_node = graph.nodes[dst]
                    robot_pose = dst_node.pose
                    active_floor = dst_node.floor
                    print(
                        f"[semantic_nav] transition result: floor={active_floor}, "
                        f"pose=({robot_pose.x:.2f}, {robot_pose.y:.2f}, {robot_pose.yaw:.2f})"
                    )

        if status.done:
            print("[semantic_nav] success: reached final target")
            print("[semantic_nav] final floor:", active_floor)
            print("[semantic_nav] final pose:", f"({robot_pose.x:.2f}, {robot_pose.y:.2f}, {robot_pose.yaw:.2f})")
            return

        robot_pose = advance_abstract_pose(robot_pose, command, args.dt)

    print("[semantic_nav] timeout")
    print("[semantic_nav] final floor:", active_floor)
    print("[semantic_nav] final pose:", f"({robot_pose.x:.2f}, {robot_pose.y:.2f}, {robot_pose.yaw:.2f})")


if __name__ == "__main__":
    main()
