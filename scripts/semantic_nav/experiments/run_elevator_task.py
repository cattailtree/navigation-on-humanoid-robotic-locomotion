from __future__ import annotations

import argparse
import sys
from pathlib import Path


SEMANTIC_NAV_ROOT = Path(__file__).resolve().parents[1]
if str(SEMANTIC_NAV_ROOT) not in sys.path:
    sys.path.insert(0, str(SEMANTIC_NAV_ROOT))

from envs.abstract_building_env import DEFAULT_BUILDING_CONFIG, load_semantic_graph
from llm.factory import add_task_parser_args, make_task_parser_from_args, normalize_target_node_id
from planners.execution_plan import build_execution_plan
from perception.factory import make_semantic_detector
from planners.semantic_task_planner import SemanticTaskPlanner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a full abstract elevator semantic navigation task.")
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
    args = parser.parse_args()

    graph = load_semantic_graph(args.building_config)
    detector = make_semantic_detector(args.detector, graph=graph, perception_endpoint=args.perception_endpoint)
    task_parser = make_task_parser_from_args(args)
    target_node_id = normalize_target_node_id(args.target)
    plan = SemanticTaskPlanner(graph, detector=detector, goal_parser=task_parser).plan(
        args.start,
        args.goal,
        target_node_id,
    )

    print("[semantic_nav] task:", plan.goal.raw_text)
    print("[semantic_nav] task parser:", args.task_parser)
    print("[semantic_nav] parsed intent:", plan.goal.intent)
    print("[semantic_nav] target floor:", plan.goal.target_floor)
    if plan.elevator_plan is not None:
        print("[semantic_nav] selected elevator:", plan.elevator_plan.elevator_node_id)

    transition_event = plan.transition_event
    if transition_event is not None:
        print("[semantic_nav] transition event:", f"{transition_event[0]} -> {transition_event[1]}")

    for idx, path in enumerate(plan.paths):
        print(f"[semantic_nav] path {idx} cost:", f"{path.cost:.2f}")
    print("[semantic_nav] full semantic route:")
    for idx, node_id in enumerate(plan.full_node_sequence):
        node = graph.nodes[node_id]
        print(
            f"  {idx:02d}. {node_id}: floor={node.floor}, kind={node.kind}, "
            f"pose=({node.pose.x:.2f}, {node.pose.y:.2f}, {node.pose.yaw:.2f})"
        )

    execution_steps = build_execution_plan(graph, plan)
    print("[semantic_nav] execution steps:")
    for idx, step in enumerate(execution_steps):
        if step.kind == "floor_transition":
            print(f"  {idx:02d}. FLOOR_TRANSITION {step.node_id} -> {step.dst_node_id}")
        else:
            print(
                f"  {idx:02d}. WALK_TO {step.node_id}: floor={step.floor}, "
                f"pose=({step.pose.x:.2f}, {step.pose.y:.2f}, {step.pose.yaw:.2f})"
            )


if __name__ == "__main__":
    main()
