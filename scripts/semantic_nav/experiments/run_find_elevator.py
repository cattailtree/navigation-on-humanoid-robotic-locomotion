from __future__ import annotations

import argparse
import sys
from pathlib import Path


SEMANTIC_NAV_ROOT = Path(__file__).resolve().parents[1]
if str(SEMANTIC_NAV_ROOT) not in sys.path:
    sys.path.insert(0, str(SEMANTIC_NAV_ROOT))

from envs.abstract_building_env import DEFAULT_BUILDING_CONFIG, load_semantic_graph
from llm.factory import add_task_parser_args, make_task_parser_from_args
from perception.factory import make_semantic_detector
from planners.elevator_planner import ElevatorPlanner


def main() -> None:
    parser = argparse.ArgumentParser(description="Find an elevator for a downstairs semantic navigation task.")
    parser.add_argument("--building-config", type=Path, default=DEFAULT_BUILDING_CONFIG)
    parser.add_argument("--start", default="start_f1", help="Start semantic node id.")
    parser.add_argument("--goal", default="go downstairs", help="Natural-language style task string.")
    add_task_parser_args(parser)
    parser.add_argument(
        "--detector",
        choices=("apexnav", "graph", "dummy_client", "http_client", "apexnav_gdino", "apexnav_yolov7"),
        default="apexnav",
    )
    parser.add_argument("--perception-endpoint", default=None, help="HTTP endpoint for external YOLO/GroundingDINO detections.")
    args = parser.parse_args()

    graph = load_semantic_graph(args.building_config)
    start_node = graph.nodes[args.start]
    task_parser = make_task_parser_from_args(args)
    parsed_task = task_parser.parse(
        args.goal,
        current_floor=start_node.floor,
        graph=graph,
        start_node_id=args.start,
    )
    goal = parsed_task.goal
    detector = make_semantic_detector(args.detector, graph=graph, perception_endpoint=args.perception_endpoint)
    detections = detector.detect(graph, current_node_id=args.start)
    plan = ElevatorPlanner(graph, detector=detector).plan_to_elevator(args.start, goal)

    print("[semantic_nav] task:", goal.raw_text)
    print("[semantic_nav] task parser:", args.task_parser)
    print("[semantic_nav] detector:", args.detector)
    print("[semantic_nav] detections:")
    for detection in detections:
        print(f"  - {detection.node_id}: label={detection.label}, score={detection.score:.3f}")
    print("[semantic_nav] parsed intent:", goal.intent)
    print("[semantic_nav] target floor:", goal.target_floor)
    print("[semantic_nav] selected elevator:", plan.elevator_node_id)
    print("[semantic_nav] transition target:", plan.transition_node_id)
    print("[semantic_nav] approach cost:", f"{plan.approach_path.cost:.2f}")
    print("[semantic_nav] subgoals:")
    for node_id in plan.subgoal_node_ids:
        node = graph.nodes[node_id]
        print(f"  - {node_id}: floor={node.floor}, kind={node.kind}, pose=({node.pose.x:.2f}, {node.pose.y:.2f}, {node.pose.yaw:.2f})")


if __name__ == "__main__":
    main()
