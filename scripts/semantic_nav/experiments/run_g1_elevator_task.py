from __future__ import annotations

import argparse
from pathlib import Path
import sys


SEMANTIC_NAV_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = SEMANTIC_NAV_ROOT.parent
SIM2SIM_ROOT = SCRIPTS_ROOT / "mujoco_sim2sim"
for path in (SEMANTIC_NAV_ROOT, SIM2SIM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import Sim2SimConfig  # noqa: E402
from envs.abstract_building_env import DEFAULT_BUILDING_CONFIG, load_semantic_graph  # noqa: E402
from executors.semantic_execution_loop import run_semantic_execution_loop  # noqa: E402
from executors.sim2sim_robot_adapter import Sim2SimRobotAdapter  # noqa: E402
from executors.waypoint_executor import WaypointExecutor, WaypointExecutorConfig  # noqa: E402
from llm.factory import add_task_parser_args, make_task_parser_from_args, normalize_target_node_id  # noqa: E402
from low_level_controller import MissingGaitController, TorchPolicyPDController, ZeroTorqueController  # noqa: E402
from mujoco_g1_env import MissingMuJoCoError  # noqa: E402
from perception.factory import make_semantic_detector  # noqa: E402
from planners.execution_plan import build_execution_plan  # noqa: E402
from planners.semantic_task_planner import SemanticTaskPlanner  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the semantic single-elevator task with MuJoCo G1 gait control.")
    parser.add_argument("--root", type=Path, default=Sim2SimConfig.root)
    parser.add_argument("--g1-xml", type=Path, default=Sim2SimConfig.g1_xml)
    parser.add_argument("--low-level-policy", type=Path, default=Sim2SimConfig.low_level_policy)
    parser.add_argument("--building-config", type=Path, default=DEFAULT_BUILDING_CONFIG)
    parser.add_argument("--start", default="start_f1")
    parser.add_argument("--goal", default="go downstairs to target room")
    parser.add_argument("--target", default="target_room_b1")
    add_task_parser_args(parser)
    parser.add_argument(
        "--detector",
        choices=("apexnav", "graph", "dummy_client", "http_client", "apexnav_gdino", "apexnav_yolov7"),
        default="apexnav",
    )
    parser.add_argument("--perception-endpoint", default=None, help="HTTP endpoint for external YOLO/GroundingDINO detections.")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--control-decimation", type=int, default=Sim2SimConfig.control_decimation)
    parser.add_argument("--physics-dt", type=float, default=Sim2SimConfig.physics_dt)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--zero-controller", action="store_true", help="Run with zero torques for wiring checks.")
    parser.add_argument("--policy-device", type=str, default="cpu")
    parser.add_argument("--policy-obs-dim", type=int, default=495)
    parser.add_argument("--policy-action-dim", type=int, default=29)
    parser.add_argument("--policy-history", type=int, default=5)
    parser.add_argument("--policy-inference-decimation", type=int, default=4)
    parser.add_argument("--allow-partial-policy-joints", action="store_true")
    parser.add_argument("--policy-action-clip", type=float, default=10.0)
    parser.add_argument("--g1-obs-layout", choices=("auto", "g1_nav_96", "g1_policy_99"), default="g1_policy_99")
    parser.add_argument(
        "--obs-axis-transform",
        choices=("identity", "rot_x_pos_90", "rot_x_neg_90", "swap_yz", "swap_yz_neg"),
        default="identity",
    )
    parser.add_argument("--xy-tolerance", type=float, default=0.45)
    parser.add_argument("--max-vx", type=float, default=0.45)
    parser.add_argument("--max-vy", type=float, default=0.08)
    parser.add_argument("--max-wz", type=float, default=0.66)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing = _missing_inputs(args)
    if missing:
        print("[semantic_nav:g1] Missing inputs:")
        for item in missing:
            print(f"  - {item}")
        if args.check_only:
            return
    elif args.check_only:
        print("[semantic_nav:g1] All configured input paths exist.")
        return

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

    controller = _build_controller(args)
    try:
        robot = Sim2SimRobotAdapter(
            xml_path=args.g1_xml,
            controller=controller,
            physics_dt=args.physics_dt,
            control_decimation=args.control_decimation,
        )
    except MissingMuJoCoError as exc:
        print(f"[semantic_nav:g1] {exc}")
        return

    robot.reset(start_node.pose)

    executor = WaypointExecutor(
        execution_steps,
        WaypointExecutorConfig(
            xy_tolerance=args.xy_tolerance,
            max_vx=args.max_vx,
            max_vy=args.max_vy,
            max_wz=args.max_wz,
        ),
    )

    print("[semantic_nav:g1] loaded:", args.g1_xml)
    print("[semantic_nav:g1] task:", task_plan.goal.raw_text)
    print("[semantic_nav:g1] task parser:", args.task_parser)
    if task_plan.elevator_plan is not None:
        print("[semantic_nav:g1] selected elevator:", task_plan.elevator_plan.elevator_node_id)
    print("[semantic_nav:g1] execution steps:")
    for idx, step in enumerate(execution_steps):
        if step.kind == "floor_transition":
            print(f"  {idx:02d}. FLOOR_TRANSITION {step.node_id} -> {step.dst_node_id}")
        else:
            print(f"  {idx:02d}. WALK_TO {step.node_id} floor={step.floor} pose=({step.pose.x:.2f},{step.pose.y:.2f},{step.pose.yaw:.2f})")

    result = run_semantic_execution_loop(
        graph=graph,
        robot=robot,
        executor=executor,
        max_steps=args.steps,
        print_every=args.print_every,
        active_floor=start_node.floor,
    )
    print(f"[semantic_nav:g1] done success={result.success} steps={result.steps} reason={result.reason}")


def _build_controller(args: argparse.Namespace):
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


def _missing_inputs(args: argparse.Namespace) -> list[str]:
    missing: list[str] = []
    try:
        import mujoco  # noqa: F401
    except ImportError:
        missing.append("Python package `mujoco` in the active environment")
    if not args.g1_xml.exists():
        missing.append(f"G1 MuJoCo XML: {args.g1_xml}")
    if not args.zero_controller and (args.low_level_policy is None or not args.low_level_policy.exists()):
        missing.append(f"G1 low-level policy file: {args.low_level_policy}")
    return missing


if __name__ == "__main__":
    main()
