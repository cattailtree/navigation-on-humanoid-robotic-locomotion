from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path as RosPath
from rclpy.node import Node
from std_msgs.msg import Bool, String

from semantic_nav_ros.ros_utils import (
    add_semantic_nav_to_syspath,
    make_path_msg,
    quaternion_from_yaw,
    repo_root_from_package,
    twist_from_velocity,
    yaw_from_quaternion,
    zero_twist,
)


add_semantic_nav_to_syspath()

from envs.abstract_building_env import DEFAULT_BUILDING_CONFIG, load_semantic_graph  # noqa: E402
from executors.waypoint_executor import WaypointExecutor, WaypointExecutorConfig  # noqa: E402
from llm.factory import make_task_parser, normalize_target_node_id  # noqa: E402
from maps.semantic_graph import Pose2D, SemanticGraph  # noqa: E402
from perception.factory import make_semantic_detector  # noqa: E402
from planners.execution_plan import ExecutionStep, build_execution_plan  # noqa: E402
from planners.semantic_task_planner import SemanticTaskPlanner  # noqa: E402


@dataclass(frozen=True)
class ActiveTask:
    goal: str
    target_node_id: str | None
    steps: list[ExecutionStep]
    executor: WaypointExecutor
    mode: str = "graph_goal"
    search_label: str | None = None
    search_prompts: tuple[str, ...] = ()


class SemanticNavNode(Node):
    """ROS2 bridge for graph-level semantic navigation and local velocity output."""

    def __init__(self) -> None:
        super().__init__("semantic_nav_node")

        self._declare_parameters()
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.cmd_topic = str(self.get_parameter("cmd_vel_raw_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.path_topic = str(self.get_parameter("path_topic").value)
        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.start_node_id = str(self.get_parameter("start_node_id").value)
        self.default_target_node_id = normalize_target_node_id(str(self.get_parameter("default_target_node_id").value))
        self.default_goal = str(self.get_parameter("default_goal").value)
        self.planner_backend = str(self.get_parameter("planner_backend").value)
        self.autostart = bool(self.get_parameter("autostart").value)
        self._paused = False
        self._current_pose: Pose2D | None = None
        self._active_task: ActiveTask | None = None
        self._detection_confirm_count = 0
        self._last_status = "idle"

        graph_path = self._resolve_repo_path(str(self.get_parameter("building_config").value))
        self.graph = load_semantic_graph(graph_path)
        self.task_parser = make_task_parser(
            str(self.get_parameter("task_parser").value),
            endpoint=_none_if_empty(str(self.get_parameter("llm_endpoint").value)),
            model=_none_if_empty(str(self.get_parameter("llm_model").value)),
            api_key_env=str(self.get_parameter("llm_api_key_env").value),
            timeout_s=float(self.get_parameter("llm_timeout_s").value),
            log_raw=bool(self.get_parameter("log_llm").value),
        )
        self.detector = make_semantic_detector(
            str(self.get_parameter("detector").value),
            graph=self.graph,
            perception_endpoint=_none_if_empty(str(self.get_parameter("perception_endpoint").value)),
            min_score=float(self.get_parameter("perception_min_score").value),
            log_detections=bool(self.get_parameter("log_detections").value),
        )

        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.path_pub = self.create_publisher(RosPath, self.path_topic, 10)
        self.prompts_pub = self.create_publisher(String, str(self.get_parameter("search_prompts_topic").value), 10)
        self.fdm_goal_pub = self.create_publisher(PoseStamped, str(self.get_parameter("fdm_goal_topic").value), 10)
        self.create_subscription(Odometry, self.odom_topic, self._on_odom, 20)
        self.create_subscription(String, str(self.get_parameter("goal_topic").value), self._on_goal, 10)
        self.create_subscription(String, str(self.get_parameter("target_node_topic").value), self._on_target_node, 10)
        self.create_subscription(Bool, str(self.get_parameter("pause_topic").value), self._on_pause, 10)
        self.create_subscription(String, str(self.get_parameter("detections_json_topic").value), self._on_detections, 10)

        self.timer = self.create_timer(1.0 / max(self.control_rate_hz, 1.0), self._tick)
        self._publish_status("ready")

        if self.planner_backend != "waypoint":
            self.get_logger().warn(
                f"planner_backend={self.planner_backend!r} requested, but this ROS2 node currently runs "
                "the deploy-safe waypoint backend. FDM-MPPI requires a real observation bridge first."
            )
        if self.autostart:
            self._start_task(self.default_goal, self.default_target_node_id)

    def _declare_parameters(self) -> None:
        default_config = repo_root_from_package() / "scripts" / "semantic_nav" / "configs" / "single_elevator_building.json"
        self.declare_parameter("building_config", str(default_config if default_config.exists() else DEFAULT_BUILDING_CONFIG))
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("cmd_vel_raw_topic", "/semantic_nav/cmd_vel_raw")
        self.declare_parameter("goal_topic", "/semantic_nav/goal")
        self.declare_parameter("target_node_topic", "/semantic_nav/target_node")
        self.declare_parameter("pause_topic", "/semantic_nav/pause")
        self.declare_parameter("status_topic", "/semantic_nav/status")
        self.declare_parameter("path_topic", "/semantic_nav/path")
        self.declare_parameter("fdm_goal_topic", "/semantic_nav/fdm_goal")
        self.declare_parameter("publish_fdm_goal", True)
        self.declare_parameter("search_prompts_topic", "/semantic_nav/search_prompts")
        self.declare_parameter("detections_json_topic", "/semantic_nav/detections_json")
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("start_node_id", "start_f1")
        self.declare_parameter("default_goal", "find the elevator")
        self.declare_parameter("default_target_node_id", "auto")
        self.declare_parameter("autostart", False)
        self.declare_parameter("planner_backend", "waypoint")
        self.declare_parameter("task_parser", "rule")
        self.declare_parameter("llm_endpoint", "")
        self.declare_parameter("llm_model", "")
        self.declare_parameter("llm_api_key_env", "SEMANTIC_NAV_LLM_API_KEY")
        self.declare_parameter("llm_timeout_s", 20.0)
        self.declare_parameter("log_llm", False)
        self.declare_parameter("detector", "graph")
        self.declare_parameter("perception_endpoint", "")
        self.declare_parameter("perception_min_score", 0.65)
        self.declare_parameter("log_detections", False)
        self.declare_parameter("open_set_min_score", 0.65)
        self.declare_parameter("open_set_confirmations", 2)
        self.declare_parameter("exploration_node_ids", ["corridor_f1", "room_f1", "elevator_f1"])
        self.declare_parameter("xy_tolerance", 0.35)
        self.declare_parameter("yaw_tolerance", 0.35)
        self.declare_parameter("require_yaw_alignment", False)
        self.declare_parameter("max_vx", 0.35)
        self.declare_parameter("max_vy", 0.05)
        self.declare_parameter("max_wz", 0.55)
        self.declare_parameter("k_vx", 0.8)
        self.declare_parameter("k_vy", 0.5)
        self.declare_parameter("k_wz", 1.2)
        self.declare_parameter("slow_radius", 1.0)

    def _resolve_repo_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return repo_root_from_package() / path

    def _on_odom(self, msg: Odometry) -> None:
        pos = msg.pose.pose.position
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self._current_pose = Pose2D(float(pos.x), float(pos.y), float(yaw))

    def _on_goal(self, msg: String) -> None:
        goal = msg.data.strip()
        if not goal:
            self._stop("empty_goal")
            return
        self._start_task(goal, self.default_target_node_id)

    def _on_target_node(self, msg: String) -> None:
        target = normalize_target_node_id(msg.data)
        self._start_task(self.default_goal, target)

    def _on_pause(self, msg: Bool) -> None:
        self._paused = bool(msg.data)
        if self._paused:
            self.cmd_pub.publish(zero_twist())
            self._publish_status("paused")
        else:
            self._publish_status("running" if self._active_task else "ready")

    def _start_task(self, goal: str, target_node_id: str | None) -> None:
        try:
            start_node = self.graph.nodes[self.start_node_id]
            parsed = self.task_parser.parse(
                goal,
                current_floor=start_node.floor,
                graph=self.graph,
                start_node_id=self.start_node_id,
            )
            effective_target = target_node_id or parsed.target_node_id
            if parsed.goal.intent == "open_set_object_search" and effective_target is None:
                steps = self._build_exploration_steps(start_node.floor)
                mode = "open_set_search"
                search_label = parsed.search_label or parsed.goal.target_label or goal
                search_prompts = parsed.search_prompts or (search_label,)
            else:
                plan = SemanticTaskPlanner(self.graph, detector=self.detector, goal_parser=self.task_parser).plan(
                    self.start_node_id,
                    goal,
                    effective_target,
                )
                steps = build_execution_plan(self.graph, plan)
                mode = "graph_goal"
                search_label = None
                search_prompts = ()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"failed to plan goal={goal!r}: {exc}")
            self._stop(f"plan_failed:{exc}")
            return

        if not steps:
            self.get_logger().warn(f"goal={goal!r} produced an empty execution plan")
            self._stop("empty_plan")
            return

        cfg = WaypointExecutorConfig(
            xy_tolerance=float(self.get_parameter("xy_tolerance").value),
            yaw_tolerance=float(self.get_parameter("yaw_tolerance").value),
            require_yaw_alignment=bool(self.get_parameter("require_yaw_alignment").value),
            max_vx=float(self.get_parameter("max_vx").value),
            max_vy=float(self.get_parameter("max_vy").value),
            max_wz=float(self.get_parameter("max_wz").value),
            k_vx=float(self.get_parameter("k_vx").value),
            k_vy=float(self.get_parameter("k_vy").value),
            k_wz=float(self.get_parameter("k_wz").value),
            slow_radius=float(self.get_parameter("slow_radius").value),
        )
        executor = WaypointExecutor(steps, cfg)
        self._active_task = ActiveTask(
            goal=goal,
            target_node_id=target_node_id,
            steps=steps,
            executor=executor,
            mode=mode,
            search_label=search_label,
            search_prompts=tuple(search_prompts),
        )
        self._detection_confirm_count = 0
        if search_prompts:
            prompts_msg = String()
            prompts_msg.data = ",".join(search_prompts)
            self.prompts_pub.publish(prompts_msg)
        self.path_pub.publish(self._path_msg_for_steps(steps))
        self.get_logger().info(
            f"started goal={goal!r} mode={mode} steps={len(steps)} target={target_node_id or 'auto'}"
        )
        self._publish_status(f"running mode={mode} goal={goal} steps={len(steps)}")

    def _tick(self) -> None:
        if self._paused:
            return
        if self._active_task is None:
            self.cmd_pub.publish(zero_twist())
            return
        if self._current_pose is None:
            self.cmd_pub.publish(zero_twist())
            self._publish_status("waiting_for_odom")
            return

        command, status = self._active_task.executor.update(self._current_pose)
        if bool(self.get_parameter("publish_fdm_goal").value):
            self._publish_fdm_goal(status.active_step)
        self.cmd_pub.publish(twist_from_velocity(command.vx, command.vy, command.wz))

        active = status.active_step.node_id if status.active_step is not None else "done"
        event = status.event or "-"
        self._publish_status(f"running active={active} idx={status.active_step_index} event={event}")
        if status.done:
            if self._active_task.mode == "open_set_search":
                self.get_logger().warn(f"open-set search route exhausted: {self._active_task.goal!r}")
                self._stop("search_exhausted")
            else:
                self.get_logger().info(f"goal complete: {self._active_task.goal!r}")
                self._stop("complete")

    def _stop(self, reason: str) -> None:
        self.cmd_pub.publish(zero_twist())
        self._active_task = None
        self._detection_confirm_count = 0
        self._publish_status(reason)

    def _on_detections(self, msg: String) -> None:
        task = self._active_task
        if task is None or task.mode != "open_set_search":
            return
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn("ignored malformed detections_json")
            return
        detections = payload.get("detections", [])
        if not isinstance(detections, list):
            return
        min_score = float(self.get_parameter("open_set_min_score").value)
        matched = self._match_open_set_detection(detections, task.search_prompts, min_score)
        if matched is None:
            self._detection_confirm_count = 0
            return
        self._detection_confirm_count += 1
        needed = int(self.get_parameter("open_set_confirmations").value)
        label = str(matched.get("label", task.search_label or "target"))
        score = float(matched.get("score", 0.0))
        if self._detection_confirm_count < needed:
            self._publish_status(
                f"pending_detection label={label} score={score:.2f} "
                f"confirm={self._detection_confirm_count}/{needed}"
            )
            return
        self.get_logger().info(f"open-set target detected label={label} score={score:.2f}")
        self._stop(f"detected label={label} score={score:.2f}")

    def _match_open_set_detection(
        self,
        detections: list[Any],
        prompts: tuple[str, ...],
        min_score: float,
    ) -> dict[str, Any] | None:
        prompt_terms = tuple(prompt.strip().lower() for prompt in prompts if prompt.strip())
        best: dict[str, Any] | None = None
        best_score = min_score
        for item in detections:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "")).lower()
            score = float(item.get("score", 0.0))
            if score < best_score:
                continue
            if prompt_terms and not any(term in label or label in term for term in prompt_terms):
                continue
            best = item
            best_score = score
        return best

    def _build_exploration_steps(self, floor: str) -> list[ExecutionStep]:
        configured = [str(item) for item in self.get_parameter("exploration_node_ids").value]
        candidates = [node_id for node_id in configured if node_id in self.graph.nodes]
        if not candidates:
            candidates = [
                node.node_id
                for node in self.graph.nodes.values()
                if node.floor == floor and node.node_id != self.start_node_id and node.kind != "elevator_lobby"
            ]
        steps: list[ExecutionStep] = []
        current = self.start_node_id
        visited_edges: set[tuple[str, str]] = set()
        for candidate in candidates:
            if candidate == current:
                continue
            path = self.graph.shortest_path(
                start=current,
                goal_fn=lambda node, target=candidate: node.node_id == target,
                edge_filter=lambda edge: edge.kind == "walk",
            )
            if path.is_empty:
                self.get_logger().warn(f"skipping unreachable exploration node={candidate}")
                continue
            for node_id in path.node_ids[1:]:
                edge_key = (current, node_id)
                node = self.graph.nodes[node_id]
                steps.append(
                    ExecutionStep(
                        kind="walk_to",
                        node_id=node.node_id,
                        floor=node.floor,
                        pose=node.pose,
                        description=f"explore to {node.node_id}",
                    )
                )
                visited_edges.add(edge_key)
                current = node_id
        return steps

    def _path_msg_for_steps(self, steps: list[ExecutionStep]) -> RosPath:
        poses = [(step.pose.x, step.pose.y, step.pose.yaw) for step in steps if step.kind == "walk_to"]
        return make_path_msg(self, frame_id=self.frame_id, poses=poses)

    def _publish_fdm_goal(self, step: ExecutionStep | None) -> None:
        if step is None or step.kind != "walk_to":
            return
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.pose.position.x = float(step.pose.x)
        msg.pose.position.y = float(step.pose.y)
        msg.pose.orientation = quaternion_from_yaw(step.pose.yaw)
        self.fdm_goal_pub.publish(msg)

    def _publish_status(self, text: str) -> None:
        if text == self._last_status:
            return
        self._last_status = text
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)


def _none_if_empty(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = SemanticNavNode()
    try:
        rclpy.spin(node)
    finally:
        node.cmd_pub.publish(zero_twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
