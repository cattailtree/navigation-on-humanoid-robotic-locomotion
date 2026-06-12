from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path as RosPath
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

from semantic_nav_ros.ros_utils import (
    make_path_msg,
    repo_root_from_package,
    twist_from_velocity,
    yaw_from_quaternion,
    zero_twist,
)


def _add_mujoco_adapter_to_syspath() -> None:
    root = repo_root_from_package()
    path = root / "scripts" / "mujoco_sim2sim"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


_add_mujoco_adapter_to_syspath()

from fdm_adapter import FDMPlannerAdapter, GoalTrackingAdapter, PlannerObservation  # noqa: E402


class FdmMppiRos2Node(Node):
    """ROS2 wrapper around the Isaac-free FDM/MPPI planner adapter."""

    def __init__(self) -> None:
        super().__init__("fdm_mppi_ros2_node")
        self._declare_parameters()

        self.frame_id = str(self.get_parameter("frame_id").value)
        self.backend = str(self.get_parameter("backend").value)
        self.height_scan_shape = tuple(int(v) for v in self.get_parameter("height_scan_shape").value)
        if len(self.height_scan_shape) != 2:
            raise ValueError("height_scan_shape must contain [height, width]")

        self._pose_xy_yaw: np.ndarray | None = None
        self._goal_xy_yaw: np.ndarray | None = None
        self._height_scan: np.ndarray | None = None
        self._fdm_state: np.ndarray | None = None
        self._fdm_proprioception: np.ndarray | None = None
        self._last_status = ""

        self.planner = self._make_planner()

        self.cmd_pub = self.create_publisher(Twist, str(self.get_parameter("cmd_vel_raw_topic").value), 10)
        self.path_pub = self.create_publisher(RosPath, str(self.get_parameter("path_topic").value), 10)
        self.status_pub = self.create_publisher(String, str(self.get_parameter("status_topic").value), 10)
        self.debug_pub = self.create_publisher(String, str(self.get_parameter("debug_topic").value), 10)
        self.create_subscription(Odometry, str(self.get_parameter("odom_topic").value), self._on_odom, 20)
        self.create_subscription(PoseStamped, str(self.get_parameter("goal_topic").value), self._on_goal, 10)
        self.create_subscription(Float32MultiArray, str(self.get_parameter("height_scan_topic").value), self._on_height_scan, 10)
        self.create_subscription(Float32MultiArray, str(self.get_parameter("fdm_state_topic").value), self._on_fdm_state, 10)
        self.create_subscription(
            Float32MultiArray,
            str(self.get_parameter("fdm_proprioception_topic").value),
            self._on_fdm_proprioception,
            10,
        )
        self.create_timer(1.0 / max(float(self.get_parameter("rate_hz").value), 1.0), self._tick)
        self._publish_status("ready")

    def _declare_parameters(self) -> None:
        default_run_dir = repo_root_from_package() / "logs" / "fdm" / "fdm_se2_prediction_depth" / "Jun11_14-20-48_fdm_train"
        self.declare_parameter("backend", "mppi_only")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("goal_topic", "/semantic_nav/fdm_goal")
        self.declare_parameter("height_scan_topic", "/semantic_nav/fdm_height_scan")
        self.declare_parameter("fdm_state_topic", "/semantic_nav/fdm_state")
        self.declare_parameter("fdm_proprioception_topic", "/semantic_nav/fdm_proprioception")
        self.declare_parameter("cmd_vel_raw_topic", "/semantic_nav/fdm_cmd_vel_raw")
        self.declare_parameter("path_topic", "/semantic_nav/fdm_path")
        self.declare_parameter("status_topic", "/semantic_nav/fdm_status")
        self.declare_parameter("debug_topic", "/semantic_nav/fdm_debug")
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("model_run_dir", str(default_run_dir))
        self.declare_parameter("checkpoint", "")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("height_scan_shape", [60, 46])
        self.declare_parameter("population_size", 256)
        self.declare_parameter("mppi_iterations", 4)
        self.declare_parameter("replan_interval", 1)
        self.declare_parameter("max_vx", 0.35)
        self.declare_parameter("max_vy", 0.05)
        self.declare_parameter("max_wz", 0.55)
        self.declare_parameter("action_min_vx", -0.05)
        self.declare_parameter("action_max_vx", 0.45)
        self.declare_parameter("action_max_vy", 0.08)
        self.declare_parameter("action_max_wz", 0.55)
        self.declare_parameter("kinematic_dt", 0.25)
        self.declare_parameter("require_observation_dims", True)
        self.declare_parameter("publish_zero_until_ready", True)

    def _make_planner(self):
        backend = self.backend.lower()
        common = dict(
            device=str(self.get_parameter("device").value),
            population_size=int(self.get_parameter("population_size").value),
            mppi_iterations=int(self.get_parameter("mppi_iterations").value),
            replan_interval=int(self.get_parameter("replan_interval").value),
            max_vx=float(self.get_parameter("max_vx").value),
            max_vy=float(self.get_parameter("max_vy").value),
            max_wz=float(self.get_parameter("max_wz").value),
        )
        if backend == "goal_tracking":
            return GoalTrackingAdapter(
                max_vx=common["max_vx"],
                max_vy=common["max_vy"],
                max_wz=common["max_wz"],
            )
        run_dir = self._resolve_path(str(self.get_parameter("model_run_dir").value))
        checkpoint_text = str(self.get_parameter("checkpoint").value).strip()
        checkpoint = self._resolve_path(checkpoint_text) if checkpoint_text else self._latest_checkpoint(run_dir)
        use_fdm_model = backend == "fdm_mppi"
        if use_fdm_model and checkpoint is None:
            raise FileNotFoundError("backend=fdm_mppi requires checkpoint or model_collection_round_*.pth in model_run_dir")
        planner = FDMPlannerAdapter(
            checkpoint=checkpoint or Path("__mppi_only_no_checkpoint__.pth"),
            run_dir=run_dir,
            use_fdm_model=use_fdm_model,
            action_min_vx=float(self.get_parameter("action_min_vx").value),
            action_max_vx=float(self.get_parameter("action_max_vx").value),
            action_max_vy=float(self.get_parameter("action_max_vy").value),
            action_max_wz=float(self.get_parameter("action_max_wz").value),
            gait_max_vx=common["max_vx"],
            gait_max_vy=common["max_vy"],
            gait_max_wz=common["max_wz"],
            kinematic_dt=float(self.get_parameter("kinematic_dt").value),
            require_observation_dims=bool(self.get_parameter("require_observation_dims").value),
            **common,
        )
        self.get_logger().info(f"FDM-MPPI ROS2 backend={backend} checkpoint={checkpoint or 'mppi_only'}")
        return planner

    def _resolve_path(self, text: str) -> Path:
        path = Path(text)
        if path.is_absolute():
            return path
        return repo_root_from_package() / path

    def _latest_checkpoint(self, run_dir: Path) -> Path | None:
        checkpoints = sorted(run_dir.glob("model_collection_round_*.pth"))
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda p: int(p.stem.rsplit("_", 1)[-1]))

    def _on_odom(self, msg: Odometry) -> None:
        pos = msg.pose.pose.position
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self._pose_xy_yaw = np.asarray([pos.x, pos.y, yaw], dtype=np.float32)

    def _on_goal(self, msg: PoseStamped) -> None:
        pos = msg.pose.position
        yaw = yaw_from_quaternion(msg.pose.orientation)
        self._goal_xy_yaw = np.asarray([pos.x, pos.y, yaw], dtype=np.float32)
        self.planner.reset()
        self._publish_status(f"goal_received x={pos.x:.2f} y={pos.y:.2f} yaw={yaw:.2f}")

    def _on_height_scan(self, msg: Float32MultiArray) -> None:
        data = np.asarray(msg.data, dtype=np.float32)
        expected = int(self.height_scan_shape[0] * self.height_scan_shape[1])
        if data.size != expected:
            self.get_logger().warn(f"height scan size mismatch: got={data.size} expected={expected}")
            return
        self._height_scan = data.reshape(self.height_scan_shape)

    def _on_fdm_state(self, msg: Float32MultiArray) -> None:
        self._fdm_state = np.asarray(msg.data, dtype=np.float32)

    def _on_fdm_proprioception(self, msg: Float32MultiArray) -> None:
        self._fdm_proprioception = np.asarray(msg.data, dtype=np.float32)

    def _tick(self) -> None:
        if self._pose_xy_yaw is None:
            self._not_ready("waiting_for_odom")
            return
        if self._goal_xy_yaw is None:
            self._not_ready("waiting_for_goal")
            return
        if self._height_scan is None:
            self._not_ready("waiting_for_height_scan")
            return
        obs = PlannerObservation(
            start_xy_yaw=self._pose_xy_yaw,
            goal_xy_yaw=self._goal_xy_yaw,
            height_scan=self._height_scan,
            fdm_state=self._fdm_state,
            fdm_proprioception=self._fdm_proprioception,
        )
        try:
            command = self.planner.command(obs)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"FDM-MPPI planning failed: {exc}")
            self.cmd_pub.publish(zero_twist())
            self._publish_status(f"plan_failed:{exc}")
            return
        self.cmd_pub.publish(twist_from_velocity(command.vx, command.vy, command.wz))
        self._publish_status(f"running cmd=({command.vx:.2f},{command.vy:.2f},{command.wz:.2f})")
        self._publish_debug()
        self._publish_path()

    def _not_ready(self, reason: str) -> None:
        if bool(self.get_parameter("publish_zero_until_ready").value):
            self.cmd_pub.publish(zero_twist())
        self._publish_status(reason)

    def _publish_debug(self) -> None:
        debug = getattr(self.planner, "debug_info", lambda: {})()
        msg = String()
        msg.data = json.dumps(debug, ensure_ascii=False)
        self.debug_pub.publish(msg)

    def _publish_path(self) -> None:
        state_traj = getattr(self.planner, "_last_state_traj", None)
        best_idx = int(getattr(self.planner, "_last_best_idx", 0))
        if state_traj is None or self._pose_xy_yaw is None:
            return
        try:
            traj = state_traj[best_idx].detach().cpu().numpy()
        except Exception:  # noqa: BLE001
            return
        start_x, start_y, start_yaw = [float(v) for v in self._pose_xy_yaw]
        cos_yaw = float(np.cos(start_yaw))
        sin_yaw = float(np.sin(start_yaw))
        poses: list[tuple[float, float, float]] = []
        for state in traj:
            local_x = float(state[0])
            local_y = float(state[1])
            local_yaw = float(np.arctan2(state[2], state[3])) if len(state) >= 4 else 0.0
            world_x = start_x + cos_yaw * local_x - sin_yaw * local_y
            world_y = start_y + sin_yaw * local_x + cos_yaw * local_y
            poses.append((world_x, world_y, start_yaw + local_yaw))
        self.path_pub.publish(make_path_msg(self, frame_id=self.frame_id, poses=poses))

    def _publish_status(self, text: str) -> None:
        if text == self._last_status:
            return
        self._last_status = text
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = FdmMppiRos2Node()
    try:
        rclpy.spin(node)
    finally:
        node.cmd_pub.publish(zero_twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
