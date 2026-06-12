from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, String

class FdmObservationBridge(Node):
    """Build standard ROS2 FDM state/proprio vectors from robot topics."""

    def __init__(self) -> None:
        super().__init__("fdm_observation_bridge")
        self._declare_parameters()
        self.state_dim = int(self.get_parameter("fdm_state_dim").value)
        self.proprio_dim = int(self.get_parameter("fdm_proprioception_dim").value)
        self.action_dim = int(self.get_parameter("action_dim").value)
        self.joint_names = self._read_joint_names()
        self.command = np.zeros(3, dtype=np.float32)
        self.low_level_action = np.zeros(self.action_dim, dtype=np.float32)
        self.second_last_low_level_action = np.zeros(self.action_dim, dtype=np.float32)
        self.odom: Odometry | None = None
        self.origin_xyz: np.ndarray | None = None
        self.joint_state: JointState | None = None
        self._last_status = ""

        self.state_pub = self.create_publisher(Float32MultiArray, str(self.get_parameter("fdm_state_topic").value), 10)
        self.proprio_pub = self.create_publisher(
            Float32MultiArray,
            str(self.get_parameter("fdm_proprioception_topic").value),
            10,
        )
        self.status_pub = self.create_publisher(String, str(self.get_parameter("status_topic").value), 10)
        self.create_subscription(Odometry, str(self.get_parameter("odom_topic").value), self._on_odom, 20)
        self.create_subscription(JointState, str(self.get_parameter("joint_state_topic").value), self._on_joint_state, 20)
        self.create_subscription(Twist, str(self.get_parameter("command_topic").value), self._on_command, 20)
        self.create_subscription(Float32MultiArray, str(self.get_parameter("low_level_action_topic").value), self._on_low_level_action, 20)
        self.create_subscription(
            Float32MultiArray,
            str(self.get_parameter("second_last_low_level_action_topic").value),
            self._on_second_last_low_level_action,
            20,
        )
        self.create_timer(1.0 / max(float(self.get_parameter("rate_hz").value), 1.0), self._tick)

    def _declare_parameters(self) -> None:
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("command_topic", "/semantic_nav/selected_cmd_vel_raw")
        self.declare_parameter("low_level_action_topic", "/semantic_nav/low_level_action")
        self.declare_parameter("second_last_low_level_action_topic", "/semantic_nav/second_last_low_level_action")
        self.declare_parameter("fdm_state_topic", "/semantic_nav/fdm_state")
        self.declare_parameter("fdm_proprioception_topic", "/semantic_nav/fdm_proprioception")
        self.declare_parameter("status_topic", "/semantic_nav/fdm_observation_status")
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("fdm_state_dim", 8)
        self.declare_parameter("fdm_proprioception_dim", 157)
        self.declare_parameter("action_dim", 29)
        self.declare_parameter(
            "proprio_layout_csv",
            "velocity_commands,projected_gravity,base_lin_vel,base_ang_vel,joint_torque,joint_pos,joint_vel,last_actions,second_last_action",
        )
        self.declare_parameter("state_position_frame", "local")
        self.declare_parameter("joint_names", [""])
        self.declare_parameter("joint_names_csv", "")
        self.declare_parameter("energy_scale", 0.0)
        self.declare_parameter("pad_missing_joints", True)

    def _on_odom(self, msg: Odometry) -> None:
        self.odom = msg
        if self.origin_xyz is None:
            pos = msg.pose.pose.position
            self.origin_xyz = np.asarray([float(pos.x), float(pos.y), float(pos.z)], dtype=np.float32)

    def _on_joint_state(self, msg: JointState) -> None:
        self.joint_state = msg

    def _on_command(self, msg: Twist) -> None:
        self.command[:] = [float(msg.linear.x), float(msg.linear.y), float(msg.angular.z)]

    def _on_low_level_action(self, msg: Float32MultiArray) -> None:
        self.low_level_action[:] = self._fit_vector(msg.data, self.action_dim)

    def _on_second_last_low_level_action(self, msg: Float32MultiArray) -> None:
        self.second_last_low_level_action[:] = self._fit_vector(msg.data, self.action_dim)

    def _tick(self) -> None:
        if self.odom is None:
            self._publish_status("waiting_for_odom")
            return
        state = self._build_state()
        proprio = self._build_proprioception()
        state_msg = Float32MultiArray()
        state_msg.data = state.tolist()
        proprio_msg = Float32MultiArray()
        proprio_msg.data = proprio.tolist()
        self.state_pub.publish(state_msg)
        self.proprio_pub.publish(proprio_msg)
        joint_status = "joint_states" if self.joint_state is not None else "no_joint_states"
        self._publish_status(f"published state_dim={len(state)} proprio_dim={len(proprio)} {joint_status}")

    def _build_state(self) -> np.ndarray:
        state = np.zeros(self.state_dim, dtype=np.float32)
        msg = self.odom
        assert msg is not None
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        xyz = np.asarray([float(pos.x), float(pos.y), float(pos.z)], dtype=np.float32)
        if str(self.get_parameter("state_position_frame").value).lower() == "local" and self.origin_xyz is not None:
            xyz = xyz - self.origin_xyz
        values = [
            float(xyz[0]),
            float(xyz[1]),
            float(xyz[2]),
            float(q.x),
            float(q.y),
            float(q.z),
            float(q.w),
            self._energy_proxy(),
        ]
        count = min(len(values), self.state_dim)
        state[:count] = values[:count]
        return state

    def _build_proprioception(self) -> np.ndarray:
        proprio = np.zeros(self.proprio_dim, dtype=np.float32)
        cursor = 0
        vectors = self._proprio_vectors()
        for term in self._proprio_layout():
            vector = vectors.get(term)
            if vector is None:
                self.get_logger().warn(f"unknown proprio layout term: {term}")
                continue
            cursor = self._write(proprio, cursor, vector)
            if cursor >= len(proprio):
                break
        return proprio

    def _proprio_vectors(self) -> dict[str, np.ndarray]:
        base_lin_vel = np.zeros(3, dtype=np.float32)
        base_ang_vel = np.zeros(3, dtype=np.float32)
        if self.odom is not None:
            linear = self.odom.twist.twist.linear
            angular = self.odom.twist.twist.angular
            base_lin_vel[:] = [float(linear.x), float(linear.y), float(linear.z)]
            base_ang_vel[:] = [float(angular.x), float(angular.y), float(angular.z)]
        positions, velocities, efforts = self._joint_vectors()
        return {
            "command": self.command,
            "commands": self.command,
            "velocity_commands": self.command,
            "projected_gravity": self._projected_gravity(),
            "base_lin_vel": base_lin_vel,
            "base_ang_vel": base_ang_vel,
            "joint_effort": efforts,
            "joint_torque": efforts,
            "joint_pos": positions,
            "joint_position": positions,
            "joint_vel": velocities,
            "joint_velocity": velocities,
            "last_action": self.low_level_action,
            "last_actions": self.low_level_action,
            "action": self.low_level_action,
            "second_last_action": self.second_last_low_level_action,
            "prev_action": self.second_last_low_level_action,
        }

    def _projected_gravity(self) -> np.ndarray:
        msg = self.odom
        if msg is None:
            return np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
        q = msg.pose.pose.orientation
        # Rotate world gravity into base frame using inverse quaternion.
        x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
        gx = 2.0 * (x * z - w * y)
        gy = 2.0 * (y * z + w * x)
        gz = w * w - x * x - y * y + z * z
        return np.asarray([gx, gy, -gz], dtype=np.float32)

    def _joint_vectors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        count = len(self.joint_names) or self.action_dim
        if count <= 0 and self.joint_state is not None:
            count = len(self.joint_state.name)
        positions = np.zeros(count, dtype=np.float32)
        velocities = np.zeros(count, dtype=np.float32)
        efforts = np.zeros(count, dtype=np.float32)
        msg = self.joint_state
        if msg is None or count <= 0:
            return positions, velocities, efforts
        name_to_idx = {name: idx for idx, name in enumerate(msg.name)}
        names = self.joint_names or list(msg.name)
        for out_idx, name in enumerate(names[:count]):
            src_idx = name_to_idx.get(name)
            if src_idx is None:
                if not bool(self.get_parameter("pad_missing_joints").value):
                    self.get_logger().warn(f"missing joint state for {name}")
                continue
            if src_idx < len(msg.position):
                positions[out_idx] = float(msg.position[src_idx])
            if src_idx < len(msg.velocity):
                velocities[out_idx] = float(msg.velocity[src_idx])
            if src_idx < len(msg.effort):
                efforts[out_idx] = float(msg.effort[src_idx])
        return positions, velocities, efforts

    def _read_joint_names(self) -> list[str]:
        csv_text = str(self.get_parameter("joint_names_csv").value).strip()
        if csv_text:
            return [name.strip() for name in csv_text.split(",") if name.strip()]
        raw_names = self.get_parameter("joint_names").value
        return [str(name).strip() for name in raw_names if str(name).strip()]

    def _proprio_layout(self) -> list[str]:
        layout = str(self.get_parameter("proprio_layout_csv").value)
        return [term.strip().lower() for term in layout.split(",") if term.strip()]

    def _energy_proxy(self) -> float:
        if self.joint_state is None or not self.joint_state.effort:
            return 0.0
        scale = float(self.get_parameter("energy_scale").value)
        if scale <= 0.0:
            return 0.0
        effort = np.asarray(self.joint_state.effort, dtype=np.float32)
        return float(np.sum(effort * effort) * scale)

    def _write(self, target: np.ndarray, cursor: int, values) -> int:
        if cursor >= len(target):
            return cursor
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        count = min(arr.size, len(target) - cursor)
        target[cursor : cursor + count] = arr[:count]
        return cursor + count

    def _fit_vector(self, values, dim: int) -> np.ndarray:
        out = np.zeros(dim, dtype=np.float32)
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        count = min(dim, arr.size)
        out[:count] = arr[:count]
        return out

    def _publish_status(self, text: str) -> None:
        if text == self._last_status:
            return
        self._last_status = text
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = FdmObservationBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
