from __future__ import annotations

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String

from semantic_nav_ros.ros_utils import zero_twist


class CmdMuxNode(Node):
    """Select one raw navigation command source before the safety gate."""

    def __init__(self) -> None:
        super().__init__("semantic_nav_cmd_mux")
        self._declare_parameters()
        self.source = str(self.get_parameter("default_source").value)
        self.timeout_s = float(self.get_parameter("source_timeout_s").value)
        self._cmds: dict[str, Twist] = {
            "waypoint": zero_twist(),
            "fdm": zero_twist(),
        }
        now = self.get_clock().now()
        self._times = {"waypoint": now, "fdm": now}
        self._last_status = ""

        self.cmd_pub = self.create_publisher(Twist, str(self.get_parameter("selected_cmd_topic").value), 10)
        self.status_pub = self.create_publisher(String, str(self.get_parameter("status_topic").value), 10)
        self.create_subscription(Twist, str(self.get_parameter("waypoint_cmd_topic").value), self._on_waypoint, 20)
        self.create_subscription(Twist, str(self.get_parameter("fdm_cmd_topic").value), self._on_fdm, 20)
        self.create_subscription(String, str(self.get_parameter("source_topic").value), self._on_source, 10)
        self.create_timer(1.0 / max(float(self.get_parameter("rate_hz").value), 1.0), self._tick)

    def _declare_parameters(self) -> None:
        self.declare_parameter("waypoint_cmd_topic", "/semantic_nav/cmd_vel_raw")
        self.declare_parameter("fdm_cmd_topic", "/semantic_nav/fdm_cmd_vel_raw")
        self.declare_parameter("selected_cmd_topic", "/semantic_nav/selected_cmd_vel_raw")
        self.declare_parameter("source_topic", "/semantic_nav/cmd_source")
        self.declare_parameter("status_topic", "/semantic_nav/cmd_mux_status")
        self.declare_parameter("default_source", "waypoint")
        self.declare_parameter("source_timeout_s", 0.5)
        self.declare_parameter("rate_hz", 30.0)

    def _on_waypoint(self, msg: Twist) -> None:
        self._cmds["waypoint"] = msg
        self._times["waypoint"] = self.get_clock().now()

    def _on_fdm(self, msg: Twist) -> None:
        self._cmds["fdm"] = msg
        self._times["fdm"] = self.get_clock().now()

    def _on_source(self, msg: String) -> None:
        source = msg.data.strip().lower()
        aliases = {"fdm_mppi": "fdm", "mppi": "fdm", "semantic": "waypoint"}
        source = aliases.get(source, source)
        if source not in self._cmds:
            self._publish_status(f"ignored_unknown_source={source}")
            return
        self.source = source
        self._publish_status(f"source={self.source}")

    def _tick(self) -> None:
        now = self.get_clock().now()
        if self.source not in self._cmds:
            self.cmd_pub.publish(zero_twist())
            self._publish_status(f"invalid_source={self.source}")
            return
        age = (now - self._times[self.source]).nanoseconds * 1.0e-9
        if age > self.timeout_s:
            self.cmd_pub.publish(zero_twist())
            self._publish_status(f"source={self.source} timeout age={age:.2f}")
            return
        self.cmd_pub.publish(self._cmds[self.source])
        self._publish_status(f"source={self.source} age={age:.2f}")

    def _publish_status(self, text: str) -> None:
        if text == self._last_status:
            return
        self._last_status = text
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = CmdMuxNode()
    try:
        rclpy.spin(node)
    finally:
        node.cmd_pub.publish(zero_twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
