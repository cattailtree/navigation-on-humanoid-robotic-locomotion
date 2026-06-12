from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import Twist
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String

from semantic_nav_ros.ros_utils import zero_twist


class SafetyGateNode(Node):
    """Last-mile velocity gate for real robot deployment."""

    def __init__(self) -> None:
        super().__init__("semantic_nav_safety_gate")
        self._declare_parameters()
        self.raw_cmd_topic = str(self.get_parameter("cmd_vel_raw_topic").value)
        self.cmd_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.status_topic = str(self.get_parameter("safety_status_topic").value)
        self.safety_enabled = bool(self.get_parameter("safety_enabled").value)
        self.stop_distance_m = float(self.get_parameter("stop_distance_m").value)
        self.slow_distance_m = float(self.get_parameter("slow_distance_m").value)
        self.forward_fov_deg = float(self.get_parameter("forward_fov_deg").value)
        self.command_timeout_s = float(self.get_parameter("command_timeout_s").value)
        self.scan_timeout_s = float(self.get_parameter("scan_timeout_s").value)
        self.max_vx = float(self.get_parameter("max_vx").value)
        self.max_vy = float(self.get_parameter("max_vy").value)
        self.max_wz = float(self.get_parameter("max_wz").value)

        self._latest_cmd = zero_twist()
        self._latest_cmd_time = self.get_clock().now()
        self._latest_scan_time = self.get_clock().now()
        self._front_clearance = math.inf
        self._estop = False
        self._last_status = ""

        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.create_subscription(Twist, self.raw_cmd_topic, self._on_raw_cmd, 20)
        self.create_subscription(LaserScan, self.scan_topic, self._on_scan, 20)
        self.create_subscription(Bool, str(self.get_parameter("estop_topic").value), self._on_estop, 10)
        self.create_timer(1.0 / max(float(self.get_parameter("rate_hz").value), 1.0), self._tick)

    def _declare_parameters(self) -> None:
        self.declare_parameter("cmd_vel_raw_topic", "/semantic_nav/cmd_vel_raw")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("estop_topic", "/semantic_nav/estop")
        self.declare_parameter("safety_status_topic", "/semantic_nav/safety_status")
        self.declare_parameter("rate_hz", 30.0)
        self.declare_parameter("safety_enabled", True)
        self.declare_parameter("stop_distance_m", 0.45)
        self.declare_parameter("slow_distance_m", 0.9)
        self.declare_parameter("forward_fov_deg", 70.0)
        self.declare_parameter("command_timeout_s", 0.4)
        self.declare_parameter("scan_timeout_s", 0.8)
        self.declare_parameter("max_vx", 0.35)
        self.declare_parameter("max_vy", 0.05)
        self.declare_parameter("max_wz", 0.55)

    def _on_raw_cmd(self, msg: Twist) -> None:
        self._latest_cmd = msg
        self._latest_cmd_time = self.get_clock().now()

    def _on_scan(self, msg: LaserScan) -> None:
        half_fov = math.radians(self.forward_fov_deg) * 0.5
        best = math.inf
        angle = msg.angle_min
        for distance in msg.ranges:
            if -half_fov <= angle <= half_fov and math.isfinite(distance):
                if msg.range_min <= distance <= msg.range_max:
                    best = min(best, float(distance))
            angle += msg.angle_increment
        self._front_clearance = best
        self._latest_scan_time = self.get_clock().now()

    def _on_estop(self, msg: Bool) -> None:
        self._estop = bool(msg.data)

    def _tick(self) -> None:
        now = self.get_clock().now()
        cmd_age = (now - self._latest_cmd_time).nanoseconds * 1.0e-9
        scan_age = (now - self._latest_scan_time).nanoseconds * 1.0e-9

        if self._estop:
            self._publish_cmd(zero_twist(), "estop")
            return
        if cmd_age > self.command_timeout_s:
            self._publish_cmd(zero_twist(), f"cmd_timeout age={cmd_age:.2f}")
            return
        if self.safety_enabled and scan_age > self.scan_timeout_s:
            self._publish_cmd(zero_twist(), f"scan_timeout age={scan_age:.2f}")
            return

        gated = self._clamped_cmd(self._latest_cmd)
        reason = f"pass clearance={self._front_clearance:.2f}"
        if self.safety_enabled and gated.linear.x > 0.0:
            if self._front_clearance <= self.stop_distance_m:
                gated.linear.x = 0.0
                gated.linear.y = 0.0
                reason = f"stop clearance={self._front_clearance:.2f}"
            elif self._front_clearance <= self.slow_distance_m:
                span = max(self.slow_distance_m - self.stop_distance_m, 1.0e-6)
                scale = max(0.0, min(1.0, (self._front_clearance - self.stop_distance_m) / span))
                gated.linear.x *= scale
                reason = f"slow scale={scale:.2f} clearance={self._front_clearance:.2f}"
        self._publish_cmd(gated, reason)

    def _clamped_cmd(self, msg: Twist) -> Twist:
        out = Twist()
        out.linear.x = _clamp(float(msg.linear.x), -self.max_vx, self.max_vx)
        out.linear.y = _clamp(float(msg.linear.y), -self.max_vy, self.max_vy)
        out.angular.z = _clamp(float(msg.angular.z), -self.max_wz, self.max_wz)
        return out

    def _publish_cmd(self, msg: Twist, status: str) -> None:
        self.cmd_pub.publish(msg)
        if status != self._last_status:
            self._last_status = status
            text = String()
            text.data = status
            self.status_pub.publish(text)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = SafetyGateNode()
    try:
        rclpy.spin(node)
    finally:
        node.cmd_pub.publish(zero_twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
