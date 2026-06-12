from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from semantic_nav_ros.ros_utils import quaternion_from_yaw


class FdmMppiSmokeInputs(Node):
    """Publish minimal flat-world inputs for the ROS2 FDM-MPPI node."""

    def __init__(self) -> None:
        super().__init__("fdm_mppi_smoke_inputs")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("goal_topic", "/semantic_nav/fdm_goal")
        self.declare_parameter("height_scan_topic", "/semantic_nav/fdm_height_scan")
        self.declare_parameter("height_scan_shape", [60, 46])
        self.declare_parameter("goal_x", 2.0)
        self.declare_parameter("goal_y", 0.0)
        self.declare_parameter("goal_yaw", 0.0)
        self.declare_parameter("rate_hz", 5.0)

        self.shape = tuple(int(v) for v in self.get_parameter("height_scan_shape").value)
        self.odom_pub = self.create_publisher(Odometry, str(self.get_parameter("odom_topic").value), 10)
        self.goal_pub = self.create_publisher(PoseStamped, str(self.get_parameter("goal_topic").value), 10)
        self.scan_pub = self.create_publisher(Float32MultiArray, str(self.get_parameter("height_scan_topic").value), 10)
        self.create_timer(1.0 / max(float(self.get_parameter("rate_hz").value), 1.0), self._tick)

    def _tick(self) -> None:
        now = self.get_clock().now().to_msg()
        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = "map"
        odom.child_frame_id = "base_link"
        odom.pose.pose.orientation = quaternion_from_yaw(0.0)
        self.odom_pub.publish(odom)

        goal = PoseStamped()
        goal.header.stamp = now
        goal.header.frame_id = "map"
        goal.pose.position.x = float(self.get_parameter("goal_x").value)
        goal.pose.position.y = float(self.get_parameter("goal_y").value)
        goal.pose.orientation = quaternion_from_yaw(float(self.get_parameter("goal_yaw").value))
        self.goal_pub.publish(goal)

        height = Float32MultiArray()
        height.data = np.zeros(self.shape[0] * self.shape[1], dtype=np.float32).tolist()
        self.scan_pub.publish(height)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = FdmMppiSmokeInputs()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
