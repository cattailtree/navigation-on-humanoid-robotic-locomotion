from __future__ import annotations

import math
import os
from pathlib import Path

from geometry_msgs.msg import PoseStamped, Quaternion, Twist
from nav_msgs.msg import Path as RosPath
from rclpy.node import Node
from std_msgs.msg import Header


def repo_root_from_package() -> Path:
    env_root = os.environ.get("FDM_REPO_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "scripts" / "semantic_nav").exists() and (parent / "exts" / "fdm").exists():
            return parent
    # Source-layout fallback: fdm/ros2_ws/src/semantic_nav_ros/semantic_nav_ros/ros_utils.py
    return current.parents[4]


def add_semantic_nav_to_syspath() -> None:
    import sys

    semantic_root = repo_root_from_package() / "scripts" / "semantic_nav"
    text = str(semantic_root)
    if text not in sys.path:
        sys.path.insert(0, text)


def yaw_from_quaternion(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


def twist_from_velocity(vx: float, vy: float, wz: float) -> Twist:
    msg = Twist()
    msg.linear.x = float(vx)
    msg.linear.y = float(vy)
    msg.angular.z = float(wz)
    return msg


def zero_twist() -> Twist:
    return twist_from_velocity(0.0, 0.0, 0.0)


def make_path_msg(
    node: Node,
    *,
    frame_id: str,
    poses: list[tuple[float, float, float]],
) -> RosPath:
    now = node.get_clock().now().to_msg()
    path = RosPath()
    path.header = Header(stamp=now, frame_id=frame_id)
    for x, y, yaw in poses:
        pose = PoseStamped()
        pose.header = Header(stamp=now, frame_id=frame_id)
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.orientation = quaternion_from_yaw(yaw)
        path.poses.append(pose)
    return path
