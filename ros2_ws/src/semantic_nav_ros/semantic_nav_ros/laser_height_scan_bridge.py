from __future__ import annotations

import math

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray, String


class LaserHeightScanBridge(Node):
    """Project a 2D LaserScan into the local FDM height-scan tensor."""

    def __init__(self) -> None:
        super().__init__("laser_height_scan_bridge")
        self._declare_parameters()
        self.shape = tuple(int(v) for v in self.get_parameter("height_scan_shape").value)
        if len(self.shape) != 2:
            raise ValueError("height_scan_shape must contain [height, width]")
        self.resolution = float(self.get_parameter("resolution").value)
        self.obstacle_height = float(self.get_parameter("obstacle_height").value)
        self.fill_height = float(self.get_parameter("fill_height").value)
        self.max_range = float(self.get_parameter("max_range").value)
        self.dilation_cells = int(self.get_parameter("dilation_cells").value)
        self.center_x_offset_m = float(self.get_parameter("center_x_offset_m").value)
        self.center_y_offset_m = float(self.get_parameter("center_y_offset_m").value)
        self._last_status = ""

        self.scan_pub = self.create_publisher(Float32MultiArray, str(self.get_parameter("height_scan_topic").value), 10)
        self.status_pub = self.create_publisher(String, str(self.get_parameter("status_topic").value), 10)
        self.create_subscription(LaserScan, str(self.get_parameter("scan_topic").value), self._on_scan, 20)

    def _declare_parameters(self) -> None:
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("height_scan_topic", "/semantic_nav/fdm_height_scan")
        self.declare_parameter("status_topic", "/semantic_nav/height_scan_status")
        self.declare_parameter("height_scan_shape", [60, 46])
        self.declare_parameter("resolution", 0.1)
        self.declare_parameter("obstacle_height", 0.20)
        self.declare_parameter("fill_height", 0.0)
        self.declare_parameter("max_range", 5.0)
        self.declare_parameter("dilation_cells", 1)
        self.declare_parameter("center_x_offset_m", 0.0)
        self.declare_parameter("center_y_offset_m", 0.0)

    def _on_scan(self, msg: LaserScan) -> None:
        height_h, height_w = self.shape
        grid = np.full((height_h, height_w), self.fill_height, dtype=np.float32)
        center_row = height_h / 2.0 - self.center_y_offset_m / self.resolution
        center_col = height_w / 2.0 - self.center_x_offset_m / self.resolution

        count = 0
        angle = float(msg.angle_min)
        for raw_distance in msg.ranges:
            distance = float(raw_distance)
            if math.isfinite(distance) and msg.range_min <= distance <= min(msg.range_max, self.max_range):
                x = math.cos(angle) * distance
                y = math.sin(angle) * distance
                col = int(round(center_col + x / self.resolution))
                row = int(round(center_row - y / self.resolution))
                if 0 <= row < height_h and 0 <= col < height_w:
                    self._write_obstacle(grid, row, col)
                    count += 1
            angle += float(msg.angle_increment)

        out = Float32MultiArray()
        out.data = grid.reshape(-1).tolist()
        self.scan_pub.publish(out)
        self._publish_status(f"published points={count} shape={height_h}x{height_w}")

    def _write_obstacle(self, grid: np.ndarray, row: int, col: int) -> None:
        radius = max(0, self.dilation_cells)
        row0 = max(0, row - radius)
        row1 = min(grid.shape[0], row + radius + 1)
        col0 = max(0, col - radius)
        col1 = min(grid.shape[1], col + radius + 1)
        grid[row0:row1, col0:col1] = self.obstacle_height

    def _publish_status(self, text: str) -> None:
        if text == self._last_status:
            return
        self._last_status = text
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = LaserHeightScanBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
