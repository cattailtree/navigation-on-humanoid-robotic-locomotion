from __future__ import annotations

import base64
import json

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from semantic_nav_ros.ros_utils import add_semantic_nav_to_syspath


add_semantic_nav_to_syspath()

from perception.apexnav_vlm_client import ApexNavGroundingDINOClient  # noqa: E402
from perception.detection_types import PerceptionRequest  # noqa: E402


class GroundingDinoBridgeNode(Node):
    """Bridge compressed ROS images to the ApexNav/GroundingDINO HTTP service."""

    def __init__(self) -> None:
        super().__init__("semantic_nav_gdino_bridge")
        self.declare_parameter("image_topic", "/camera/color/image_raw/compressed")
        self.declare_parameter("goal_prompts_topic", "/semantic_nav/search_prompts")
        self.declare_parameter("detections_topic", "/semantic_nav/detections_json")
        self.declare_parameter("endpoint", "http://127.0.0.1:12181/gdino")
        self.declare_parameter("default_prompts", ["elevator", "lift", "elevator door", "elevator sign"])
        self.declare_parameter("box_threshold", 0.35)
        self.declare_parameter("text_threshold", 0.25)
        self.declare_parameter("timeout_s", 10.0)
        self.declare_parameter("min_publish_period_s", 0.25)

        self.prompts = tuple(str(item) for item in self.get_parameter("default_prompts").value)
        self.min_publish_period_s = float(self.get_parameter("min_publish_period_s").value)
        self._last_request_time = self.get_clock().now()
        self.client = ApexNavGroundingDINOClient(
            endpoint=str(self.get_parameter("endpoint").value),
            box_threshold=float(self.get_parameter("box_threshold").value),
            text_threshold=float(self.get_parameter("text_threshold").value),
            timeout_s=float(self.get_parameter("timeout_s").value),
        )

        self.pub = self.create_publisher(String, str(self.get_parameter("detections_topic").value), 10)
        self.create_subscription(String, str(self.get_parameter("goal_prompts_topic").value), self._on_prompts, 10)
        self.create_subscription(CompressedImage, str(self.get_parameter("image_topic").value), self._on_image, 5)

    def _on_prompts(self, msg: String) -> None:
        pieces = [piece.strip() for piece in msg.data.replace(".", ",").split(",")]
        prompts = tuple(piece for piece in pieces if piece)
        if prompts:
            self.prompts = prompts
            self.get_logger().info(f"updated GroundingDINO prompts={self.prompts}")

    def _on_image(self, msg: CompressedImage) -> None:
        now = self.get_clock().now()
        age = (now - self._last_request_time).nanoseconds * 1.0e-9
        if age < self.min_publish_period_s:
            return
        self._last_request_time = now

        image_b64 = base64.b64encode(bytes(msg.data)).decode("ascii")
        try:
            response = self.client.detect(PerceptionRequest(prompts=self.prompts, image_jpeg_b64=image_b64))
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"GroundingDINO request failed: {exc}")
            return

        payload = {
            "stamp": {"sec": msg.header.stamp.sec, "nanosec": msg.header.stamp.nanosec},
            "frame_id": msg.header.frame_id,
            "prompts": list(self.prompts),
            "detections": [
                {
                    "label": det.label,
                    "score": det.score,
                    "bbox": None
                    if det.bbox is None
                    else {
                        "x1": det.bbox.x1,
                        "y1": det.bbox.y1,
                        "x2": det.bbox.x2,
                        "y2": det.bbox.y2,
                    },
                    "source": det.source,
                }
                for det in response.detections
            ],
        }
        out = String()
        out.data = json.dumps(payload, ensure_ascii=False)
        self.pub.publish(out)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = GroundingDinoBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
