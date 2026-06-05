from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import gettempdir
import base64
from math import cos, sin

import cv2
import numpy as np
import torch

from maps.semantic_graph import Pose2D

_CAMERA_CAPTURE_COUNTER = 0
_CAMERA_CACHE: dict[tuple[int, str, tuple[int, int], tuple[str, ...]], object] = {}


@dataclass(frozen=True)
class RobotViewCameraObservation:
    image_jpeg_b64: str
    depth: np.ndarray | None
    intrinsics: np.ndarray | None
    camera_position: np.ndarray | None
    camera_orientation_ros: np.ndarray | None
    camera_eye: np.ndarray | None
    camera_target: np.ndarray | None
    world_origin: np.ndarray | None


@dataclass(frozen=True)
class LabRunRecorderConfig:
    out_dir: Path
    every: int = 10
    resolution: tuple[int, int] = (640, 480)
    top_center: tuple[float, float] = (4.8, 0.4)
    top_height: float = 12.0


@dataclass(frozen=True)
class ElevatorSceneConfig:
    prim_prefix: str = "/World/SemanticNav/Elevator"
    door_width: float = 1.85
    door_height: float = 2.25
    wall_width: float = 4.6
    wall_height: float = 3.1
    thickness: float = 0.06


def corridor_lobby_wall_specs() -> list[tuple[str, Pose2D, tuple[float, float]]]:
    return [
        ("corridor_upper", Pose2D(3.0, 2.60, 0.0), (6.0, 0.16)),
        ("corridor_lower", Pose2D(3.0, -2.60, 0.0), (6.0, 0.16)),
        ("lobby_upper", Pose2D(8.4, 4.35, 0.0), (5.6, 0.16)),
        ("lobby_lower", Pose2D(8.4, -3.55, 0.0), (5.6, 0.16)),
        ("lobby_right", Pose2D(11.2, 0.40, 0.0), (0.16, 7.8)),
    ]


def spawn_minimal_elevator_scene(
    *,
    origin: torch.Tensor,
    elevator_pose: Pose2D,
    device: str,
    cfg: ElevatorSceneConfig = ElevatorSceneConfig(),
    collision: bool = True,
) -> None:
    """Spawn a lightweight elevator-looking visual target in the Lab stage."""

    x = float((origin[0] + elevator_pose.x).item())
    y = float((origin[1] + elevator_pose.y).item())
    yaw = float(elevator_pose.yaw)

    # The current single-elevator graph approaches the elevator from the corridor,
    # so this first version places a wall/door facade perpendicular to +x.
    wall_center = _local_to_world(x, y, yaw, forward=0.0, lateral=0.0, z=cfg.wall_height * 0.5)
    door_left = _local_to_world(x, y, yaw, forward=-0.045, lateral=-cfg.door_width * 0.25, z=cfg.door_height * 0.5)
    door_right = _local_to_world(x, y, yaw, forward=-0.045, lateral=cfg.door_width * 0.25, z=cfg.door_height * 0.5)
    center_gap = _local_to_world(x, y, yaw, forward=-0.09, lateral=0.0, z=cfg.door_height * 0.5)
    door_frame_top = _local_to_world(x, y, yaw, forward=-0.08, lateral=0.0, z=cfg.door_height + 0.06)
    door_frame_left = _local_to_world(x, y, yaw, forward=-0.08, lateral=-cfg.door_width * 0.56, z=cfg.door_height * 0.5)
    door_frame_right = _local_to_world(x, y, yaw, forward=-0.08, lateral=cfg.door_width * 0.56, z=cfg.door_height * 0.5)
    sign = _local_to_world(x, y, yaw, forward=-0.11, lateral=0.0, z=cfg.door_height + 0.42)
    sign_arrow_up = _local_to_world(x, y, yaw, forward=-0.13, lateral=-0.18, z=cfg.door_height + 0.43)
    sign_arrow_down = _local_to_world(x, y, yaw, forward=-0.13, lateral=0.18, z=cfg.door_height + 0.43)
    panel = _local_to_world(x, y, yaw, forward=-0.12, lateral=cfg.door_width * 0.82, z=1.32)
    panel_button_high = _local_to_world(x, y, yaw, forward=-0.14, lateral=cfg.door_width * 0.82, z=1.55)
    panel_button_low = _local_to_world(x, y, yaw, forward=-0.14, lateral=cfg.door_width * 0.82, z=1.18)

    _cuboid(
        f"{cfg.prim_prefix}/wall",
        wall_center,
        scale=(cfg.thickness, cfg.wall_width, cfg.wall_height),
        color=(0.82, 0.83, 0.80),
        collision=collision,
    )
    _cuboid(
        f"{cfg.prim_prefix}/door_left",
        door_left,
        scale=(cfg.thickness * 1.2, cfg.door_width * 0.48, cfg.door_height),
        color=(0.58, 0.62, 0.64),
        collision=collision,
    )
    _cuboid(
        f"{cfg.prim_prefix}/door_right",
        door_right,
        scale=(cfg.thickness * 1.2, cfg.door_width * 0.48, cfg.door_height),
        color=(0.50, 0.54, 0.57),
        collision=collision,
    )
    _cuboid(
        f"{cfg.prim_prefix}/center_gap",
        center_gap,
        scale=(cfg.thickness * 1.4, 0.035, cfg.door_height * 0.98),
        color=(0.02, 0.02, 0.025),
        collision=collision,
    )
    _cuboid(
        f"{cfg.prim_prefix}/door_frame_top",
        door_frame_top,
        scale=(cfg.thickness * 1.7, cfg.door_width * 1.16, 0.12),
        color=(0.14, 0.15, 0.15),
        collision=collision,
    )
    _cuboid(
        f"{cfg.prim_prefix}/door_frame_left",
        door_frame_left,
        scale=(cfg.thickness * 1.7, 0.10, cfg.door_height + 0.10),
        color=(0.14, 0.15, 0.15),
        collision=collision,
    )
    _cuboid(
        f"{cfg.prim_prefix}/door_frame_right",
        door_frame_right,
        scale=(cfg.thickness * 1.7, 0.10, cfg.door_height + 0.10),
        color=(0.14, 0.15, 0.15),
        collision=collision,
    )
    _cuboid(
        f"{cfg.prim_prefix}/sign",
        sign,
        scale=(cfg.thickness * 2.0, 1.35, 0.38),
        color=(0.02, 0.16, 0.52),
        collision=collision,
    )
    _cuboid(
        f"{cfg.prim_prefix}/sign_arrow_up",
        sign_arrow_up,
        scale=(cfg.thickness * 2.4, 0.16, 0.22),
        color=(0.94, 0.95, 0.92),
        collision=collision,
    )
    _cuboid(
        f"{cfg.prim_prefix}/sign_arrow_down",
        sign_arrow_down,
        scale=(cfg.thickness * 2.4, 0.16, 0.22),
        color=(0.94, 0.95, 0.92),
        collision=collision,
    )
    _cuboid(
        f"{cfg.prim_prefix}/call_panel",
        panel,
        scale=(cfg.thickness * 2.0, 0.24, 0.72),
        color=(0.10, 0.11, 0.12),
        collision=collision,
    )
    _cuboid(
        f"{cfg.prim_prefix}/call_button_up",
        panel_button_high,
        scale=(cfg.thickness * 2.4, 0.13, 0.13),
        color=(0.82, 0.84, 0.78),
        collision=collision,
    )
    _cuboid(
        f"{cfg.prim_prefix}/call_button_down",
        panel_button_low,
        scale=(cfg.thickness * 2.4, 0.13, 0.13),
        color=(0.82, 0.84, 0.78),
        collision=collision,
    )


def spawn_elevator_nodes(
    *,
    origin: torch.Tensor,
    node_poses: dict[str, Pose2D],
    device: str,
    collision: bool = True,
) -> None:
    """Spawn static elevator facades for multiple semantic nodes."""

    for node_id, pose in node_poses.items():
        cfg = ElevatorSceneConfig(prim_prefix=f"/World/SemanticNav/Elevator_{node_id}")
        spawn_minimal_elevator_scene(origin=origin, elevator_pose=pose, device=device, cfg=cfg, collision=collision)


def spawn_blind_search_arena(
    *,
    origin: torch.Tensor,
    center: Pose2D = Pose2D(4.0, 0.6, 0.0),
    size: tuple[float, float] = (9.5, 5.2),
    wall_height: float = 1.2,
    wall_thickness: float = 0.12,
) -> None:
    """Spawn a bounded rectangular arena for blind-search experiments."""

    cx = float((origin[0] + center.x).item())
    cy = float((origin[1] + center.y).item())
    sx, sy = size
    z = wall_height * 0.5
    specs = [
        ("front", (cx, cy + sy * 0.5, z), (sx, wall_thickness, wall_height)),
        ("back", (cx, cy - sy * 0.5, z), (sx, wall_thickness, wall_height)),
        ("left", (cx - sx * 0.5, cy, z), (wall_thickness, sy, wall_height)),
        ("right", (cx + sx * 0.5, cy, z), (wall_thickness, sy, wall_height)),
    ]
    for name, position, scale in specs:
        _cuboid(
            f"/World/SemanticNav/BlindArena/{name}",
            position,
            scale=scale,
            color=(0.45, 0.47, 0.50),
        )


def spawn_rect_wall(
    *,
    origin: torch.Tensor,
    name: str,
    center: Pose2D,
    size: tuple[float, float],
    height: float = 1.8,
) -> None:
    """Spawn a rectangular wall obstacle aligned with the local world axes."""

    cx = float((origin[0] + center.x).item())
    cy = float((origin[1] + center.y).item())
    sx, sy = size
    _cuboid(
        f"/World/SemanticNav/Obstacles/{name}",
        (cx, cy, height * 0.5),
        scale=(sx, sy, height),
        color=(0.28, 0.34, 0.32),
    )


def spawn_corridor_lobby_walls(*, origin: torch.Tensor) -> list[tuple[float, float, float, float]]:
    """Spawn a wide corridor that opens into a lobby.

    Returns obstacle rectangles as (center_x, center_y, size_x, size_y) in local frame.
    """

    walls = corridor_lobby_wall_specs()
    for name, center, size in walls:
        spawn_rect_wall(origin=origin, name=name, center=center, size=size, height=1.6)
    return [(center.x, center.y, size[0], size[1]) for _, center, size in walls]


def capture_elevator_camera_b64(
    *,
    env,
    elevator_pose: Pose2D,
    image_path: Path | None = None,
    resolution: tuple[int, int] = (640, 480),
) -> str:
    """Render a fixed Lab RGB camera looking at the elevator facade."""

    origin = env.scene.env_origins[0]
    target = origin + torch.tensor([elevator_pose.x, elevator_pose.y, 1.15], device=env.device)
    eye = origin + torch.tensor([elevator_pose.x - 4.0, elevator_pose.y, 3.0], device=env.device)
    return _capture_camera_b64(
        env=env,
        eye=eye,
        target=target,
        image_path=image_path,
        resolution=resolution,
        camera_prim_path="/World/SemanticNav/elevator_detector_camera",
    )


def capture_node_camera_b64(
    *,
    env,
    node_pose: Pose2D,
    image_path: Path | None = None,
    resolution: tuple[int, int] = (640, 480),
) -> str:
    return capture_elevator_camera_b64(
        env=env,
        elevator_pose=node_pose,
        image_path=image_path,
        resolution=resolution,
    )


def capture_robot_view_camera_b64(
    *,
    env,
    robot_pose: Pose2D,
    image_path: Path | None = None,
    resolution: tuple[int, int] = (640, 480),
    eye_height: float = 1.65,
    forward_offset: float = 0.20,
    lookahead: float = 3.2,
    target_height: float = 1.15,
) -> str:
    """Render RGB from a camera placed at the robot's current forward view."""

    origin = env.scene.env_origins[0]
    yaw = float(robot_pose.yaw)
    forward = torch.tensor([np.cos(yaw), np.sin(yaw), 0.0], device=env.device, dtype=torch.float32)
    pos = origin + torch.tensor([robot_pose.x, robot_pose.y, 0.0], device=env.device, dtype=torch.float32)
    eye = pos + forward * forward_offset + torch.tensor([0.0, 0.0, eye_height], device=env.device)
    target = pos + forward * lookahead + torch.tensor([0.0, 0.0, target_height], device=env.device)
    return _capture_camera_b64(
        env=env,
        eye=eye,
        target=target,
        image_path=image_path,
        resolution=resolution,
        camera_prim_path="/World/SemanticNav/robot_view_camera",
    )


def capture_robot_view_camera_observation(
    *,
    env,
    robot_pose: Pose2D,
    image_path: Path | None = None,
    resolution: tuple[int, int] = (640, 480),
    eye_height: float = 1.65,
    forward_offset: float = 0.20,
    lookahead: float = 3.2,
    target_height: float = 1.15,
) -> RobotViewCameraObservation:
    """Render RGB and depth from the robot's current forward view."""

    origin = env.scene.env_origins[0]
    yaw = float(robot_pose.yaw)
    forward = torch.tensor([np.cos(yaw), np.sin(yaw), 0.0], device=env.device, dtype=torch.float32)
    pos = origin + torch.tensor([robot_pose.x, robot_pose.y, 0.0], device=env.device, dtype=torch.float32)
    eye = pos + forward * forward_offset + torch.tensor([0.0, 0.0, eye_height], device=env.device)
    target = pos + forward * lookahead + torch.tensor([0.0, 0.0, target_height], device=env.device)
    return _capture_camera_observation(
        env=env,
        eye=eye,
        target=target,
        image_path=image_path,
        resolution=resolution,
        camera_prim_path="/World/SemanticNav/robot_view_camera_depth",
    )


class LabRunRecorder:
    """Capture synchronized robot-view and top-down RGB frames during a Lab run."""

    def __init__(self, *, env, cfg: LabRunRecorderConfig):
        import isaaclab.sim as sim_utils
        from isaaclab.sensors.camera import Camera, CameraCfg

        self.env = env
        self.cfg = cfg
        self.frame_index = 0
        self.robot_dir = cfg.out_dir / "robot_view_frames"
        self.top_dir = cfg.out_dir / "topdown_frames"
        self.robot_dir.mkdir(parents=True, exist_ok=True)
        self.top_dir.mkdir(parents=True, exist_ok=True)

        width, height = cfg.resolution

        def make_camera(prim_path: str) -> Camera:
            spawn_cfg = None
            if not sim_utils.find_matching_prims(prim_path):
                spawn_cfg = sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.1, 1000.0),
                )
            camera_cfg = CameraCfg(
                prim_path=prim_path,
                width=width,
                height=height,
                update_period=0.0,
                data_types=["rgb"],
                spawn=spawn_cfg,
            )
            camera = Camera(camera_cfg)
            if not camera.is_initialized:
                camera._initialize_callback(None)
            camera.reset()
            return camera

        self.robot_camera = make_camera("/World/SemanticNav/record_robot_view_camera")
        self.top_camera = make_camera("/World/SemanticNav/record_topdown_camera")

    def capture(self, *, step_idx: int, robot_pose: Pose2D) -> None:
        if self.cfg.every <= 0 or step_idx % self.cfg.every != 0:
            return

        origin = self.env.scene.env_origins[0]
        yaw = float(robot_pose.yaw)
        forward = torch.tensor([cos(yaw), sin(yaw), 0.0], device=self.env.device, dtype=torch.float32)
        base = origin + torch.tensor([robot_pose.x, robot_pose.y, 0.0], device=self.env.device, dtype=torch.float32)
        robot_eye = base + forward * 0.20 + torch.tensor([0.0, 0.0, 1.65], device=self.env.device)
        robot_target = base + forward * 3.2 + torch.tensor([0.0, 0.0, 1.15], device=self.env.device)

        center_x, center_y = self.cfg.top_center
        top_target = origin + torch.tensor([center_x, center_y, 0.0], device=self.env.device, dtype=torch.float32)
        top_eye = origin + torch.tensor(
            [center_x, center_y - 0.03, self.cfg.top_height],
            device=self.env.device,
            dtype=torch.float32,
        )

        robot_rgb = self._render_camera(self.robot_camera, robot_eye, robot_target)
        top_rgb = self._render_camera(self.top_camera, top_eye, top_target)

        name = f"{self.frame_index:06d}_step_{step_idx:05d}.jpg"
        cv2.imwrite(str(self.robot_dir / name), cv2.cvtColor(robot_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(self.top_dir / name), cv2.cvtColor(top_rgb, cv2.COLOR_RGB2BGR))
        self.frame_index += 1

    def _render_camera(self, camera, eye: torch.Tensor, target: torch.Tensor) -> np.ndarray:
        camera.set_world_poses_from_view(eye.unsqueeze(0), target.unsqueeze(0))
        rgb = None
        for attempt in range(3):
            for _ in range(4 + attempt * 4):
                self.env.sim.render()
                camera.update(self.env.step_dt)
            rgb = camera.data.output["rgb"][0].detach().cpu().numpy().astype(np.uint8)
            if float(rgb.max()) > 5.0:
                return rgb
        if rgb is None:
            raise RuntimeError("Recorder camera did not produce an RGB frame")
        return rgb


class ViewportRunRecorder:
    """Capture synchronized robot-view and top-down frames from USD render products."""

    def __init__(self, *, env, cfg: LabRunRecorderConfig):
        self.env = env
        self.cfg = cfg
        self.frame_index = 0
        self.robot_dir = cfg.out_dir / "robot_view_frames"
        self.top_dir = cfg.out_dir / "topdown_frames"
        self.robot_dir.mkdir(parents=True, exist_ok=True)
        self.top_dir.mkdir(parents=True, exist_ok=True)
        self.robot_annotator = None
        self.top_annotator = None
        self.robot_render_product = None
        self.top_render_product = None
        self.robot_camera_path = None
        self.top_camera_path = None
        print(f"[semantic_nav:record] render-product screenshot recorder initialized at {cfg.out_dir}", flush=True)

    def capture(self, *, step_idx: int, robot_pose: Pose2D) -> None:
        if self.cfg.every <= 0 or step_idx % self.cfg.every != 0:
            return

        self._ensure_render_products()
        if self.frame_index == 0:
            print(f"[semantic_nav:record] capturing first render-product frame at step {step_idx}", flush=True)

        import omni.replicator.core as rep
        from pxr import Gf, UsdGeom

        origin = self.env.scene.env_origins[0]
        yaw = float(robot_pose.yaw)
        forward_x = cos(yaw)
        forward_y = sin(yaw)
        base_x = float((origin[0] + robot_pose.x).item())
        base_y = float((origin[1] + robot_pose.y).item())

        robot_camera = UsdGeom.Camera(self.stage.GetPrimAtPath(self.robot_camera_path))
        robot_camera.GetProjectionAttr().Set(UsdGeom.Tokens.perspective)
        robot_camera.GetFocalLengthAttr().Set(18.0)
        robot_eye = Gf.Vec3d(base_x + 0.2 * forward_x, base_y + 0.2 * forward_y, 1.65)
        robot_target = Gf.Vec3d(base_x + 3.2 * forward_x, base_y + 3.2 * forward_y, 1.15)
        self._set_camera_look_at(self.robot_camera_path, robot_eye, robot_target)

        center_x, center_y = self.cfg.top_center
        top_camera = UsdGeom.Camera(self.stage.GetPrimAtPath(self.top_camera_path))
        top_camera.GetProjectionAttr().Set(UsdGeom.Tokens.orthographic)
        top_camera.GetHorizontalApertureAttr().Set(12.0)
        top_camera.GetVerticalApertureAttr().Set(9.0)
        top_eye = Gf.Vec3d(float((origin[0] + center_x).item()), float((origin[1] + center_y - 0.03).item()), self.cfg.top_height)
        top_target = Gf.Vec3d(float((origin[0] + center_x).item()), float((origin[1] + center_y).item()), 0.0)
        self._set_camera_look_at(self.top_camera_path, top_eye, top_target)

        for _ in range(2):
            self.env.sim.render()
        rep.orchestrator.step(rt_subframes=2, pause_timeline=True, delta_time=0.0)

        name = f"{self.frame_index:06d}_step_{step_idx:05d}.png"
        self._write_rgb(self.robot_annotator.get_data(), self.robot_dir / name)
        self._write_rgb(self.top_annotator.get_data(), self.top_dir / name)

        self.frame_index += 1

    def _ensure_render_products(self) -> None:
        if self.robot_annotator is not None:
            return

        import omni.usd
        import omni.replicator.core as rep
        from pxr import Gf, Sdf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        robot_camera_path = Sdf.Path("/World/SemanticNav/record_viewport_robot_camera")
        top_camera_path = Sdf.Path("/World/SemanticNav/record_viewport_top_camera")
        robot_camera = UsdGeom.Camera.Define(stage, robot_camera_path)
        robot_camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 1000.0))
        top_camera = UsdGeom.Camera.Define(stage, top_camera_path)
        top_camera.GetProjectionAttr().Set(UsdGeom.Tokens.orthographic)
        top_camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 1000.0))

        self.stage = stage
        self.robot_camera_path = robot_camera_path
        self.top_camera_path = top_camera_path
        self.robot_render_product = rep.create.render_product(str(robot_camera_path), tuple(self.cfg.resolution))
        self.top_render_product = rep.create.render_product(str(top_camera_path), tuple(self.cfg.resolution))
        self.robot_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self.top_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self.robot_annotator.attach(self.robot_render_product)
        self.top_annotator.attach(self.top_render_product)
        rep.orchestrator.set_capture_on_play(False)

    def _set_camera_look_at(self, camera_path, eye, target) -> None:
        from isaacsim.core.utils.rotations import lookat_to_quatf
        from pxr import Gf, Usd, UsdGeom

        camera = UsdGeom.Xformable(self.stage.GetPrimAtPath(camera_path))
        if not camera.GetPrim().GetAttribute("xformOp:transform"):
            camera.AddTransformOp()
        quat = lookat_to_quatf(Gf.Vec3f(float(eye[0]), float(eye[1]), float(eye[2])), Gf.Vec3f(float(target[0]), float(target[1]), float(target[2])), Gf.Vec3f(0.0, 0.0, 1.0))
        transform = Gf.Matrix4d().SetRotateOnly(quat).SetTranslateOnly(eye)
        camera.GetPrim().GetAttribute("xformOp:transform").Set(transform, Usd.TimeCode.Default())

    def _write_rgb(self, rgba, out_path: Path) -> None:
        rgba = np.asarray(rgba)
        if rgba.ndim != 3 or rgba.shape[2] < 3:
            raise RuntimeError(f"Unexpected RGB annotator output shape: {rgba.shape}")
        rgb = rgba[:, :, :3].astype(np.uint8)
        cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _capture_camera_b64(
    *,
    env,
    eye: torch.Tensor,
    target: torch.Tensor,
    image_path: Path | None,
    resolution: tuple[int, int],
    camera_prim_path: str,
    unique_prim: bool = False,
) -> str:
    global _CAMERA_CAPTURE_COUNTER
    if unique_prim:
        _CAMERA_CAPTURE_COUNTER += 1
        camera_prim_path = f"{camera_prim_path}_{_CAMERA_CAPTURE_COUNTER:06d}"

    width, height = resolution
    camera = _get_capture_camera(
        env=env,
        camera_prim_path=camera_prim_path,
        resolution=(width, height),
        cache=not unique_prim,
    )
    camera.set_world_poses_from_view(eye.unsqueeze(0), target.unsqueeze(0))

    rgb = None
    for attempt in range(6):
        for _ in range(10 + attempt * 10):
            env.sim.render()
            camera.update(env.step_dt)
        rgb = camera.data.output["rgb"][0].detach().cpu().numpy().astype(np.uint8)
        if float(rgb.mean()) > 1.0 or float(rgb.max()) > 5.0:
            break
        print(
            f"[semantic_nav:lab_camera:warn] black frame attempt={attempt + 1} "
            f"mean={float(rgb.mean()):.3f} max={float(rgb.max()):.1f} prim={camera_prim_path}",
            flush=True,
        )
        env.sim.render()
    if rgb is None:
        raise RuntimeError("Camera did not produce an RGB frame")
    if image_path is None:
        image_path = Path(gettempdir()) / "semantic_nav_lab_camera.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("Failed to encode Lab camera image")
    print(f"[semantic_nav:lab_camera] saved={image_path}", flush=True)
    return base64.b64encode(buffer).decode("utf-8")


def _capture_camera_observation(
    *,
    env,
    eye: torch.Tensor,
    target: torch.Tensor,
    image_path: Path | None,
    resolution: tuple[int, int],
    camera_prim_path: str,
) -> RobotViewCameraObservation:
    width, height = resolution
    camera = _get_capture_camera(
        env=env,
        camera_prim_path=camera_prim_path,
        resolution=(width, height),
        cache=True,
        data_types=("rgb", "distance_to_image_plane"),
    )
    camera.set_world_poses_from_view(eye.unsqueeze(0), target.unsqueeze(0))

    rgb = None
    depth = None
    for attempt in range(6):
        for _ in range(10 + attempt * 10):
            env.sim.render()
            camera.update(env.step_dt)
        rgb = camera.data.output["rgb"][0].detach().cpu().numpy().astype(np.uint8)
        depth = camera.data.output["distance_to_image_plane"][0].detach().cpu().numpy().astype(np.float32)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[:, :, 0]
        if float(rgb.mean()) > 1.0 or float(rgb.max()) > 5.0:
            break
        print(
            f"[semantic_nav:lab_camera:warn] black depth frame attempt={attempt + 1} "
            f"mean={float(rgb.mean()):.3f} max={float(rgb.max()):.1f} prim={camera_prim_path}",
            flush=True,
        )
        env.sim.render()
    if rgb is None or depth is None:
        raise RuntimeError("Camera did not produce an RGB-D frame")
    if image_path is None:
        image_path = Path(gettempdir()) / "semantic_nav_lab_camera.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("Failed to encode Lab camera image")
    print(f"[semantic_nav:lab_camera] saved={image_path}", flush=True)
    return RobotViewCameraObservation(
        image_jpeg_b64=base64.b64encode(buffer).decode("utf-8"),
        depth=depth,
        intrinsics=camera.data.intrinsic_matrices[0].detach().cpu().numpy().astype(np.float32),
        camera_position=camera.data.pos_w[0].detach().cpu().numpy().astype(np.float32),
        camera_orientation_ros=camera.data.quat_w_ros[0].detach().cpu().numpy().astype(np.float32),
        camera_eye=eye.detach().cpu().numpy().astype(np.float32),
        camera_target=target.detach().cpu().numpy().astype(np.float32),
        world_origin=env.scene.env_origins[0].detach().cpu().numpy().astype(np.float32),
    )


def _get_capture_camera(
    *,
    env,
    camera_prim_path: str,
    resolution: tuple[int, int],
    cache: bool,
    data_types: tuple[str, ...] = ("rgb",),
):
    import isaaclab.sim as sim_utils
    from isaaclab.sensors.camera import Camera, CameraCfg

    key = (id(env), camera_prim_path, resolution, data_types)
    if cache and key in _CAMERA_CACHE:
        return _CAMERA_CACHE[key]

    width, height = resolution
    spawn_cfg = None
    if not sim_utils.find_matching_prims(camera_prim_path):
        spawn_cfg = sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1000.0),
        )
    camera_cfg = CameraCfg(
        prim_path=camera_prim_path,
        width=width,
        height=height,
        update_period=0.0,
        data_types=list(data_types),
        spawn=spawn_cfg,
    )
    camera = Camera(camera_cfg)
    for _ in range(4):
        env.sim.render()
    if not camera.is_initialized:
        camera._initialize_callback(None)
    camera.reset()
    if cache:
        _CAMERA_CACHE[key] = camera
    return camera


def _cuboid(
    prim_path: str,
    position: tuple[float, float, float],
    *,
    scale: tuple[float, float, float],
    color: tuple[float, float, float],
    collision: bool = True,
) -> None:
    import isaacsim.core.utils.prims as prim_utils
    from pxr import Gf, UsdGeom, UsdPhysics

    prim_utils.create_prim(
        prim_path,
        "Cube",
        translation=position,
        scale=scale,
    )
    geom_prim = UsdGeom.Cube(prim_utils.get_prim_at_path(prim_path))
    geom_prim.CreateDisplayColorAttr()
    geom_prim.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
    if collision:
        UsdPhysics.CollisionAPI.Apply(geom_prim.GetPrim())


def _local_to_world(x: float, y: float, yaw: float, *, forward: float, lateral: float, z: float) -> tuple[float, float, float]:
    cos_yaw = float(np.cos(yaw))
    sin_yaw = float(np.sin(yaw))
    world_x = x + cos_yaw * forward - sin_yaw * lateral
    world_y = y + sin_yaw * forward + cos_yaw * lateral
    return world_x, world_y, z
