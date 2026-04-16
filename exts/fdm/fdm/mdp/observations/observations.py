# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.actuators import ActuatorNetMLP
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster, RayCasterCamera
from isaaclab.sim import SimulationContext
from isaaclab.utils.warp import raycast_mesh

from nav_tasks.mdp import GoalCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from fdm.mdp import MixedCommand, NavigationSE2Action

"""
Root state.
"""


def base_orientation_xyzw(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Orientation of the asset's root in world frame.

    Note: converts the quaternion to (x, y, z, w) format."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_quat_w[:, [1, 2, 3, 0]]


def base_position(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Position of the asset's root in world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w


def joint_torque(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint positions of the asset."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.applied_torque


def joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint positions of the asset."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos


def joint_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint velocities of the asset."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel


def joint_pos_error_history(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), history_idx: int = 0
) -> torch.Tensor:
    # 返回 shape 存在的 Tensor，避免 ObservationManager .shape 报错
    return torch.zeros((env.num_envs, 0), device=env.device, dtype=torch.float32)
def joint_velocity_history(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), history_idx: int = 0
) -> torch.Tensor:
    return torch.zeros((env.num_envs, 0), device=env.device, dtype=torch.float32)


class FrictionObservation:
    """Friction observation."""

    def __init__(self):
        pass

    def _setup_view(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]

        self._num_shapes_per_body_mapping = []

        for link_path in asset.root_physx_view.link_paths[0]:
            link_physx_view = asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
            self._num_shapes_per_body_mapping.append(link_physx_view.max_shapes)

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        """The friction coefficients of the asset."""
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]

        # setup the view
        if not hasattr(self, "_num_shapes_per_body_mapping"):
            self._setup_view(env, asset_cfg)

        # get t materials of the bodies
        materials = asset.root_physx_view.get_material_properties()
        static_friction = torch.zeros(
            (env.num_envs, len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies, 1),
            device=env.device,
        )

        # get material properties for the bodies
        for idx, body_id in enumerate(
            asset_cfg.body_ids if isinstance(asset_cfg.body_ids, list) else range(asset.num_bodies)
        ):
            # start index of shape
            start_idx = sum(self._num_shapes_per_body_mapping[:body_id])
            # end index of shape
            end_idx = start_idx + self._num_shapes_per_body_mapping[body_id]
            # get the static friction
            if end_idx - start_idx > 1:
                static_friction[:, idx, 0] = materials[:, start_idx:end_idx, 0].mean(dim=-1)
            else:
                static_friction[:, idx] = materials[:, start_idx:end_idx, 0]

        return static_friction.squeeze(-1)

    def __name__(self):
        return "FrictionObservation"


def se2_root_position(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The root position of the asset in the SE(2) frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # get yaw angle of the root
    yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)[2]
    # return the root position in the SE(2)
    return torch.cat([asset.data.root_pos_w[:, :2], yaw.unsqueeze(-1)], dim=-1)
class Se2RootPositionLocal:
    """Return (dx, dy, dyaw) in episode-local frame (aligned with initial yaw)."""

    def __init__(self):
        pass

    def _ensure(self, env: ManagerBasedRLEnv, device):
        if not hasattr(self, "_init_pos_w"):
            self._init_pos_w = torch.zeros((env.num_envs, 2), device=device, dtype=torch.float32)
            self._init_yaw = torch.zeros((env.num_envs,), device=device, dtype=torch.float32)
            self._inited = torch.zeros((env.num_envs,), device=device, dtype=torch.bool)

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        device = env.device
        self._ensure(env, device)

        pos_w = asset.data.root_pos_w[:, :2]
        yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)[2]

        # 1) 尽量找到真正的 reset buf
        reset = None

        # IsaacLab 常见位置（按优先级尝试）
        if hasattr(env, "reset_buf"):
            reset = env.reset_buf
        elif hasattr(env, "termination_manager") and hasattr(env.termination_manager, "reset_buf"):
            reset = env.termination_manager.reset_buf
        elif hasattr(env, "reset_manager") and hasattr(env.reset_manager, "reset_buf"):
            reset = env.reset_manager.reset_buf

        # 2) 兜底：第一次运行时初始化
        if reset is None:
            reset = ~self._inited


        if torch.any(reset):
            idx = reset.nonzero(as_tuple=False).squeeze(-1)
            self._init_pos_w[idx] = pos_w[idx]
            self._init_yaw[idx] = yaw[idx]
            self._inited[idx] = True

        dp = pos_w - self._init_pos_w  # world delta

        # rotate into initial heading frame (so local x is "forward at episode start")
        cy = torch.cos(self._init_yaw)
        sy = torch.sin(self._init_yaw)
        dx =  cy * dp[:, 0] + sy * dp[:, 1]
        dy = -sy * dp[:, 0] + cy * dp[:, 1]

        dyaw = math_utils.wrap_to_pi(yaw - self._init_yaw)

        return torch.stack([dx, dy, dyaw], dim=-1)

    def __name__(self):
        return "Se2RootPositionLocal"

class BasePositionLocalXYZ:
    """Return (dx, dy, dz) in episode-local frame (origin at episode start)."""

    def _ensure(self, env, device):
        if not hasattr(self, "_init_pos_w"):
            self._init_pos_w = torch.zeros((env.num_envs, 3), device=device, dtype=torch.float32)
            self._inited = torch.zeros((env.num_envs,), device=device, dtype=torch.bool)

    def __call__(self, env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        device = env.device
        self._ensure(env, device)

        pos_w = asset.data.root_pos_w  # (N,3)

        # 关键：正确拿 reset 信号（否则会跨 episode 漂移）
        reset = None
        if hasattr(env, "reset_buf"):
            reset = env.reset_buf
        elif hasattr(env, "termination_manager") and hasattr(env.termination_manager, "reset_buf"):
            reset = env.termination_manager.reset_buf
        elif hasattr(env, "reset_manager") and hasattr(env.reset_manager, "reset_buf"):
            reset = env.reset_manager.reset_buf
        if reset is None:
            reset = ~self._inited

        if torch.any(reset):
            idx = reset.nonzero(as_tuple=False).squeeze(-1)
            self._init_pos_w[idx] = pos_w[idx]
            self._inited[idx] = True

        dpos = pos_w - self._init_pos_w  # (dx,dy,dz)
        return dpos

    def __name__(self):
        return "BasePositionLocalXYZ"


"""
Sensors
"""


def lidar2Dnormalized(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Lidar scan from the given sensor w.r.t. the sensor's frame."""
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # return the height scan
    distances = torch.norm((sensor.data.ray_hits_w - sensor.data.pos_w[:, None, :]), dim=-1)
    # clip inf values to max_distance
    distances[torch.isinf(distances)] = sensor.cfg.max_distance
    # returned clipped to the sensor's range
    return torch.clip(distances, 0.0, sensor.cfg.max_distance) / sensor.cfg.max_distance


def raycast_depth_camera_data(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, data_type: str) -> torch.Tensor:
    """Images generated by the raycast camera."""
    # extract the used quantities (to enable type-hinting)
    sensor: RayCasterCamera = env.scene.sensors[sensor_cfg.name]

    # return the data
    output = sensor.data.output[data_type].clone().unsqueeze(-1)
    output[torch.isnan(output)] = sensor.cfg.max_distance
    output[torch.isinf(output)] = sensor.cfg.max_distance

    # normalize the data
    # output = torch.clip(output, 0.0, sensor.cfg.max_distance) / sensor.cfg.max_distance
    return output


def height_scan_inf_filtered(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # height scan: height = sensor_height - hit_point_z - offset
    height = sensor.data.ray_hits_w[..., 2] + offset - sensor.data.pos_w[:, 2].unsqueeze(1)
    # assign max distance to inf values
    height[torch.isinf(height)] = sensor.cfg.max_distance
    height[torch.isnan(height)] = sensor.cfg.max_distance

    return height


def height_scan_square_fdm(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    shape: list[int] | None = None,
    offset: float = 0.5,
    **kwargs,
) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame given in the square pattern of the sensor."""
    # call regular height scanner function
    height = height_scan_inf_filtered(env, sensor_cfg, offset=offset)
    shape = shape if shape is not None else [int(math.sqrt(height.shape[1])), int(math.sqrt(height.shape[1]))]
    # unflatten the height scan to make use of spatial information
    height_square = torch.unflatten(height, 1, (shape[0], shape[1]))
    # the height scan is mirrored as the pattern is created from neg to pos whereas in the robotics frame, the left of
    # the robot is positive and the right is negative
    height_square = torch.flip(height_square, dims=[1])
    # unqueeze to make compatible with convolutional layers
    return height_square.unsqueeze(1)


def height_scan_door_recognition_fdm(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    shape: list[int] | None = None,
    door_height_thres: float = 1.25,
    offset: float = 0.5,
    return_height: bool = True,
) -> torch.Tensor | None:
    """Height scan from the given sensor w.r.t. the sensor's frame given in the square pattern of the sensor.

    Explicitly account for doors in the scene."""

    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    # get the sensor hit points
    ray_origins = sensor.data.ray_hits_w.clone()

    # we raycast one more time shortly above the ground up and down, if the up raycast hits and is lower than the
    # initial raycast, a potential door is detected
    ray_origins[..., 2] = 0.5
    ray_directions = torch.zeros_like(ray_origins)
    ray_directions[..., 2] = -1.0

    hit_point_down = raycast_mesh(
        ray_origins,
        ray_directions,
        mesh=sensor.meshes[sensor.cfg.mesh_prim_paths[0]],
        max_dist=sensor.cfg.max_distance,
    )[0]

    ray_directions[..., 2] = 1.0

    hit_point_up = raycast_mesh(
        ray_origins,
        ray_directions,
        mesh=sensor.meshes[sensor.cfg.mesh_prim_paths[0]],
        max_dist=sensor.cfg.max_distance,
    )[0]

    lower_height = (
        (hit_point_up[..., 2] < (sensor.data.ray_hits_w[..., 2] - 1e-3))
        & torch.isfinite(hit_point_up[..., 2])
        & ((hit_point_up[..., 2] - hit_point_down[..., 2]) > door_height_thres)
        & torch.isfinite(hit_point_down[..., 2])
    )

    # overwrite the data
    sensor.data.ray_hits_w[lower_height] = hit_point_down[lower_height]

    # debug
    if False:
        env_render_steps = 1000

        # provided height scan
        positions = sensor.data.ray_hits_w.clone()
        # flatten positions
        positions = positions.view(-1, 3)

        # in headless mode, we cannot visualize the graph and omni.debug.draw is not available
        try:
            import omni.isaac.debug_draw._debug_draw as omni_debug_draw

            draw_interface = omni_debug_draw.acquire_debug_draw_interface()
            draw_interface.draw_points(
                positions.tolist(),
                [(1.0, 0.5, 0, 1)] * positions.shape[0],
                [5] * positions.shape[0],
            )

            sim = SimulationContext.instance()
            for _ in range(env_render_steps):
                sim.render()

            # clear the drawn points and lines
            draw_interface.clear_points()
            draw_interface.clear_lines()

        except ImportError:
            print("[WARNING] Cannot visualize occluded height scan in headless mode.")

    if return_height:
        # call regular height scanner function
        return height_scan_square_fdm(env, sensor_cfg, shape, offset)
    else:
        return None


def height_scan_square_fdm_exp_occlu(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    shape: list[int] | None = None,
    offset: float = 0.5,
    **kwargs,
) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame given in the square pattern of the sensor.

    Explicitly account for occulsions of the terrain."""

    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    # get the sensor hit points
    ray_hits = sensor.data.ray_hits_w.clone()
    # account for the sensor offset
    robot_position = asset.data.root_pos_w + math_utils.quat_apply(
        asset.data.root_quat_w, torch.tensor([[0.4, 0.0, 0.0]], device=asset.device).repeat(env.num_envs, 1)
    )
    robot_position = robot_position[:, None, :].repeat(1, ray_hits.shape[1], 1)
    ray_directions = ray_hits - robot_position

    # NOTE: ray directions can never be inf or nan, otherwise the raycasting takes forever
    ray_directions[torch.isinf(ray_directions)] = 0.0
    ray_directions[torch.isnan(ray_directions)] = 0.0

    # raycast from the robot to intended hit positions
    ray_hits_w = raycast_mesh(
        robot_position,
        ray_directions,
        mesh=sensor.meshes[sensor.cfg.mesh_prim_paths[0]],
        max_dist=sensor.cfg.max_distance,
    )[0]

    # get not visible parts of the height-scan
    unseen = torch.norm(ray_hits_w - ray_hits, dim=2) > 0.01

    # overwrite the data
    if torch.any(unseen):
        unseen_points = sensor.data.ray_hits_w[unseen]
        unseen_points[..., 2] = sensor.cfg.max_distance
        sensor.data.ray_hits_w[unseen] = unseen_points

    # debug
    if False:
        env_render_steps = 1000

        # provided height scan
        positions = sensor.data.ray_hits_w.clone()
        # flatten positions
        positions = positions.view(-1, 3)

        # in headless mode, we cannot visualize the graph and omni.debug.draw is not available
        try:
            import omni.isaac.debug_draw._debug_draw as omni_debug_draw

            draw_interface = omni_debug_draw.acquire_debug_draw_interface()
            draw_interface.draw_points(
                positions.tolist(),
                [(1.0, 0.5, 0, 1)] * positions.shape[0],
                [5] * positions.shape[0],
            )

            sim = SimulationContext.instance()
            for _ in range(env_render_steps):
                sim.render()

            # clear the drawn points and lines
            draw_interface.clear_points()
            draw_interface.clear_lines()

        except ImportError:
            print("[WARNING] Cannot visualize occluded height scan in headless mode.")

    # run regular height scan
    return height_scan_square_fdm(env, sensor_cfg, shape, offset)


def height_scan_square_fdm_exp_occlu_with_door_recognition(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    shape: list[int] | None = None,
    door_height_thres: float = 1.25,
    offset: float = 0.5,
    **kwargs,
) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame given in the square pattern of the sensor.

    Explicitly account for occulsions of the terrain and doors in the scene.
    """

    height_scan_door_recognition_fdm(
        env,
        sensor_cfg,
        shape,
        door_height_thres=door_height_thres,
        offset=offset,
        return_height=False,
    )
    return height_scan_square_fdm_exp_occlu(env, asset_cfg, sensor_cfg, shape, offset)


"""
Collision
"""


def base_collision(
    env: ManagerBasedRLEnv,
    threshold: float = 60.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    feet_cfg: SceneEntityCfg | None = None,
    K: int = 2,
    feet_support_threshold: float = 5.0,
    require_no_support: bool = False,
) -> torch.Tensor:
    """
    Humanoid-friendly hard collision / failure signal.

    Triggers when:
      1) non-foot body contact exceeds threshold
      2) optionally combined with no foot support
      3) persists for K consecutive calls
    """

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history  # (N, T, B, 3)

    # --------------------------------------------------
    # 1) Non-foot body contact
    # --------------------------------------------------
    body_force = torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1)  # (N, T, nbodies)
    body_force = body_force.flatten(start_dim=1)  # (N, T*nbodies)
    body_hit = torch.max(body_force, dim=1)[0] > threshold  # (N,)

    # --------------------------------------------------
    # 2) Feet support
    # --------------------------------------------------
    if feet_cfg is not None:
        feet_force = torch.norm(net_contact_forces[:, :, feet_cfg.body_ids], dim=-1)  # (N, T, nfeet)
        feet_max = torch.max(feet_force, dim=1)[0]  # (N, nfeet)

        # at least one supporting foot
        support_any = (feet_max > feet_support_threshold).any(dim=-1)  # (N,)
        no_support = ~support_any
    else:
        no_support = torch.zeros_like(body_hit, dtype=torch.bool)

    # --------------------------------------------------
    # 3) Current hard-failure logic
    # --------------------------------------------------
    if feet_cfg is None:
        hard_now = body_hit
    else:
        if require_no_support:
            hard_now = body_hit & no_support
        else:
            # More practical for humanoid navigation:
            # strong body hit OR complete loss of support
            hard_now = body_hit | no_support

    # --------------------------------------------------
    # 4) Debounce: K consecutive
    # --------------------------------------------------
    if not hasattr(env, "_hard_collision_counter"):
        env._hard_collision_counter = torch.zeros(
            env.num_envs, device=net_contact_forces.device, dtype=torch.long
        )

    env._hard_collision_counter[hard_now] += 1
    env._hard_collision_counter[~hard_now] = 0

    return env._hard_collision_counter >= K

def base_collision_obs(
    env: ManagerBasedRLEnv,
    threshold: float = 60.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    feet_cfg: SceneEntityCfg | None = None,
    K: int = 3,
    feet_support_threshold: float = 5.0,
) -> torch.Tensor:
    collision = base_collision(
        env=env,
        threshold=threshold,
        K=K,
        feet_support_threshold=feet_support_threshold,
        sensor_cfg=sensor_cfg,
        feet_cfg=feet_cfg,
    )
    return collision.unsqueeze(-1)   # (N, 1)

def hard_faliure(
    env: ManagerBasedRLEnv,
    body_force_threshold: float = 20.0,
    feet_support_threshold: float = 10.0,
    min_base_height: float = 0.45,
    max_abs_roll: float = 0.8,
    max_abs_pitch: float = 0.8,
    stuck_steps: int = 5,
    min_progress: float = 0.003,
    command_threshold: float = 0.15,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    feet_cfg: SceneEntityCfg | None = None,
    K: int = 1,
    # near-obstacle patch
    extero_key: str = "extero_obs",
    near_obstacle_height_th: float = 0.08,
    near_obstacle_front_x: float = 0.8,
    near_obstacle_half_width: float = 0.35,
) -> torch.Tensor:
    """
    Humanoid-friendly hard failure signal.

    Covers:
      1) bad body contact
      2) fallen / bad attitude / low base height
      3) stuck with non-trivial command
      4) obstacle too close in front (using local height scan if available)

    Returns a one-step pulse:
      True only when the debounce counter first reaches K.
    """
    device = env.device
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history  # (N, T, B, 3)

    robot = env.scene.articulations["robot"]

    # --------------------------------------------------
    # 0) command intent
    # --------------------------------------------------
    if hasattr(env, "_last_applied_action"):
        cmd = env._last_applied_action
    else:
        cmd = torch.zeros(env.num_envs, 3, device=device)

    cmd_mag = torch.norm(cmd[:, :2], dim=-1) + 0.5 * torch.abs(cmd[:, 2])
    trying_to_move = cmd_mag > command_threshold

    # --------------------------------------------------
    # 1) Bad body contact
    # --------------------------------------------------
    body_force = torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1)  # (N,T,B)
    body_force_last = body_force[:, -1, :]                                           # (N,B)
    body_force_peak = torch.max(body_force_last, dim=1)[0]                           # (N,)
    bad_contact = body_force_peak > body_force_threshold

    # --------------------------------------------------
    # 2) Feet support
    # --------------------------------------------------
    if feet_cfg is not None:
        feet_force = torch.norm(net_contact_forces[:, :, feet_cfg.body_ids], dim=-1)  # (N,T,nfeet)
        feet_last = feet_force[:, -1, :]
        support_sum = torch.sum(feet_last, dim=-1)
        low_support = support_sum < feet_support_threshold
    else:
        low_support = torch.zeros(env.num_envs, device=device, dtype=torch.bool)

    # --------------------------------------------------
    # 3) Fallen / unstable
    # --------------------------------------------------
    root_pos = robot.data.root_pos_w
    root_quat = robot.data.root_quat_w
    roll, pitch, _ = math_utils.euler_xyz_from_quat(root_quat)

    low_height = root_pos[:, 2] < min_base_height
    bad_attitude = (torch.abs(roll) > max_abs_roll) | (torch.abs(pitch) > max_abs_pitch)

    # 保留 low_support 作为“姿态差时”的辅助判断，避免正常支撑时过早判失败
    fallen = low_height | (bad_attitude & low_support)

    # --------------------------------------------------
    # 4) Stuck
    # --------------------------------------------------
    if not hasattr(env, "_stuck_counter"):
        env._stuck_counter = torch.zeros(env.num_envs, device=device, dtype=torch.long)
    if not hasattr(env, "_stuck_prev_pos"):
        env._stuck_prev_pos = root_pos[:, :2].clone()

    progress = torch.norm(root_pos[:, :2] - env._stuck_prev_pos, dim=-1)
    env._stuck_prev_pos = root_pos[:, :2].clone()

    stuck_now = trying_to_move & (progress < min_progress)
    env._stuck_counter[stuck_now] += 1
    env._stuck_counter[~stuck_now] = 0

    stuck = env._stuck_counter >= stuck_steps

    # --------------------------------------------------
    # 5) Near obstacle in front (cheap patch)
    # --------------------------------------------------
    near_obstacle = torch.zeros(env.num_envs, device=device, dtype=torch.bool)

    # Try to fetch local exteroceptive height scan from common locations.
    extero_obs = None
    if hasattr(env, "obs") and isinstance(env.obs, dict) and extero_key in env.obs:
        extero_obs = env.obs[extero_key]
    elif hasattr(env, "_obs") and isinstance(env._obs, dict) and extero_key in env._obs:
        extero_obs = env._obs[extero_key]

    if extero_obs is not None:
        # expected shape: (N,1,H,W) or (N,H,W)
        if extero_obs.dim() == 4:
            height_scan = extero_obs.squeeze(1).to(device).float()  # (N,H,W)
        else:
            height_scan = extero_obs.to(device).float()

        N, H, W = height_scan.shape

        # assume scan is centered on robot and aligned with robot frame
        xs = torch.linspace(
            - (W // 2), W // 2, W, device=device, dtype=height_scan.dtype
        )
        ys = torch.linspace(
            - (H // 2), H // 2, H, device=device, dtype=height_scan.dtype
        )

        # try to infer resolution from env if available, else fall back to 0.1 m
        resolution = 0.1
        if hasattr(env, "height_scan_resolution"):
            resolution = float(env.height_scan_resolution)

        xs = xs * resolution
        ys = ys * resolution

        # NOTE:
        # We only care about a narrow frontal corridor:
        # x in [0, near_obstacle_front_x], |y| <= near_obstacle_half_width
        # Here we assume scan columns roughly correspond to forward x,
        # rows roughly correspond to lateral y. If your scan indexing differs,
        # swap xx / yy accordingly.
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")

        frontal_mask = (xx >= 0.0) & (xx <= near_obstacle_front_x-0.4) & (torch.abs(yy) <= near_obstacle_half_width-0.2)
        frontal_mask = frontal_mask[None].expand(N, -1, -1)

        obstacle_in_front = torch.any((height_scan > near_obstacle_height_th+0.2)& (height_scan < near_obstacle_height_th+4.2)& frontal_mask, dim=(1, 2))

        # only treat as hard failure if robot is actually trying to move
        near_obstacle = trying_to_move & obstacle_in_front

    # --------------------------------------------------
    # 6) Aggregate
    # --------------------------------------------------
    hard_now = bad_contact |fallen|stuck|near_obstacle

    # --------------------------------------------------
    # 7) Debounce with one-shot pulse
    # --------------------------------------------------
    if not hasattr(env, "_hard_failure_counter"):
        env._hard_failure_counter = torch.zeros(env.num_envs, device=device, dtype=torch.long)

    env._hard_failure_counter[hard_now] += 1
    env._hard_failure_counter[~hard_now] = 0

    hard_trigger = env._hard_failure_counter >= K

    # one-shot pulse
    env._hard_failure_counter[hard_trigger] = 0

    return hard_trigger
def hard_faliure_obs(
    env: ManagerBasedRLEnv,
    body_force_threshold: float = 20.0,
    feet_support_threshold: float = 10.0,
    min_base_height: float = 0.45,
    max_abs_roll: float = 0.8,
    max_abs_pitch: float = 0.8,
    stuck_steps: int = 5,
    min_progress: float = 0.003,
    command_threshold: float = 0.15,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    feet_cfg: SceneEntityCfg | None = None,
    K: int = 1,
    # near-obstacle patch
    extero_key: str = "extero_obs",
    near_obstacle_height_th: float = 0.08,
    near_obstacle_front_x: float = 0.8,
    near_obstacle_half_width: float = 0.35,
) -> torch.Tensor:
    collision = hard_faliure(
        env=env,
        body_force_threshold=body_force_threshold,
        feet_support_threshold=feet_support_threshold,
        min_base_height=min_base_height,
        max_abs_roll=max_abs_roll,
        max_abs_pitch=max_abs_pitch,
        stuck_steps=stuck_steps,
        min_progress=min_progress,
        command_threshold=command_threshold,
        sensor_cfg=sensor_cfg,
        feet_cfg=feet_cfg,
        K=K,
        extero_key=extero_key,
        near_obstacle_height_th=near_obstacle_height_th,
        near_obstacle_front_x=near_obstacle_front_x,
        near_obstacle_half_width=near_obstacle_half_width,
    )
    return collision.unsqueeze(-1)
"""
Actions.
"""


def last_low_level_action(
    env: ManagerBasedRLEnv, action_term: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The last low-level action."""
    action_term: NavigationSE2Action = env.action_manager._terms[action_term]
    return action_term.low_level_actions[:, asset_cfg.joint_ids]


def second_last_low_level_action(
    env: ManagerBasedRLEnv, action_term: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The second to last low level action."""
    action_term: NavigationSE2Action = env.action_manager._terms[action_term]
    return action_term.prev_low_level_actions[:, asset_cfg.joint_ids]


"""
Commands.
"""


def vel_commands(env: ManagerBasedRLEnv, action_term: str) -> torch.Tensor:
    """The velocity command generated by the planner and given as input to the step function"""
    action_term: NavigationSE2Action = env.action_manager._terms[action_term]
    return action_term.processed_actions


def goal_command_w_se2(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    command_term: GoalCommand = env.command_manager._terms[command_name]
    goal = command_term.pos_command_w.clone()
    goal[:, 2] = 0.0
    return goal


def goal_command_w_se2_mixed(env: ManagerBasedRLEnv, command_name: str, subterm_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    mixed_command_term: MixedCommand = env.command_manager._terms[command_name]
    command_term: GoalCommand = mixed_command_term.get_subterm(subterm_name)
    goal = command_term.pos_command_w.clone()
    goal[:, 2] = 0.0
    return goal


"""
Energy consumption
"""


def energy_consumption(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), energy_scale_factor: float = 0.001
) -> torch.Tensor:
    """The energy consumption of the asset. Computed as the sum of the squared applied torques."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return (asset.data.applied_torque**2).sum(dim=-1).unsqueeze(-1) * energy_scale_factor
def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase
