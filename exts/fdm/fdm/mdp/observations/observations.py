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
    threshold: float = 100.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    feet_cfg: SceneEntityCfg | None = None,
    K: int = 2,
    feet_support_threshold: float = 5.0,
) -> torch.Tensor:
    """
    Humanoid-friendly collision:
      pelvis hit (force > threshold)
      AND (optional) feet have no support (support_sum < feet_support_threshold)
      AND persists for K consecutive calls (debounce)
    """

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history  # (N, T, B, 3)

    # -------------------------
    # 1) pelvis / main-body hit
    # -------------------------
    # 关键：这里的 sensor_cfg.body_ids 必须已经是 “整型索引”
    pelvis_force = torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1)  # (N, T, npelvis)
    pelvis_force = pelvis_force.flatten(start_dim=1)                                  # (N, T*npelvis)
    pelvis_hit = torch.max(pelvis_force, dim=1)[0] > threshold                         # (N,)

    # -------------------------
    # 2) feet support (optional)
    # -------------------------
    if feet_cfg is None:
        hard_now = pelvis_hit
    else:
        feet_force = torch.norm(net_contact_forces[:, :, feet_cfg.body_ids], dim=-1)  # (N, T, nfeet)
        feet_max = torch.max(feet_force, dim=1)[0]                                     # (N, nfeet)
        support_sum = torch.sum(feet_max, dim=-1)                                      # (N,)
        support_any = (feet_max > feet_support_threshold).any(dim=-1)   # (N,)
        no_support = ~support_any

        hard_now = pelvis_hit & no_support

    # -------------------------
    # 3) debounce: K consecutive
    # -------------------------
    # 注意：这里不能在 base_collision 里“找 ids”，但可以存计数器（按 env 维度）
    if not hasattr(env, "_hard_collision_counter"):
        env._hard_collision_counter = torch.zeros(
            env.num_envs, device=net_contact_forces.device, dtype=torch.long
        )

    env._hard_collision_counter[hard_now] += 1
    env._hard_collision_counter[~hard_now] = 0

    return (env._hard_collision_counter >= K)

def base_collision_obs(
    env: ManagerBasedRLEnv,
    threshold: float = 100.0,
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
