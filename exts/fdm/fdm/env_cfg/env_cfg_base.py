# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import isaaclab.envs.mdp as mdpp

from nav_suite.terrains import NavTerrainImporterCfg

import fdm.mdp as mdp
from fdm import FDM_DATA_DIR

# 从你自己的 G1 配置里引入
from fdm.env_cfg.robot_cfg_g1 import G1_CFG, G1_29DOF_JOINT_NAMES

##
# Constants
##

TERRAIN_ANALYSIS_CFG = mdp.TerrainAnalysisCfg(
    semantic_cost_mapping=None,
    raycaster_sensor="height_scanner",  # 🔥 用我们在 Scene 里定义的 RayCasterCfg
    viz_graph=False,
    viz_height_map=False,
    sample_points=30000,
    height_diff_threshold=0.2,
    wall_height=2.25,
    door_filtering=True,
    grid_resolution=0.05,
    door_height_threshold=1.2,
    max_terrain_size=350.0,
)


##
# Scene definition
##


@configclass
class TerrainSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a humanoid G1 robot."""

    # USD TERRAIN
    terrain = NavTerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=os.path.join(
            FDM_DATA_DIR, "Terrains", "navigation_terrain_wall_usd_merge_large_single_object_maze.usd"
        ),
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
        usd_uniform_env_spacing=10.0,  # 10m spacing between environment origins in the usd environment
    )

    # 机器人：使用你在 robot_cfg_g1 里定义的 G1_CFG
    robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 传感器：高度扫描（给 FDM / terrain analysis 用）
    height_scanner = RayCasterCfg(
        # ✅ G1 的 usd 根 prim 是 g1_29dof_rev_1_0，所以这里挂在那个 prim 上
        prim_path="{ENV_REGEX_NS}/Robot",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),  # 相对 G1 根往上 0.5m
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # 对整条 Robot 树启用 contact 传感器
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=6,
        debug_vis=False,
    )

    # 光源
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color=(1.0, 1.0, 1.0),
            intensity=2000.0,
        ),
    )

    def __post_init__(self):
        # G1 这里是否需要关闭 self-collision 可以按实际模型再调
        self.robot.spawn.articulation_props.enabled_self_collisions = False


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP.

    在这个 FDM 场景里，我们用顶层 action（NavigationSE2Action）直接给速度命令，
    不再另外生成 command，所以保留 NullCommand。
    """

    command: mdp.NullCommandCfg = mdp.NullCommandCfg()


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # 顶层 RL/FDM 输出的动作 term，使用 NavigationSE2Action 来桥接到 G1 低层策略
    velocity_cmd = mdp.NavigationSE2ActionCfg(
        asset_name="robot",
        low_level_action=mdp.JointPositionActionCfg(
            asset_name="robot",
            # ✅ 使用 G1 的 29DOF 关节列表（和 robot_cfg_g1 完全一致）
            joint_names=G1_29DOF_JOINT_NAMES,
            scale=0.25,              # 与你 G1ActionsCfg 保持一致
            use_default_offset=True,
        ),
        low_level_decimation=4,
        # ✅ 这里换成你训练好的 G1 行走策略 .pt（现在暂时用 ANYmal 路径也行，记得之后改）
        low_level_policy_file="/home/ubuntu/fdm/exts/fdm/data/ANYmal-D-New/policy.pt",
        # NavigationSE2Action 在 apply_actions 里会用这个 group 名去取低层 obs
        low_level_obs_group="policy",
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP (G1 低层策略)."""

    @configclass
    class PolicyCfg(ObsGroup):
        """低层 G1 policy 的 obs（结构对齐 unitree_rl_lab 的 RobotEnvCfg.PolicyCfg）"""

        # 1) 角速度（base frame）
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.2,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )

        # 2) 重力方向
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        # 3) 速度指令：这里我们不用 generated_commands，
        #    而是从顶层 action term "velocity_cmd" 里读 SE2 命令（vx, vy, yaw）
        velocity_commands = ObsTerm(
            func=mdp.vel_commands,
            params={"action_term": "velocity_cmd"},
        )

        # 4) 关节位置（相对）——严格使用 G1_29DOF_JOINT_NAMES 顺序
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="robot",
                    joint_names=G1_29DOF_JOINT_NAMES,
                    preserve_order=True,
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        # 5) 关节速度（相对）
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="robot",
                    joint_names=G1_29DOF_JOINT_NAMES,
                    preserve_order=True,
                )
            },
            scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )

        # 6) 上一时刻的低层动作（和 unitree 里 last_action 对应）
        last_action = ObsTerm(
            func=mdp.last_low_level_action,
            params={
                "action_term": "velocity_cmd",
                "asset_cfg": SceneEntityCfg(
                    name="robot",
                    joint_names=G1_29DOF_JOINT_NAMES,
                    preserve_order=True,
                ),
            },
        )

        def __post_init__(self):
            # ⚠️ 关键：和训练时保持一致
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    # ===== 下面这些保持你 FDM 用的结构（如果有 FDM 相关 obs） =====

    @configclass
    class ObsProceptiveCfg(ObsGroup):
        """Proprioceptive observations for the FDM."""

        velocity_commands = ObsTerm(func=mdp.vel_commands, params={"action_term": "velocity_cmd"})
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_torque = ObsTerm(func=mdp.joint_torque)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel_idx0 = ObsTerm(func=mdp.joint_vel)
        joint_pos_error_idx0 = ObsTerm(func=mdp.joint_pos_error_history, params={"history_idx": 0})
        joint_pos_error_idx2 = ObsTerm(func=mdp.joint_pos_error_history, params={"history_idx": 2})
        joint_pos_error_idx4 = ObsTerm(func=mdp.joint_pos_error_history, params={"history_idx": 4})
        joint_vel_idx2 = ObsTerm(func=mdp.joint_velocity_history, params={"history_idx": 2})
        joint_vel_idx4 = ObsTerm(func=mdp.joint_velocity_history, params={"history_idx": 4})
        last_actions = ObsTerm(func=mdp.last_low_level_action, params={"action_term": "velocity_cmd"})
        second_last_action = ObsTerm(
            func=mdp.second_last_low_level_action, params={"action_term": "velocity_cmd"}
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class FdmStateCfg(ObsGroup):
        """Observations of the state of the FDM."""
        base_position = ObsTerm(func=mdp.BasePositionLocalXYZ())

        #base_position = ObsTerm(func=mdp.base_position)
        base_orientation = ObsTerm(func=mdp.base_orientation_xyzw)
        base_collision = ObsTerm(
            func=mdp.base_collision_obs,
            params={
        "threshold": 100.0,
        "K": 3,
        "feet_support_threshold": 5.0,
        "sensor_cfg": SceneEntityCfg(
            "contact_forces",
            body_names=[".*pelvis.*"],
        ),
        "feet_cfg": SceneEntityCfg(
            "contact_forces",
            body_names=[
                "left_ankle_roll_link",
                "right_ankle_roll_link",
            ],
        ),
    },
        )
        hard_contact = ObsTerm(func=mdp.energy_consumption, params={"energy_scale_factor": 0.001})
        friction = ObsTerm(
            func=mdp.FrictionObservation(),
            params={
                # 这个后面我们再精细选 body；先随便给一个，不影响 obs 维度
                "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link",
                "right_ankle_roll_link"])
            },
        )

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = True

    # group 注册
    policy: PolicyCfg = PolicyCfg()
    fdm_obs_proprioception: ObsProceptiveCfg = ObsProceptiveCfg()
    fdm_obs_exteroceptive: ObsGroup = None
    fdm_state: FdmStateCfg = FdmStateCfg()



@configclass
class EventsCfg:
    """Configuration for events."""

    # startup：随机摩擦系数
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material_uniform_static_dynamic_friction,
        mode="startup",
        params={
            # ✅ 这里不再指定 foot/body 名，直接对整个 robot
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # reset：根据地形分析重置 base
    reset_base = EventTerm(
        func=mdp.TerrainAnalysisRootReset(
            cfg=TERRAIN_ANALYSIS_CFG,
            robot_dim=0.6,
        ),
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "yaw_range": (-3.14, 3.14),
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0, 0),
                "roll": (0, 0),
                "pitch": (0, 0),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP (人形 G1 版本)."""

    # 1) 超时结束（可选）
    print("[DEBUG] Using TerminationsCfg from:", __file__)

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # 2) 根节点高度太低 -> 视为摔倒
    fallen_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        # 这个阈值可以调：
        #   G1 正常站立大概 0.78 左右，你可以先设 0.35~0.4
        params={"minimum_height": 0.35},
    )

    # 3) 姿态倾斜太大 -> 视为摔倒
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        # limit_angle 是 “躺倒多少算摔”，单位弧度
        # 四足一般用 ~0.8（大约 45 度），人形可以宽松或更严一点：
        #   0.8 ~ 1.0 都可以先试试
        params={"limit_angle": 1.0},
    )
##
# Environment configuration
##


@configclass
class FDMCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment with Unitree G1."""

    # Scene settings
    scene: TerrainSceneCfg = TerrainSceneCfg(num_envs=2048, env_spacing=2.5, replicate_physics=False)
    rerender_on_reset = True

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    # set rewards to None（FDM 框架本身定义了 reward）
    rewards = None

    def __post_init__(self):
        """Post initialization."""
        # 不再覆盖 robot.spawn.usd_path，让它保持 G1_CFG 中的路径
        self.seed = 1234
        self.decimation = 4
        # simulation settings
        self.sim.dt = 0.005
        self.sim.disable_contact_processing = True
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "min"  # important so that the robots are slipping
        self.sim.physics_material.restitution_combine_mode = "min"
        # render interval
        self.sim.render_interval = self.decimation
        # viewer settings
        self.viewer.eye = (-5.0, 0, 4)
        # sensors update
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt

        # terrain curriculum
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
