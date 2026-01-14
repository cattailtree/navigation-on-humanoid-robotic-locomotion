# Copyright (c) 2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import os
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg
from isaaclab.utils import configclass

import nav_tasks.sensors as nav_patterns
import fdm.mdp as mdp

if TYPE_CHECKING:
    from fdm.env_cfg.env_cfg_base import FDMCfg

UNITREE_MODEL_DIR = "/home/ubuntu/fdm/unitree_model"

@configclass
class UnitreeArticulationCfg(ArticulationCfg):
    joint_sdk_names: list[str] = None
    soft_joint_pos_limit_factor = 0.9

@configclass
class UnitreeUsdFileCfg(sim_utils.UsdFileCfg):
    activate_contact_sensors: bool = True
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4
    )

G1_29DOF_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint"
]


class UnitreeG1Dims:
    n_dof = len(G1_29DOF_JOINT_NAMES)
    contact_dim = 2

from isaaclab.actuators import ImplicitActuatorCfg
# ---- Recommended: use the official-ish standing pose and actuator grouping ----

G1_CFG = UnitreeArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=UnitreeUsdFileCfg(
        usd_path="/home/ubuntu/fdm/unitree_model/G1/29dof/usd/g1_29dof_rev_1_0/configuration/g1_29dof_rev_1_0_physics.usd"
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        # ✅ 用 unitree_rl_lab 的站姿（不算半蹲，属于轻微屈膝的可站稳初始姿态）
        joint_pos={
            "left_hip_pitch_joint": -0.1,
            "right_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            ".*_shoulder_pitch_joint": 0.3,
            "left_shoulder_roll_joint": 0.25,
            "right_shoulder_roll_joint": -0.25,
            ".*_elbow_joint": 0.97,
            "left_wrist_roll_joint": 0.15,
            "right_wrist_roll_joint": -0.15,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # ----------------------------
        # Hips + waist yaw (stronger motors)
        # ----------------------------
        "N7520-14.3": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_.*",
                ".*_hip_yaw_.*",
                "waist_yaw_joint",
            ],
            effort_limit_sim=88,
            velocity_limit_sim=32.0,
            stiffness={
                ".*_hip_.*": 100.0,
                "waist_yaw_joint": 200.0,
            },
            damping={
                ".*_hip_.*": 2.0,
                "waist_yaw_joint": 5.0,
            },
            armature=0.01,
        ),

        # ----------------------------
        # Hip roll + knees (strongest leg motors)
        # ----------------------------
        "N7520-22.5": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_roll_.*",
                ".*_knee_.*",
            ],
            effort_limit_sim=139,
            velocity_limit_sim=20.0,
            stiffness={
                ".*_hip_roll_.*": 100.0,
                ".*_knee_.*": 150.0,
            },
            damping={
                ".*_hip_roll_.*": 2.0,
                ".*_knee_.*": 4.0,
            },
            armature=0.01,
        ),

        # ----------------------------
        # Ankles + shoulders + elbows + wrist_roll + waist pitch/roll (medium motors)
        # ----------------------------
        "N5020-16": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_.*",
                ".*_elbow_.*",
                ".*_wrist_roll.*",
                ".*_ankle_.*",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            effort_limit_sim=25,
            velocity_limit_sim=37.0,
            stiffness=40.0,  # default stiffness for all in this group
            damping={
                ".*_shoulder_.*": 1.0,
                ".*_elbow_.*": 1.0,
                ".*_wrist_roll.*": 1.0,
                ".*_ankle_.*": 2.0,       # ankles slightly higher damping
                "waist_.*_joint": 5.0,     # waist pitch/roll higher damping
            },
            armature=0.01,
        ),

        # ----------------------------
        # Wrist pitch/yaw (small motors)
        # ----------------------------
        "W4010-25": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_wrist_pitch.*",
                ".*_wrist_yaw.*",
            ],
            effort_limit_sim=5,
            velocity_limit_sim=22.0,
            stiffness=40.0,
            damping=1.0,
            armature=0.01,
        ),
    },
    joint_sdk_names=G1_29DOF_JOINT_NAMES.copy(),
)

# 保持一致
G1_CFG.joint_names = G1_29DOF_JOINT_NAMES.copy()


@configclass
class G1PolicyCfg(ObsGroup):
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    velocity_commands = ObsTerm(func=mdp.vel_commands, params={"action_term": "velocity_cmd"})

    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=G1_29DOF_JOINT_NAMES, preserve_order=True)},
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=G1_29DOF_JOINT_NAMES, preserve_order=True)},
        scale=0.05,
    )
    actions = ObsTerm(
        func=mdp.last_low_level_action,
        params={"action_term": "velocity_cmd", "asset_cfg": SceneEntityCfg(name="robot", joint_names=G1_29DOF_JOINT_NAMES, preserve_order=True)},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True

@configclass
class G1ActionsCfg:
    velocity_cmd = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=G1_29DOF_JOINT_NAMES,
        scale=0.25,
        use_default_offset=True,
    )

def unitree_g1_env(cfg: FDMCfg) -> FDMCfg:
    for name in ("foot_scanner_lf", "foot_scanner_rf", "foot_scanner_lh", "foot_scanner_rh"):
        if hasattr(cfg.scene, name):
            setattr(cfg.scene, name, None)

    cfg.scene.robot = G1_CFG
    cfg.observations.policy = G1PolicyCfg()
    cfg.actions = G1ActionsCfg()

    if hasattr(cfg.commands.command, "traj_sampling"):
        cfg.commands.command.traj_sampling.terrain_analysis.raycaster_sensor = "foot_scanner_r"
    return cfg

__all__ = ["unitree_g1_env", "UnitreeG1Dims"]
