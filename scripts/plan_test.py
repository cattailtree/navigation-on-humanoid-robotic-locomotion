# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to test the planning of the FDM.

The script offers following modes:
   - `--mode test`: **qualitative** evaluation in a test environment
   - `--mode metric --env_type 2D`: **qualitative** evaluation in a 2D environment
   - `--mode metric --env_type 3D`: **qualitative** evaluation in a 3D environment
   - `--mode plot`: generate the Fig. **TODO** of the paper
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse
import csv
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher
# local imports
import utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Test Script for MPPI Planning with the FDM model.")
parser.add_argument(
    "--run",
    type=str,
    default="fdm_latest",
    help="Name of the run.",
)
parser.add_argument("--terrain_analysis_points", type=int, default=10000, help="Number of points for terrain analysis.")
parser.add_argument(
    "--mode",
    type=str,
    default="plot",
    choices=["metric", "test", "plot", "plot_video"],
    help="Mode of the script.",
)
parser.add_argument("--env_type", type=str, default="2D", choices=["2D", "3D"], help="Specific environment to pick.")
parser.add_argument(
    "--cost_show",
    type=str,
    default="None",
    choices=["None", "Cost", "Goal_Distance", "Collision", "Height_Scan_Cost", "Pose_Reward"],
    help="Cost visualization mode.",
)
parser.add_argument(
    "--debug_risk",
    action="store_true",
    help="Print risk prediction statistics during planning.",
)
parser.add_argument(
    "--collect_cvae_data",
    action="store_true",
    help="Collect MPPI mean/top-k trajectories for CVAE training.",
)
parser.add_argument(
    "--cvae_dump_path",
    type=str,
    default=r"D:\fdm_data\cvae_dataset\mppi_cvae_dataset.pt",
    help="Path to save the CVAE dataset (.pt).",
)
parser.add_argument(
    "--cvae_topk",
    type=int,
    default=4,
    help="Number of top trajectories per env to store as supervision targets.",
)
parser.add_argument(
    "--cvae_max_samples",
    type=int,
    default=50000,
    help="Maximum number of samples kept in dumped CVAE dataset.",
)
parser.add_argument(
    "--cvae_collect_all_iterations",
    action="store_true",
    default=None,
    help="Collect CVAE tuples at all optimizer iterations, subject to stride.",
)
parser.add_argument(
    "--cvae_collect_iteration_stride",
    type=int,
    default=1,
    help="Collect one CVAE sample round every N optimizer iterations.",
)
parser.add_argument(
    "--cvae_labeled_ratio_min",
    type=float,
    default=0.60,
    help="Minimum labeled share kept in dumped CVAE dataset.",
)
parser.add_argument(
    "--cvae_flush_every_n_samples",
    type=int,
    default=4096,
    help="Flush CVAE dataset to disk after at least this many new samples.",
)
parser.add_argument(
    "--disable_neighbor_spread",
    action="store_true",
    help="Disable MPPI neighbor-spread collision cost propagation for lower memory use.",
)
parser.add_argument(
    "--record-demo",
    action="store_true",
    help="Record a lightweight top-down MP4 during --mode test without using plot_video.",
)
parser.add_argument(
    "--record-demo-dir",
    type=str,
    default=r"D:\fdm_data\planner_eval\record_demo",
    help="Directory for lightweight test-mode recording outputs.",
)
parser.add_argument(
    "--record-demo-env-ids",
    type=int,
    nargs="+",
    default=[0, 1],
    help="Environment ids to draw in the lightweight recording.",
)
parser.add_argument(
    "--record-demo-every",
    type=int,
    default=2,
    help="Record one animation frame every N Lab planner steps.",
)
parser.add_argument(
    "--record-demo-max-frames",
    type=int,
    default=900,
    help="Maximum lightweight recording frames before stopping recording.",
)
parser.add_argument(
    "--record-demo-use-planner-eval-terrain",
    action="store_true",
    help="Use plan_test's normal PLANNER_EVAL_CFG terrain while recording the lightweight demo.",
)
parser.add_argument(
    "--apr07-legacy-model",
    action="store_true",
    help="Use compatibility settings for the Apr07 height FDM checkpoint.",
)
parser.add_argument(
    "--pure-mppi-no-dynamics",
    action="store_true",
    help="Run a command-only MPPI baseline without FDM or ideal SE(2) rollout dynamics.",
)
# append common FDM cli arguments
cli_args.add_fdm_args(parser, default_num_envs=24)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# This repo's Lab planner/elevator setup is G1-based. Do not let the shared
# FDM CLI default ("anymal") silently put plan_test into ANYmal ablation logic.
args_cli.robot = "g1"
if args_cli.ablation_mode in ["no_state_obs", "no_proprio_obs"]:
    raise ValueError(
        "plan_test G1 ablations only support no_height/no_height_scan and no_risk. "
        "no_state_obs/no_proprio_obs are legacy ANYmal-style ablations."
    )

# Keep the plain test-mode command as the MPPI-only baseline, but preserve
# the default height observation/environment config instead of switching to
# the heuristic cost-map planner.
default_test_mppi_only = (
    args_cli.mode == "test"
    and "--env" not in sys.argv
    and "--run" not in sys.argv
    and not args_cli.pure_mppi_no_dynamics
)
if args_cli.run == "Apr07_17-01-32_fdm_train":
    args_cli.apr07_legacy_model = True

# script modes
if args_cli.mode == "test":
    if args_cli.record_demo:
        args_cli.num_envs = max(max(args_cli.record_demo_env_ids) + 1, len(args_cli.record_demo_env_ids), 1)
        args_cli.enable_cameras = False
        args_cli.headless = False
        if getattr(args_cli, "rendering_mode", None) is None:
            args_cli.rendering_mode = "performance"
    else:
        args_cli.num_envs = 12*2
        args_cli.headless = True
elif args_cli.mode == "metric":
    args_cli.num_envs = 12 * 8
    args_cli.headless = True
elif args_cli.mode == "plot" or args_cli.mode == "plot_video":
    args_cli.num_envs = 5

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import torch
import yaml

import fdm.env_cfg.terrain_cfg as fdm_terrain_cfg
import fdm.mdp as mdp
from fdm.env_cfg import TERRAIN_ANALYSIS_CFG

# activate planner mode
from fdm.planner import FDMPlanner, get_planner_cfg
from fdm.utils.args_cli_utils import (
    ablation_studies_modifications,
    cfg_modifier_pre_init,
    env_modifier_post_init,
    planner_cfg_init,
    robot_changes,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def add_env_cameras(planner: FDMPlanner, mode: str):
    from isaacsim.sensors.camera import Camera

    # add camera for each environment
    cameras = []
    for i in range(planner.env.num_envs) if mode == "plot" else range(0, planner.env.num_envs - 1, 2):
        if mode == "plot":
            camera = Camera(prim_path=f"/World/floating_camera_{i}", resolution=(360, 240))
            camera_pos = planner.env.scene.env_origins[i] + torch.tensor([-15, 0.0, 15], device=planner.env.device)
        else:
            camera = Camera(prim_path=f"/World/floating_camera_{i}", resolution=(720, 240))
            camera_pos = (planner.env.scene.env_origins[i] + planner.env.scene.env_origins[i + 1]) / 2 + torch.tensor(
                [-25, 0.0, 23], device=planner.env.device
            )
        camera.set_world_pose(position=camera_pos.tolist(), orientation=[0.9396926, 0.0, 0.3420201, 0.0])
        camera.initialize()
        cameras.append(camera)

    return cameras


def main():
    # reduce required number of samples for the terrain analysis
    if args_cli.mode in [ "plot", "plot_video"]:
        args_cli.terrain_analysis_points = 2000

    # setup runner
    cfg = planner_cfg_init(args_cli)
    if args_cli.apr07_legacy_model:
        args_cli.reduced_obs = False
        args_cli.remove_torque = False
        cfg.load_checkpoint = "model.pth"
        if hasattr(cfg.model_cfg, "use_geometric_collision_head"):
            cfg.model_cfg.use_geometric_collision_head = False
            cfg.model_cfg.geometric_collision_loss_weight = 0.0
    # robot changes
    cfg = robot_changes(cfg, args_cli)
    # modify cfg
    cfg = cfg_modifier_pre_init(cfg, args_cli)
    # ablation studies
    cfg = ablation_studies_modifications(cfg, args_cli)
    if args_cli.record_demo and getattr(cfg.model_cfg, "use_geometric_collision_head", False):
        print("[INFO] record_demo: disabling geometric collision head to match legacy Apr checkpoints.")
        cfg.model_cfg.use_geometric_collision_head = False
        cfg.model_cfg.geometric_collision_loss_weight = 0.0

    # swap environment
    use_planner_eval_terrain = (not args_cli.record_demo) or args_cli.record_demo_use_planner_eval_terrain
    if use_planner_eval_terrain:
        cfg.env_cfg.scene.terrain.terrain_type = "generator"
    if args_cli.mode == "test":
        if use_planner_eval_terrain:
            cfg.env_cfg.scene.terrain.terrain_generator = fdm_terrain_cfg.PLANNER_EVAL_CFG
    elif args_cli.mode == "metric":
        cfg.env_cfg.scene.terrain.terrain_generator = fdm_terrain_cfg.PLANNER_EVAL_CFG
        if args_cli.env == "baseline":
            # NOTE: for comparability, increase the obstacle height so that detactable by the 2D lidar
            cfg.env_cfg.scene.terrain.terrain_generator.sub_terrains["single_box"].height_range[0] = 1.0
            cfg.env_cfg.scene.terrain.terrain_generator.sub_terrains["single_cylinder"].height_range[0] = 1.0
            cfg.env_cfg.scene.terrain.terrain_generator.sub_terrains["single_wall"].height_range[0] = 1.0
            cfg.env_cfg.scene.terrain.terrain_generator.sub_terrains["box_cross_pattern"].height_range[0] = 1.0
            cfg.env_cfg.scene.terrain.terrain_generator.sub_terrains["cylinder_cross_pattern"].height_range[0] = 1.0
            cfg.env_cfg.scene.terrain.terrain_generator.sub_terrains["wall_cross_pattern"].height_range[0] = 1.0
    elif args_cli.mode == "metric" and args_cli.env_type == "3D":
        cfg.env_cfg.scene.terrain.terrain_generator = fdm_terrain_cfg.PLANNER_EVAL_3D_CFG
    elif args_cli.mode == "plot" or args_cli.mode == "plot_video":
        cfg.env_cfg.scene.terrain.terrain_generator = fdm_terrain_cfg.PAPER_PLANNER_FIGURE_TERRAIN_CFG
        # change the initial spawning and resetting function
        cfg.env_cfg.events.reset_base.func = mdp.reset_root_state_center
        cfg.env_cfg.events.reset_base.params = {}
    else:
        raise ValueError(f"Invalid mode {args_cli.mode} and env_type {args_cli.env_type}")

    # make origin selection deterministic
    cfg.env_cfg.scene.terrain.random_seed = 0
    if getattr(cfg.env_cfg.scene.terrain, "terrain_generator", None) is not None:
        terrain_vertical_scale = float(cfg.env_cfg.scene.terrain.terrain_generator.vertical_scale)
        for sub_cfg in cfg.env_cfg.scene.terrain.terrain_generator.sub_terrains.values():
            if hasattr(sub_cfg, "noise_step"):
                sub_cfg.noise_step = max(float(sub_cfg.noise_step), terrain_vertical_scale)

    # set name of the run
    if args_cli.run is not None:
        cfg.load_run = args_cli.run

    # modify the reset function for the robot base state
    if args_cli.mode != "plot" and args_cli.mode != "plot_video":
        # remove the randomization of yaw in the reset_base event
        cfg.env_cfg.events.reset_base.params["yaw_range"] = (0.0, 0.0)
        # enable that it is spawned relative to the env origin
        cfg.env_cfg.events.reset_base.params["spawn_in_env_frame"] = False
        # remove the velocity randomization in the reset_base event
        cfg.env_cfg.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

    # set a fix goal point
    cfg.env_cfg.commands.command = mdp.FixGoalCommandCfg(
        resampling_time_range=(1000000.0, 1000000.0),  # only resample once at the beginning
        debug_vis=True,
        fix_goal_position=[5.0, 0.0, 0.5],
        relative_terrain_origin="origin",
        project_onto_terrain=True,
        terrain_analysis=TERRAIN_ANALYSIS_CFG,
        vis_line=False,
    )
    # add goal randomizations for the metric case and set number of samples
    if args_cli.mode == "test":
        cfg.env_cfg.commands.command.goal_rand_x = (-0.2, 0.2)
        cfg.env_cfg.commands.command.goal_rand_y = (-0.2, 0.2)
        cfg.env_cfg.commands.command.trajectory_num_samples = 480

    # get planner cfg
    sampling_planner_cfg_dict = get_planner_cfg(
        args_cli.num_envs, traj_dim=10, debug=False, device="cuda", population_size=512
    )
    if args_cli.debug_risk:
        sampling_planner_cfg_dict["to_cfg"]["debug"] = True
    if args_cli.collect_cvae_data:
        if not os.path.isabs(args_cli.cvae_dump_path):
            args_cli.cvae_dump_path = os.path.join(r"D:\fdm_data", args_cli.cvae_dump_path)
        args_cli.cvae_dump_path = os.path.abspath(args_cli.cvae_dump_path)
        sampling_planner_cfg_dict["to_cfg"]["cvae_dataset_dump_path"] = args_cli.cvae_dump_path
        sampling_planner_cfg_dict["to_cfg"]["cvae_dataset_topk"] = args_cli.cvae_topk
        sampling_planner_cfg_dict["to_cfg"]["cvae_dataset_max_samples"] = args_cli.cvae_max_samples
        if args_cli.cvae_collect_all_iterations is not None:
            sampling_planner_cfg_dict["to_cfg"]["cvae_collect_all_iterations"] = args_cli.cvae_collect_all_iterations
        sampling_planner_cfg_dict["to_cfg"]["cvae_collect_iteration_stride"] = args_cli.cvae_collect_iteration_stride
        sampling_planner_cfg_dict["to_cfg"]["cvae_labeled_ratio_min"] = args_cli.cvae_labeled_ratio_min
        sampling_planner_cfg_dict["to_cfg"]["cvae_flush_every_n_samples"] = args_cli.cvae_flush_every_n_samples
        cvae_collect_all_iterations = sampling_planner_cfg_dict["to_cfg"]["cvae_collect_all_iterations"]
        print(
            "[CVAE] enabled data collection: "
            f"path={args_cli.cvae_dump_path}, topk={args_cli.cvae_topk}, max_samples={args_cli.cvae_max_samples}, "
            f"all_iters={cvae_collect_all_iterations}, "
            f"stride={args_cli.cvae_collect_iteration_stride}, "
            f"labeled_ratio_min={args_cli.cvae_labeled_ratio_min}, "
            f"flush_every={args_cli.cvae_flush_every_n_samples}"
        )

    if args_cli.env == "heuristic":
        sampling_planner_cfg_dict["to_cfg"]["control"] = "velocity_control"
        sampling_planner_cfg_dict["to_cfg"]["states_cost_w_cost_map"] = True
        sampling_planner_cfg_dict["to_cfg"]["state_cost_w_fatal_trav"] = sampling_planner_cfg_dict["to_cfg"][
            "collision_cost_high_risk_factor"
        ]
        # Elevate height scan to make sure all obstacles are captured
        pos_offset = list(cfg.env_cfg.scene.env_sensor.offset.pos)
        pos_offset[2] = 2.0
        cfg.env_cfg.scene.env_sensor.offset.pos = tuple(pos_offset)
    elif args_cli.env == "baseline":
        sampling_planner_cfg_dict["to_cfg"]["control"] = "fdm_baseline"
        sampling_planner_cfg_dict["to_cfg"]["num_neighbors"] = 3
        sampling_planner_cfg_dict["optim"]["population_size"] = 256
        sampling_planner_cfg_dict["to_cfg"]["collision_cost_safety_factor"] = -0.1
        sampling_planner_cfg_dict["to_cfg"]["collision_cost_high_risk_factor"] = 10 if args_cli.env_type == "2D" else 20
        cfg.env_cfg.episode_length_s = 120.0
        cfg.max_path_time = 120.0
    elif args_cli.env == "height":
        # Elevate height scan to make sure all obstacles are captured
        pos_offset = list(cfg.env_cfg.scene.env_sensor.offset.pos)
        pos_offset[2] = 2.0
        cfg.env_cfg.scene.env_sensor.offset.pos = tuple(pos_offset)
        sampling_planner_cfg_dict["to_cfg"]["num_neighbors"] = 2
        if default_test_mppi_only:
            sampling_planner_cfg_dict["to_cfg"]["control"] = "velocity_control"
            sampling_planner_cfg_dict["to_cfg"]["num_neighbors"] = 0
            sampling_planner_cfg_dict["to_cfg"]["collision_cost_neighbor_spread_weight"] = 0.0
        if args_cli.pure_mppi_no_dynamics:
            sampling_planner_cfg_dict["to_cfg"]["control"] = "direct_command"
            sampling_planner_cfg_dict["to_cfg"]["states_cost_w_cost_map"] = False
            sampling_planner_cfg_dict["to_cfg"]["num_neighbors"] = 0
            sampling_planner_cfg_dict["to_cfg"]["collision_cost_traj_factor"] = 0.0
            sampling_planner_cfg_dict["to_cfg"]["collision_cost_high_risk_factor"] = 0.0
            sampling_planner_cfg_dict["to_cfg"]["collision_cost_neighbor_spread_weight"] = 0.0

    if args_cli.disable_neighbor_spread:
        sampling_planner_cfg_dict["to_cfg"]["num_neighbors"] = 0
        sampling_planner_cfg_dict["to_cfg"]["collision_cost_neighbor_spread_weight"] = 0.0
    if args_cli.ablation_mode == "no_risk":
        sampling_planner_cfg_dict["to_cfg"]["collision_cost_traj_factor"] = 0.0
        sampling_planner_cfg_dict["to_cfg"]["collision_cost_high_risk_factor"] = 0.0
        sampling_planner_cfg_dict["to_cfg"]["num_neighbors"] = 0
        sampling_planner_cfg_dict["to_cfg"]["collision_cost_neighbor_spread_weight"] = 0.0

    # build planner
    planner = FDMPlanner(cfg, sampling_planner_cfg_dict, args_cli=args_cli)

    # post modify runner and env
    planner = env_modifier_post_init(planner, args_cli=args_cli)


    # set the defined cost visualization
    if args_cli.cost_show != "None":
        planner.env._window.current_cost_viz_mode = args_cli.cost_show.replace("_", " ")

    if args_cli.mode == "plot" or args_cli.mode == "plot_video":
        cameras = add_env_cameras(planner, args_cli.mode)
        # navigate
        planner.test(cameras)
        # planner.test()
    elif args_cli.mode == "test":
        # navigate
        metrics = planner.navigate()

        # save the predictions
        save_dir = os.path.abspath(os.path.join(r"D:\fdm_data", "planner_eval"))
        os.makedirs(save_dir, exist_ok=True)
        with open(save_dir + f"/planner_eval_metric_method_{args_cli.env}_env_{args_cli.env_type}.yaml", "w") as f:
            yaml.dump(metrics, f)
    else:
        # navigate
        metrics = planner.navigate()

        # save the predictions
        save_dir = os.path.abspath(os.path.join(r"D:\fdm_data", "planner_eval"))
        os.makedirs(save_dir, exist_ok=True)
        with open(save_dir + f"/planner_eval_metric_method_{args_cli.env}_env_{args_cli.env_type}.yaml", "w") as f:
            yaml.dump(metrics, f)


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
