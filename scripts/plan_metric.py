# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to evaluate planning performance against baseline methods given random paths in a defined environment.

Does a comparison of 10000 paths in simulation with the following methods:
- MPPI using the presented FDM
- MPPI using the baseline FDM
- MPPI using a cost-map
"""
from __future__ import annotations

# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to evaluate planning performance for MPPI with the learned FDM only."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import time

from isaaclab.app import AppLauncher

# local imports
import utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate planning performance for MPPI with learned FDM only.")
parser.add_argument(
    "--run",
    type=str,
    default="Nov19_20-56-45_MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_reducedObs_Occlusion_NoEarlyCollFilter_NoTorque",
    help="Name of the run.",
)
parser.add_argument("--mode", type=str, default="debug", choices=["full", "debug"], help="Mode of the script.")

# append common FDM cli arguments
cli_args.add_fdm_args(parser, default_num_envs=5)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli = parser.parse_args()
args_cli.headless = args_cli.mode != "debug"
args_cli.num_envs = 2 if args_cli.mode == "debug" else 200

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import yaml

import wandb

from fdm.env_cfg import TERRAIN_ANALYSIS_CFG
from fdm.planner import FDMPlanner, get_planner_cfg
from fdm.utils.args_cli_utils import cfg_modifier_pre_init, env_modifier_post_init, planner_cfg_init, robot_changes

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _safe_getattr(obj, name, default=None):
    return getattr(obj, name, default) if obj is not None else default


def collect_terrain_info(planner: FDMPlanner) -> dict:
    """Collect terrain information as robustly as possible without assuming a fixed cfg schema."""
    info = {}

    # cfg-side terrain information
    terrain_cfg = None
    try:
        terrain_cfg = planner.cfg.env_cfg.scene.terrain
    except Exception:
        terrain_cfg = None

    if terrain_cfg is not None:
        for key in [
            "terrain_type",
            "usd_path",
            "prim_path",
            "max_init_terrain_level",
            "num_rows",
            "num_cols",
        ]:
            value = _safe_getattr(terrain_cfg, key, None)
            if value is not None:
                info[key] = value

        terrain_generator = _safe_getattr(terrain_cfg, "terrain_generator", None)
        if terrain_generator is not None:
            info["terrain_generator"] = str(terrain_generator)

    # scene-side information
    try:
        scene_terrain = planner.env.scene.terrain
        info["scene_terrain_class"] = scene_terrain.__class__.__name__
        terrain_origins = _safe_getattr(scene_terrain, "terrain_origins", None)
        if terrain_origins is not None:
            try:
                info["terrain_origins_shape"] = tuple(terrain_origins.shape)
            except Exception:
                info["terrain_origins_shape"] = str(type(terrain_origins))
    except Exception:
        pass

    # helpful context
    info["args_env"] = getattr(args_cli, "env", None)
    info["run"] = planner.cfg.load_run

    return info


def print_and_save_terrain_info(planner: FDMPlanner, save_dir: str):
    terrain_info = collect_terrain_info(planner)

    print("[INFO] Terrain information:")
    for k, v in terrain_info.items():
        print(f"  - {k}: {v}")

    os.makedirs(save_dir, exist_ok=True)
    terrain_info_path = os.path.join(save_dir, "planner_used_terrain.yaml")
    with open(terrain_info_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(terrain_info, f, allow_unicode=True, sort_keys=False)

    print(f"[INFO] Terrain info saved to: {terrain_info_path}")


def load_planner() -> FDMPlanner:
    # setup cfg
    cfg = planner_cfg_init(args_cli)
    cfg = robot_changes(cfg, args_cli)
    cfg = cfg_modifier_pre_init(cfg, args_cli)

    # define the number of runs
    cfg.env_cfg.commands.command.trajectory_config = (
        {
            "num_paths": 40,
            "max_path_length": 8,
            "min_path_length": 3,
        }
        if args_cli.mode == "debug"
        else {
            "num_paths": 1000,
            "max_path_length": 8,
            "min_path_length": 3,
        }
    )

    cfg.env_cfg.commands.command.traj_sampling.terrain_analysis.max_path_length = (
        cfg.env_cfg.commands.command.trajectory_config["max_path_length"]
    )

    # set name of the run
    if args_cli.run is not None:
        cfg.load_run = args_cli.run

    # get planner cfg
    sampling_planner_cfg_dict = get_planner_cfg(
        args_cli.num_envs,
        traj_dim=10,
        debug=False,
        device="cuda",
    )

    # explicitly keep only our learned FDM planner mode
    sampling_planner_cfg_dict["to_cfg"]["control"] = "fdm"

    # build planner
    planner = FDMPlanner(cfg, sampling_planner_cfg_dict, args_cli=args_cli)
    planner = env_modifier_post_init(planner, args_cli=args_cli)
    return planner


def main():
    TERRAIN_ANALYSIS_CFG.sample_points = 10000

    planner = load_planner()
    print(f"[INFO] Planner loaded with run: {planner.cfg.load_run}")

    # robust extraction of trajectory settings
    traj_cfg = planner.cfg.env_cfg.commands.command.trajectory_config
    num_paths = int(traj_cfg["num_paths"])
    min_length = int(traj_cfg["min_path_length"])
    max_length = int(traj_cfg["max_path_length"])

    # save root
    save_dir = os.path.join(planner.log_root_path, planner.cfg.load_run)
    os.makedirs(save_dir, exist_ok=True)

    # print/save terrain info before navigation
    print_and_save_terrain_info(planner, save_dir)

    # init wandb logging
    if args_cli.mode == "full":
        wb_entity = os.getenv("WANDB_ENTITY")
        wb_mode = os.getenv("WANDB_MODE", "online")
        wb_api_key = os.getenv("WANDB_API_KEY")

        if not wb_api_key:
            print("[WARNING] WANDB_API_KEY environment variable not set. Wandb logging will be disabled.")
        else:
            try:
                wandb.init(
                    project="planner_eval",
                    entity=wb_entity,
                    name=args_cli.run,
                    config=planner.cfg.to_dict() | planner.planner_cfg,
                    dir=os.path.join("logs", "fdm", "fdm_se2_prediction_depth", args_cli.run),
                    mode=wb_mode,
                )
            except Exception as e:
                print(f"[WARNING] Wandb init failed: {e}")

    metrics_file = os.path.join(
        save_dir,
        f"planner_eval_metric_mppi_fdm_num{num_paths}_min{min_length}_max{max_length}.yaml",
    )

    if os.path.exists(metrics_file):
        print("[INFO] Planner evaluation metrics for mppi_fdm already available.")
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics = yaml.safe_load(f)
        planner.print_metrics(metrics)
    else:
        print("[INFO] Evaluating planner: mppi_fdm")
        start = time.time()
        metrics = planner.navigate()
        print(f"[INFO] Time taken: {time.time() - start:.3f}s")

        with open(metrics_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(metrics, f, allow_unicode=True, sort_keys=False)

        print(f"[INFO] Metrics saved to: {metrics_file}")
        planner.print_metrics(metrics)

    planner.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()