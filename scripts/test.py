# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train a Forward-Dynamics-Model"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import torch
import torch.utils.benchmark as benchmark

from isaaclab.app import AppLauncher

# local imports
import utils.cli_args as cli_args  # isort: skip


# ----------------------------
# CLI
# ----------------------------
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--terrain-cfg", type=str, default=None, help="Name of the terrain config to load.")
parser.add_argument("--regular", action="store_true", default=False, help="Spawn robots in a regular pattern.")
parser.add_argument(
    "--runs",
    type=str,
    nargs="+",
    default="Dec03_20-27-43_MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_Noise_reducedObs_Occlusion_NoTorque_NewHeightScanNoise_NewNNTrainNoise_SchedEp10_Wait4_Decay5e5",
    help="Name of the run.",
)
parser.add_argument("--equal-actions", action="store_true", default=False, help="Have the same actions for all envs.")
parser.add_argument("--paper-figure", action="store_true", default=False, help="Run paper figure test.")
parser.add_argument("--paper-platform-figure", action="store_true", default=False, help="Run paper platform figure test.")
parser.add_argument("--terrain_analysis_points", type=int, default=2000, help="Number of points for terrain analysis.")
parser.add_argument("--record", action="store_true", default=False, help="Record the simulation.")

# append common FDM cli arguments
cli_args.add_fdm_args(parser, default_num_envs=360)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

# adapt number of environments for the paper figure
if args_cli.paper_figure:
    args_cli.num_envs = 16
    args_cli.enable_cameras = True
    args_cli.terrain_analysis_points = 500
elif args_cli.paper_platform_figure:
    args_cli.num_envs = 4
    args_cli.enable_cameras = True
    args_cli.terrain_analysis_points = 500

# ✅ 核心：只要 record，就强制启用 cameras（所有模式都录）
if args_cli.record:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# ----------------------------
# Imports after SimulationApp
# ----------------------------
# ---- after SimulationApp is created ----
import omni.kit.app

app = omni.kit.app.get_app()
ext_mgr = app.get_extension_manager()

# 依次尝试启用几种常见 camera 相关 extension
candidates = [
    "isaacsim.sensors.camera",
    "omni.isaac.sensor",          # 你原来用的旧路径对应的 extension 名通常是这个
    "omni.isaac.camera",          # 有些版本是这个
]

enabled_any = False
for ext in candidates:
    try:
        ext_mgr.set_extension_enabled_immediate(ext, True)
        print(f"[INFO] Enabled extension: {ext}")
        enabled_any = True
        break
    except Exception as e:
        print(f"[WARN] Failed enabling {ext}: {e}")

if not enabled_any:
    raise RuntimeError("No camera extension could be enabled. Your Isaac Sim install/experience likely lacks camera exts.")

# 现在再 import（谁能 import 成功就用谁）
try:
    from isaacsim.sensors.camera import Camera
except Exception:
    from omni.isaac.sensor import Camera
    

import isaaclab.sim.spawners as sim_spawners
from isaaclab.assets import AssetBaseCfg
from isaaclab_tasks.utils import get_checkpoint_path

from nav_suite.collectors import TrajectorySamplingCfg

import fdm.env_cfg.terrain_cfg as terrain_cfg
import fdm.mdp as mdp
from fdm.agents import PaperFigureAgentCfg
from fdm.env_cfg.env_cfg_base import TERRAIN_ANALYSIS_CFG
from fdm.env_cfg.env_cfg_base_mixed import PlannerObservationsCfg
from fdm.model.utils import TorchPolicyExporter
from fdm.runner import FDMRunner, FDMRunnerCfg
from fdm.utils.args_cli_utils import cfg_modifier_pre_init, env_modifier_post_init, robot_changes, runner_cfg_init


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def export_to_jit(runner: FDMRunner):
    with torch.inference_mode():
        torch_policy_exporter = TorchPolicyExporter(runner, device="cuda")
        resume_path = get_checkpoint_path(
            runner.trainer.log_root_path, runner.trainer.cfg.load_run, runner.trainer.cfg.load_checkpoint
        )
        dir_path, _ = os.path.split(resume_path)

        torch_policy_exporter.export(os.path.join(dir_path, "export"), "model_cuda_jit.pth")
        torch_policy_exporter_cpu = TorchPolicyExporter(runner, device="cpu")
        torch_policy_exporter_cpu.export(os.path.join(dir_path, "export"), "model_cpu_jit.pth")

        # load jit models
        jit_model_cuda = torch.jit.load(os.path.join(dir_path, "export", "model_cuda_jit.pth")).to(runner.model.device)
        jit_model_cuda.eval()
        jit_model_cpu = torch.jit.load(os.path.join(dir_path, "export", "model_cpu_jit.pth")).to("cpu")
        jit_model_cpu.eval()

        example_input = [torch.rand(curr_size) for curr_size in torch_policy_exporter.example_input]
        _ = jit_model_cuda(example_input)
        _ = runner.model(example_input)
        _ = jit_model_cpu(example_input)

        # benchmark
        timer_model = benchmark.Timer(
            stmt="jit_model_cuda(example_input)",
            globals={"jit_model_cuda": jit_model_cuda, "example_input": example_input},
            num_threads=1,
            label="jit_model_cuda",
        )
        time_value = timer_model.blocked_autorange().median
        print("\tTime for model (jit, cuda)\t :", time_value / 1e-6, "us")

        timer_model = benchmark.Timer(
            stmt="model(example_input)",
            globals={"model": runner.model, "example_input": example_input},
            num_threads=1,
            label="runner.model",
        )
        time_value = timer_model.blocked_autorange().median
        print("\tTime for model (torch, cuda)\t :", time_value / 1e-6, "us")

        timer_model = benchmark.Timer(
            stmt="jit_model_cpu(example_input)",
            globals={"jit_model_cpu": jit_model_cpu, "example_input": example_input},
            num_threads=1,
            label="jit_model_cpu",
        )
        time_value = timer_model.blocked_autorange().median
        print("\tTime for model (jit, cpu)\t :", time_value / 1e-6, "us")

        runner.model.device = "cpu"
        runner.model.proprioceptive_normalizer.to("cpu")
        timer_model = benchmark.Timer(
            stmt="model(example_input)",
            globals={"model": runner.model.to("cpu"), "example_input": example_input},
            num_threads=1,
            label="runner.model",
        )
        time_value = timer_model.blocked_autorange().median
        print("\tTime for model (torch, cpu)\t :", time_value / 1e-6, "us")

    runner.model.device = "cuda"
    runner.model.proprioceptive_normalizer.to("cuda")
    runner.model.to(runner.model.device)


def _make_default_record_cameras(runner: FDMRunner):
    """Create a default camera set for ALL modes when --record is enabled."""
    cameras = []

    cam = Camera(
        prim_path="/World/record_camera",
        resolution=(1920, 1080),
    )
    # 一个通用视角：斜后上方看原点附近（足够先跑通）
    cam.set_world_pose(
        position=[-6.0, 0.0, 4.0],
        orientation=[0.9250441, 0.0, 0.3798598, 0.0],
    )
    cam.initialize()
    cameras.append(cam)

    return cameras


def main():
    runner = None
    try:
        # init runner cfg
        cfg = runner_cfg_init(args_cli)
        cfg = robot_changes(cfg, args_cli)
        cfg = cfg_modifier_pre_init(cfg, args_cli)

        # overwrite some configs for easier debugging
        cfg.replay_buffer_cfg.trajectory_length = 50 if not args_cli.paper_figure and not args_cli.paper_platform_figure else 15
        cfg.trainer_cfg.num_samples = 2000
        cfg.trainer_cfg.logging = False

        # swap environment
        cfg.env_cfg.scene.terrain.terrain_type = "generator"
        if args_cli.paper_figure:
            cfg.env_cfg.scene.terrain.terrain_generator = terrain_cfg.PAPER_FIGURE_TERRAIN_CFG
        elif args_cli.paper_platform_figure:
            cfg.env_cfg.scene.terrain.terrain_generator = terrain_cfg.PAPER_PLATFORM_FIGURE_TERRAIN_CFG
        else:
            cfg.env_cfg.scene.terrain.terrain_generator = terrain_cfg.PILLAR_TERRAIN_EVAL_CFG

        # debug for baseline
        if args_cli.env == "baseline":
            cfg.env_cfg.scene.terrain.terrain_generator = terrain_cfg.FDM_TERRAINS_CFG
            cfg.env_cfg.scene.terrain.terrain_generator.num_cols = 6
            cfg.env_cfg.scene.terrain.terrain_generator.num_rows = 12

        cfg.env_cfg.scene.terrain.random_seed = 0
        cfg.env_cfg.scene.terrain.regular_spawning = True

        # set name of the run
        if args_cli.runs is not None:
            cfg.trainer_cfg.load_run = args_cli.runs[0] if isinstance(args_cli.runs, list) else args_cli.runs

        # set regular spawning pattern
        if args_cli.paper_figure or args_cli.paper_platform_figure:
            cfg.env_cfg.events.reset_base.func = mdp.reset_root_state_paper_plot
        else:
            cfg.env_cfg.events.reset_base.func = mdp.reset_root_state_center

        pop_items = [item for item in cfg.env_cfg.events.reset_base.params.keys() if item != "asset_cfg"]
        for item in pop_items:
            cfg.env_cfg.events.reset_base.params.pop(item)

        # restrict agent to be purely random, temporal-correlated actions with adjusted horizon
        if args_cli.paper_figure or args_cli.paper_platform_figure:
            cfg.agent_cfg = PaperFigureAgentCfg(
                horizon=cfg.model_cfg.prediction_horizon + 1, platform_figure=args_cli.paper_platform_figure
            )
        else:
            cfg.agent_cfg = FDMRunnerCfg().agent_cfg
            cfg.agent_cfg.horizon = cfg.model_cfg.prediction_horizon + 1

        # add planner observations
        cfg.env_cfg.observations.planner_obs = PlannerObservationsCfg.PlannerObsCfg()

        # restrict goal generator
        cfg.env_cfg.commands.command = mdp.GoalCommandCfg(
            resampling_time_range=(1000000.0, 1000000.0),
            sampling_mode="bounded",
            debug_vis=False,
            traj_sampling=TrajectorySamplingCfg(terrain_analysis=TERRAIN_ANALYSIS_CFG),
        )
        cfg.env_cfg.observations.planner_obs.goal.func = mdp.goal_command_w_se2
        cfg.env_cfg.observations.planner_obs.goal.params = {"command_name": "command"}

        # remove reset when in collision
        cfg.env_cfg.terminations.base_contact = None

        # add distance light to the scene
        if args_cli.paper_figure:
            cfg.env_cfg.scene.light_1 = AssetBaseCfg(
                prim_path="/World/light_1",
                spawn=sim_spawners.SphereLightCfg(color=(1.0, 1.0, 1.0), intensity=200, exposure=8.0),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -6.0, 10.0)),
            )
            cfg.env_cfg.scene.light_2 = AssetBaseCfg(
                prim_path="/World/light_2",
                spawn=sim_spawners.SphereLightCfg(color=(1.0, 1.0, 1.0), intensity=200, exposure=8.0),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -18.0, 10.0)),
            )
            cfg.env_cfg.scene.light_3 = AssetBaseCfg(
                prim_path="/World/light_3",
                spawn=sim_spawners.SphereLightCfg(color=(1.0, 1.0, 1.0), intensity=200, exposure=8.0),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 6.0, 10.0)),
            )
            cfg.env_cfg.scene.light_4 = AssetBaseCfg(
                prim_path="/World/light_4",
                spawn=sim_spawners.SphereLightCfg(color=(1.0, 1.0, 1.0), intensity=200, exposure=8.0),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 18.0, 10.0)),
            )
            cfg.env_cfg.scene.light.spawn.intensity = 2000.0
        elif args_cli.paper_platform_figure:
            cfg.env_cfg.scene.light_1 = AssetBaseCfg(
                prim_path="/World/light_1",
                spawn=sim_spawners.SphereLightCfg(color=(1.0, 1.0, 1.0), intensity=30, exposure=8.0),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 10.0)),
            )
            cfg.env_cfg.scene.light.spawn.intensity = 2000.0

        # setup runner
        runner = FDMRunner(cfg=cfg, args_cli=args_cli, eval=True)

        # post modify runner and env
        runner = env_modifier_post_init(runner, args_cli=args_cli)

        # export model to jit (optional; keep as you had)
        if args_cli.env != "baseline" and not args_cli.paper_figure and not args_cli.paper_platform_figure:
            export_to_jit(runner)

        # ----------------------------
        # Cameras for recording (ALL MODES)
        # ----------------------------
        cameras = None

        if args_cli.record:
            # paper modes: keep your original camera logic
            if args_cli.paper_figure or args_cli.paper_platform_figure:
                import omni  # noqa: F401
                from PIL import Image  # noqa: F401

                if args_cli.paper_figure:
                    cameras = []
                    for idx in range(runner.env.scene.terrain.terrain_origins.shape[1]):
                        cam = Camera(
                            prim_path=f"/World/floating_camera_{idx}",
                            resolution=(3600, 2430),
                        )
                        cam.set_world_pose(
                            position=(
                                runner.env.scene.terrain.terrain_origins[0, idx]
                                + torch.tensor([25, 0.0, 28], device=runner.env.device)
                            ).tolist(),
                            orientation=[0.4146932, 0.0, 0.9099613, 0.0],
                        )
                        cam.initialize()
                        cameras.append(cam)
                else:
                    camera = Camera(prim_path="/World/floating_camera", resolution=(3600, 2430))
                    camera.set_world_pose(position=[-32, 0.0, 32], orientation=[0.9250441, 0.0, 0.3798598, 0.0])
                    camera.initialize()
                    camera_rob_1 = Camera(prim_path="/World/floating_camera_robot_1", resolution=(3600, 2430))
                    camera_rob_1.set_world_pose(position=[-5, 0.0, 4], orientation=[0.9250441, 0.0, 0.3798598, 0.0])
                    camera_rob_1.initialize()
                    camera_rob_2 = Camera(prim_path="/World/floating_camera_robot_2", resolution=(3600, 2430))
                    camera_rob_2.set_world_pose(position=[-5, 0.0, 4], orientation=[0.9250441, 0.0, 0.3798598, 0.0])
                    camera_rob_2.initialize()
                    cameras = [camera, camera_rob_1, camera_rob_2]
            else:
                # ✅ 非 paper 模式：也必须给默认相机，才能录视频
                cameras = _make_default_record_cameras(runner)

        # ----------------------------
        # Run test
        # ----------------------------
        if args_cli.record:
            runner.test(cameras=cameras)
        else:
            runner.test()

        # If not recording, keep simulation alive
        if not args_cli.record:
            print("Simulation will keep running until the user closes it.")
            while simulation_app.is_running():
                runner.env.render()

    finally:
        if runner is not None:
            runner.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
