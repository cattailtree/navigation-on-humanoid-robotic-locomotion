# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import os
import pickle
import prettytable
import random
import statistics
import time
import torch
from copy import deepcopy
from dataclasses import MISSING
from torch.utils.data import DataLoader

import carb
import cv2
import hydra
import omegaconf
import pypose as pp
import wandb
import fdm.mdp as mdp
import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path

from fdm import LARGE_UNIFIED_HEIGHT_SCAN, PAPER_COLORS_RGBA_F, VEL_RANGE_X, VEL_RANGE_Y
from fdm.data_buffers import ReplayBuffer
from fdm.utils.colors import generate_colors

from ..agents import MixedAgent
from ..model import FDMModel, FDMModelVelocityMultiStep, FDMProprioceptionModel, FDMProprioceptionVelocityModel
from ..planner import SimpleSE2TrajectoryOptimizer, get_planner_cfg
from .runner_cfg import FDMRunnerCfg
from .trainer import Trainer

# can only be imported if gui activated
try:
    from isaacsim.util.debug_draw import _debug_draw as omni_debug_draw
except ImportError:
    omni_debug_draw = None


class FDMRunner:
    def __init__(self, cfg: FDMRunnerCfg, args_cli, eval: bool = False, render_mode: str | None = None):
        self.cfg = cfg
        self.args_cli = args_cli
        self.eval = eval
        self.render_mode = render_mode

        # set drawing parameters
        self.nb_draw_traj = 10
        self.step_draw_traj = 2

        # update cfg
        self.cfg.env_cfg.scene.num_envs = self.args_cli.num_envs
        if hasattr(args_cli, "run_name"):
            self.cfg.trainer_cfg.run_name = (
                self.cfg.trainer_cfg.run_name + self.args_cli.run_name
                if isinstance(self.cfg.trainer_cfg.run_name, str)
                else self.args_cli.run_name
            )

        if self.eval:
            self.cfg.trainer_cfg.resume = True
            self.cfg.trainer_cfg.logging = False

            # check if multiple runs are passed
            self.eval_multi_run = (
                hasattr(args_cli, "runs") and isinstance(args_cli.runs, list) and len(args_cli.runs) > 1
            )
            # init draw colors
            if not self.eval_multi_run:
                self.safe_colors = [
                    PAPER_COLORS_RGBA_F["ours"]
                ] * self.nb_draw_traj  # generate_colors(self.nb_draw_traj, start_hue=0.3, end_hue=0.4)
                self.collision_colors = [
                    PAPER_COLORS_RGBA_F["collision"]
                ] * self.nb_draw_traj  # generate_colors(self.nb_draw_traj, start_hue=0.0, end_hue=0.05)
                self.trajectory_color = [PAPER_COLORS_RGBA_F["future_traj"]]
                self.perfect_velocity_color = [PAPER_COLORS_RGBA_F["constant_vel"]]
            else:
                # separate hue value for the number of runs (safe + collision and path color)
                hue_step = 1.0 / (2 * len(args_cli.runs) + 1)
                colors = [
                    generate_colors(
                        self.nb_draw_traj,
                        start_hue=hue_step * step - 0.25 * hue_step,
                        end_hue=hue_step * step + 0.25 * hue_step,
                    )
                    for step in range(1, 2 * len(args_cli.runs) + 2)
                ]
                self.safe_colors = colors[: len(args_cli.runs)]
                self.collision_colors = colors[len(args_cli.runs) : -1]
                self.trajectory_color = [colors[-1][4]]

            if omni_debug_draw is not None:
                # init debug draw
                self.draw_interface = omni_debug_draw.acquire_debug_draw_interface()

        # override the resampling command of the command generator with `trainer_cfg.command_timestep`
        self.cfg.env_cfg.episode_length_s = self.cfg.model_cfg.command_timestep * (
            self.cfg.replay_buffer_cfg.trajectory_length + 1
        )

        # setup
        self.setup()



    """
    Properties
    """

    @property
    def device(self) -> str:
        """The device to use for training."""
        return self.env.device

    """
    Operations
    """
    def setup(self):
        # ------------------------------------------------------------------
        # 1. save noise cfgs
        # ------------------------------------------------------------------
        proprioceptive_noise_cfg = {
            obs_name: getattr(self.cfg.env_cfg.observations.fdm_obs_proprioception, obs_name).noise
            for obs_name, obs_cfg in self.cfg.env_cfg.observations.fdm_obs_proprioception.to_dict().items()
            if isinstance(obs_cfg, dict) and "noise" in obs_cfg
        }
        exteroceptive_noise_cfg = deepcopy(
            self.cfg.env_cfg.observations.fdm_obs_exteroceptive.env_sensor.noise
        )

        # ------------------------------------------------------------------
        # 2. create env FIRST
        # ------------------------------------------------------------------
        # default spawn z: use env var FDM_FORCE_SPAWN_Z if set, otherwise default to 0.8
        force_z_env = os.getenv("FDM_FORCE_SPAWN_Z", None)
        try:
            self.force_spawn_z = float(force_z_env) if force_z_env is not None else 0.8
        except Exception:
            self.force_spawn_z = 0.8

        self.env: ManagerBasedRLEnv = ManagerBasedRLEnv(
            self.cfg.env_cfg, render_mode=self.render_mode
        )

        # ------------------------------------------------------------------
        # 3. compute proprio dim BEFORE model construction
        # ------------------------------------------------------------------
        dims = self.env.observation_manager.group_obs_term_dim["fdm_obs_proprioception"]

        def _numel(x):
            if isinstance(x, (list, tuple)):
                n = 1
                for v in x:
                    n *= int(v)
                return n
            return int(x)

        if isinstance(dims, (list, tuple)):
            D = sum(_numel(d) for d in dims)
        else:
            D = _numel(dims)

        print(f"[DEBUG] env proprio dim (computed) = {D}  from dims={dims}")

        # ------------------------------------------------------------------
        # 4. IMPORTANT: write D into model_cfg BEFORE instantiating model
        # ------------------------------------------------------------------
        # ⚠️ 字段名必须是 model 真正使用的那个
        # 如果你 encoder 那里用的是 cfg.proprioceptive_dim，这一行就是对的
        self.cfg.model_cfg.proprioceptive_dim = D

        # ------------------------------------------------------------------
        # 5. NOW instantiate model (encoder input_size will be correct)
        # ------------------------------------------------------------------
        self.model: (
            FDMModel
            | FDMModelVelocityMultiStep
            | FDMProprioceptionModel
            | FDMProprioceptionVelocityModel
        ) = self.cfg.model_cfg.class_type(
            cfg=self.cfg.model_cfg, device=self.device
        )
        self.model.to(self.device)

        print(
            "[DEBUG] encoder input_size =",
            self.model.state_obs_proprioceptive_encoder.input_size,
        )

        # ------------------------------------------------------------------
        # 6. replay buffer
        # ------------------------------------------------------------------
        self.replay_buffer = ReplayBuffer(
            self.cfg.replay_buffer_cfg, self.model.cfg, self.env
        )

        # ------------------------------------------------------------------
        # 7. trainer (uses D, but no longer controls encoder size)
        # ------------------------------------------------------------------
        self.trainer = Trainer(
            cfg=self.cfg.trainer_cfg,
            replay_buffer_cfg=self.cfg.replay_buffer_cfg,
            model=self.model,
            device=self.device,
            eval=self.eval,
            proprioceptive_noise_cfg=proprioceptive_noise_cfg,
            proprioceptive_dim=D,
            exteroceptive_noise_cfg=exteroceptive_noise_cfg,
        )

        # ------------------------------------------------------------------
        # 8. FIX normalizer (optional but correct)
        # ------------------------------------------------------------------
        if hasattr(self.model, "proprioceptive_normalizer"):
            NormCls = self.model.proprioceptive_normalizer.__class__
            self.model.proprioceptive_normalizer = NormCls(shape=(D,)).to(self.device)

        # ------------------------------------------------------------------
        # 9. feet contact buffers
        # ------------------------------------------------------------------
        self.feet_idx, _ = self.env.scene.sensors["contact_forces"].find_bodies(
            self.cfg.body_regex_contact_checking
        )
        #print("[DEBUG] feet_idx =", self.feet_idx)
        #print("[DEBUG] feet_idx len =", len(self.feet_idx))

        self.feet_contact = torch.zeros(
            (self.env.num_envs, len(self.feet_idx)),
            dtype=torch.bool,
            device=self.env.device,
        )
        self.feet_non_contact_counter = torch.zeros(
            self.env.num_envs, dtype=torch.long, device=self.env.device
        )
        # setup planning agent
        if self.cfg.agent_cfg is not MISSING and self.cfg.agent_cfg is not None:
            self.agent = self.cfg.agent_cfg.class_type(self.cfg.agent_cfg, self)
        else:
            self.agent = None

        # add entire config to wandb
        if not self.eval and self.cfg.trainer_cfg.logging:
            save_cfg = deepcopy(self.cfg).to_dict()
            dump_yaml(filename=f"{self.trainer.log_dir}/params/config.yaml", data=save_cfg)
            save_cfg.pop("trainer_cfg")
            wandb.config.update(save_cfg)

        print("[INFO]: Setup complete.")

    def train(self):
        # buffers
        train_loss_list = torch.zeros(self.cfg.collection_rounds, device=self.device)
        val_loss_list = torch.zeros(self.cfg.collection_rounds, device=self.device)

        # collect validation dataset
        self._collect(eval=True)

        # set learning rate progress step
        if hasattr(self.model, "set_learning_progress_step"):
            self.model.set_learning_progress_step(1 / (self.cfg.collection_rounds * self.cfg.trainer_cfg.epochs))

        for collection_round in range(self.cfg.collection_rounds):
            # collect data
            self._collect()

            # dump the training dataset
            with open(f"{self.trainer.log_dir}/train_dataset_{collection_round}.pkl", "wb") as train_dataset:
                pickle.dump(self.trainer.train_dataset, train_dataset)

            # train model
            train_loss_list[collection_round], val_loss_list[collection_round] = self.trainer.train(collection_round)
            # save model
            path = self.trainer.model.get_model_path(
                self.trainer.log_dir, "_collection_round_" + str(collection_round).zfill(2)
            )
            self.trainer.model.save(path)

        # print losses as table
        table = prettytable.PrettyTable()
        table.field_names = ["Collection Round", "Train Loss", "Val Loss"]
        for i in range(self.cfg.collection_rounds):
            table.add_row([i, train_loss_list[i], val_loss_list[i]])
        print(table)

        # save losses
        dump_yaml(
            filename=f"{self.trainer.log_dir}/losses.yaml",
            data={"train_loss": train_loss_list.tolist(), "val_loss": val_loss_list.tolist()},
        )

    def eval_metric(self) -> tuple[dict, dict]:
        # reset the environment
        with torch.inference_mode():
            # reset env
            self.env.reset(1)
            try:
                robot = self.env.scene.articulations["robot"]
                p = robot.data.root_pos_w[:5, 2].detach().cpu().numpy()
                origins = self.env.scene.env_origins[:5].cpu().numpy()
                #print(f"[RESET-DBG] post-reset root_z[:5]={p} env_origins[:5]={origins}", flush=True)
                # apply configured/default spawn z (self.force_spawn_z)
                try:
                    if getattr(self, "force_spawn_z", None) is not None:
                        desired_z = float(self.force_spawn_z)
                        # modify world positions directly (write world coordinates)
                        world_pos = robot.data.root_pos_w.clone()
                        origins = self.env.scene.env_origins.to(world_pos.device)
                        # per-env world z = env_origin.z + offset
                        world_pos[:, 2] = origins[:, 2] + desired_z
                        orientations = robot.data.root_quat_w.clone()
                        env_ids = torch.arange(world_pos.shape[0], device=world_pos.device)
                        robot.write_root_pose_to_sim(torch.cat([world_pos, orientations], dim=-1), env_ids=env_ids)
                        try:
                            zero_vel = torch.zeros((world_pos.shape[0], 6), device=world_pos.device)
                            robot.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
                        except Exception:
                            pass
                        #print(f"[FORCE-DBG] forced world_z_offset={desired_z} wrote world_z[:5]={world_pos[:5,2].cpu().numpy()}", flush=True)
                except Exception as e:
                    print(f"[FORCE-DBG] failed to force spawn z: {e}", flush=True)
            except Exception:
                pass
        # collect data
        if self.env.cfg.scene.terrain.terrain_type == "usd":
            # get dataset path
            _, terrain_name = os.path.split(self.env.cfg.scene.terrain.usd_path)
            log_dir, _ = os.path.split(self.trainer.log_root_path)
            terrain_name = os.path.splitext(terrain_name)[0]

            # follow naming convention of test datasets (used in args_cli_utils.py)
            suffix = ""
            if self.args_cli.env != "baseline":
                if hasattr(self.args_cli, "reduced_obs") and self.args_cli.reduced_obs:
                    suffix += "_reducedObs"
                if hasattr(self.args_cli, "remove_torque") and self.args_cli.remove_torque:
                    suffix += "_noTorque"
                if hasattr(self.args_cli, "noise") and self.args_cli.noise:
                    suffix += "_noise"
                elif hasattr(self.args_cli, "occlusions") and self.args_cli.occlusions:
                    suffix += "_occlusions"
            else:
                if hasattr(self.args_cli, "noise") and self.args_cli.noise:
                    suffix = "_noise_baseline"
                else:
                    suffix = "_baseline"
            if LARGE_UNIFIED_HEIGHT_SCAN:
                suffix += "_largeHeightScan"
            if hasattr(self.args_cli, "ablation_mode") and self.args_cli.ablation_mode is not None:
                suffix += f"_ablation_{self.args_cli.ablation_mode}"

            eval_dataset_path = os.path.join(log_dir, "test_datasets", f"{terrain_name}{suffix}_dataset.pkl")

            if not os.path.isfile(eval_dataset_path):
                self._collect(eval=False)
                # make sure directory exists
                os.makedirs(os.path.dirname(eval_dataset_path), exist_ok=True)
                # save dataset
                with open(eval_dataset_path, "wb") as eval_dataset:
                    pickle.dump(self.trainer.train_dataset, eval_dataset)
                print(f"[INFO]: Data saved to {eval_dataset_path}")
            else:
                print("[INFO]: Using existing eval dataset.")
                # load dataset
                with open(eval_dataset_path, "rb") as test_dataset:
                    self.trainer.dataloader = DataLoader(
                        pickle.load(test_dataset),
                        batch_size=self.cfg.trainer_cfg.batch_size,
                        shuffle=False,
                        num_workers=self.cfg.trainer_cfg.num_workers,
                        pin_memory=True,
                    )
        else:
            self._collect(eval=False)

        # evaluate model
        return self.trainer.evaluate(self.trainer.dataloader, plot_mode=True)

    def evaluate(self):
        """Run the visual evaluation of the model.

        Args:
            initial_warm_up (bool, optional): Let the environments run until the history buffers are filled. Then
                performs a first prediciton and return the results. Defaults to True.

        """
        # reset the environment
        with torch.inference_mode():
            obs, _ = self.env.reset(1)
            try:
                robot = self.env.scene.articulations["robot"]
                p = robot.data.root_pos_w[:5, 2].detach().cpu().numpy()
                origins = self.env.scene.env_origins[:5].cpu().numpy()
                #print(f"[RESET-DBG] eval post-reset root_z[:5]={p} env_origins[:5]={origins}", flush=True)
                try:
                    if getattr(self, "force_spawn_z", None) is not None:
                        desired_z = float(self.force_spawn_z)
                        world_pos = robot.data.root_pos_w.clone()
                        origins = self.env.scene.env_origins.to(world_pos.device)
                        world_pos[:, 2] = origins[:, 2] + desired_z
                        orientations = robot.data.root_quat_w.clone()
                        env_ids = torch.arange(world_pos.shape[0], device=world_pos.device)
                        robot.write_root_pose_to_sim(torch.cat([world_pos, orientations], dim=-1), env_ids=env_ids)
                        try:
                            zero_vel = torch.zeros((world_pos.shape[0], 6), device=world_pos.device)
                            robot.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
                        except Exception:
                            pass
                        #print(f"[FORCE-DBG] forced world_z_offset={desired_z} wrote world_z[:5]={world_pos[:5,2].cpu().numpy()}", flush=True)
                except Exception as e:
                    print(f"[FORCE-DBG] failed to force spawn z: {e}", flush=True)
            except Exception:
                pass
        # reset agent
        actions = self.agent.reset(obs)

        # make actions equal in the case of an equal agent
        if self.args_cli.equal_actions:
            actions[:] = actions[0]

        # buffer to save trajectories for plotting
        pred_trajectories = {x: [] for x in range(self.env.num_envs)}
        pred_collision = {x: [] for x in range(self.env.num_envs)}
        final_pred_error = {x: [] for x in range(self.env.num_envs)}
        # step counter
        counter = 0
        meta_eval: dict(str, list(float)) = {}

        # reset feet contact
        self.feet_contact[:] = False
        self.feet_non_contact_counter[:] = 0

        # if multiple runs are passed
        models = []
        if self.eval_multi_run:
            # load the different models
            models = [deepcopy(self.trainer.model)]
            for run in self.args_cli.runs[1:]:
                resume_path = get_checkpoint_path(self.trainer.log_root_path, run, self.trainer.cfg.load_checkpoint)
                self.trainer.model.load(resume_path)
                self.trainer.model.eval()
                models.append(deepcopy(self.trainer.model))
                print(f"[INFO]: Loaded model checkpoint from: {resume_path}")
            # expand buffers for number of models, deepcopy to make sure they are not linked
            pred_trajectories = [deepcopy(pred_trajectories) for _ in range(len(self.args_cli.runs))]
            pred_collision = [deepcopy(pred_collision) for _ in range(len(self.args_cli.runs))]
            final_pred_error = [deepcopy(final_pred_error) for _ in range(len(self.args_cli.runs))]
            meta_eval = [deepcopy(meta_eval) for _ in range(len(self.args_cli.runs))]
            counter = [deepcopy(counter) for _ in range(len(self.args_cli.runs))]

        while not self.replay_buffer.is_filled:
            # step environment
            with torch.inference_mode():
                obs, _, dones, _, _ = self.env.step(actions)
            # also mark every env as done where the replay buffer is filled
            dones = dones | self.replay_buffer.env_buffer_filled.to(self.device)

            ###
            # Determine feet contact
            ###
            # Note: only start recording and changing actions when all feet have touched the ground
            feet_all_contact, dones, obs_new = self._feet_contact_handler(dones)
            obs = obs_new if obs_new is not None else obs

            ###
            # Get the actions
            ###
            # replan for environments where actions run out of horizon - normally done in agent.act but not with the horizon
            env_to_replan = self.agent._ALL_INDICES[
                self.agent._plan_step >= (self.agent.cfg.horizon - self.cfg.model_cfg.prediction_horizon)
            ]
            leftover_actions = self.agent._plan[
                env_to_replan, (self.agent.cfg.horizon - self.cfg.model_cfg.prediction_horizon) :
            ].clone()
            self.agent.plan(env_ids=env_to_replan, obs=obs, random_init=False)
            self.agent._plan_step[env_to_replan] = 0
            self.agent._plan[env_to_replan, : leftover_actions.shape[1]] = leftover_actions
            # plan actions
            actions = self.agent.act(obs, dones.to(torch.bool).clone(), feet_contact=feet_all_contact)
            # make actions equal in the case of an equal agent
            if self.args_cli.equal_actions:
                actions[:] = actions[0]

            ###
            # update replay buffer and get completed predictions
            ###
            dones = dones.to(self.replay_buffer.device)
            self.replay_buffer.add(
                states=obs["fdm_state"].clone(),
                obersevations_proprioceptive=obs["fdm_obs_proprioception"].clone(),
                obersevations_exteroceptive=(
                    obs["fdm_obs_exteroceptive"].clone() if "fdm_obs_exteroceptive" in obs else None
                ),
                actions=actions.clone(),
                dones=dones.to(torch.bool).clone(),
                feet_contact=feet_all_contact,
                add_observation_exteroceptive=(
                    obs["fdm_add_obs_exteroceptive"] if "fdm_add_obs_exteroceptive" in obs else None
                ),
            )
            if torch.any(dones):
                # for done environments reset replay_buffer
                self.replay_buffer.reset(env_ids=self.replay_buffer._ALL_INDICES[dones])
                # reset saved trajectories
                if not self.eval_multi_run:
                    [pred_trajectories[env_id.item()].clear() for env_id in self.replay_buffer._ALL_INDICES[dones]]
                    [pred_collision[env_id.item()].clear() for env_id in self.replay_buffer._ALL_INDICES[dones]]
                    [final_pred_error[env_id.item()].clear() for env_id in self.replay_buffer._ALL_INDICES[dones]]
                else:
                    [
                        [
                            pred_trajectories[run_idx][env_id.item()].clear()
                            for run_idx in range(len(self.args_cli.runs))
                        ]
                        for env_id in self.replay_buffer._ALL_INDICES[dones]
                    ]
                    [
                        [pred_collision[run_idx][env_id.item()].clear() for run_idx in range(len(self.args_cli.runs))]
                        for env_id in self.replay_buffer._ALL_INDICES[dones]
                    ]
                    [
                        [final_pred_error[run_idx][env_id.item()].clear() for run_idx in range(len(self.args_cli.runs))]
                        for env_id in self.replay_buffer._ALL_INDICES[dones]
                    ]
            # decide for which environments to do a new prediction
            # note: first time prediction can be done when fill idx is increased to 1, next when it reaches 1+prediction_horizon, etc.
            env_new_prediction = self.replay_buffer._ALL_INDICES[
                self.replay_buffer.env_step_counter % int(self.replay_buffer.data_collection_interval) == 0
            ]
            env_new_prediction = env_new_prediction[self.replay_buffer.fill_idx[env_new_prediction] > 0]
            if not torch.any(env_new_prediction):
                continue

            ###
            # Loss, Drawing and Info print
            ###
            # clear previous drawings
            self.draw_interface.clear_lines()
            # if multiple self.args_cli.runs are passed, make predictions for all of them
            if not self.eval_multi_run:
                # update predictions and save them for plotting
                model_out = self._eval_predict(env_new_prediction)
                # calculate loss
                if (
                    self.model.cfg.class_type == FDMProprioceptionModel
                    or self.model.cfg.class_type == FDMProprioceptionVelocityModel
                ):
                    # pred_collision is here actually the friction
                    pred_trajectories, pred_collision, final_pred_error, meta_eval, counter = (
                        self._eval_loss_proprioception(
                            env_new_prediction,
                            model_out,
                            pred_trajectories,
                            pred_collision,
                            final_pred_error,
                            meta_eval,
                            counter,
                        )
                    )
                    # for plotting set the collision estimation to None
                    draw_collision_pred = None
                else:
                    pred_trajectories, pred_collision, final_pred_error, meta_eval, counter = self._eval_loss_fdm(
                        env_new_prediction,
                        model_out,
                        pred_trajectories,
                        pred_collision,
                        final_pred_error,
                        meta_eval,
                        counter,
                    )
                    draw_collision_pred = pred_collision
                # update drawing of prediction and walked trajectories
                self._draw_trajectories(
                    pred_trajectories,
                    draw_collision_pred,
                    final_pred_error,
                    safe_colors=self.safe_colors,
                    collision_colors=self.collision_colors,
                    draw_trajectory=True,
                )
                # print loss information
                if counter > 1000:
                    # print losses as table
                    table = prettytable.PrettyTable()
                    table.field_names = [f"Loss ({counter} predictions)", "Mean Value", "Std Value"]
                    for key, value in meta_eval.items():
                        table.add_row([key, statistics.mean(value[5:]), statistics.stdev(value[5:])])
                    print(table)
                    # reset loss
                    meta_eval = {}
                    # reset counter
                    counter = 0
            else:
                for run_idx, run in enumerate(self.args_cli.runs):
                    # update predictions and save them for plotting
                    model_out = self._eval_predict(env_new_prediction, models[run_idx])
                    # calculate loss
                    if (
                        self.model.cfg.class_type == FDMProprioceptionModel
                        or self.model.cfg.class_type == FDMProprioceptionVelocityModel
                    ):
                        # pred_collision is here actually the friction
                        (
                            pred_trajectories[run_idx],
                            pred_collision[run_idx],
                            final_pred_error[run_idx],
                            meta_eval[run_idx],
                            counter[run_idx],
                        ) = self._eval_loss_proprioception(
                            env_new_prediction,
                            model_out,
                            pred_trajectories[run_idx],
                            pred_collision[run_idx],
                            final_pred_error[run_idx],
                            meta_eval[run_idx],
                            counter[run_idx],
                        )
                        # for plotting set the collision estimation to None
                        draw_collision_pred = None
                    else:
                        (
                            pred_trajectories[run_idx],
                            pred_collision[run_idx],
                            final_pred_error[run_idx],
                            meta_eval[run_idx],
                            counter[run_idx],
                        ) = self._eval_loss_fdm(
                            env_new_prediction,
                            model_out,
                            pred_trajectories[run_idx],
                            pred_collision[run_idx],
                            final_pred_error[run_idx],
                            meta_eval[run_idx],
                            counter[run_idx],
                        )
                        draw_collision_pred = pred_collision[run_idx]
                    # update drawing of prediction and walked trajectories
                    self._draw_trajectories(
                        pred_trajectories[run_idx],
                        draw_collision_pred,
                        final_pred_error[run_idx],
                        safe_colors=self.safe_colors[run_idx],
                        collision_colors=self.collision_colors[run_idx],
                        draw_trajectory=run_idx == 0,
                    )
                    # print loss information
                    if counter[run_idx] > 1000:
                        # print losses as table
                        table = prettytable.PrettyTable()
                        table.field_names = ["Model", "Loss (1000 predictions)", "Mean Value", "Std Value"]
                        for key, value in meta_eval[run_idx].items():
                            table.add_row([run, key, statistics.mean(value[5:]), statistics.stdev(value[5:])])
                        print(table)
                        # reset loss
                        meta_eval[run_idx] = {}
                        # reset counter
                        counter[run_idx] = 0

    def test(self, use_planner: bool = True, cameras: list = None):
        # set manual seed
        torch.manual_seed(0)
        # reset the environment
        with torch.inference_mode():
            obs, _ = self.env.reset(0)
            try:
                robot = self.env.scene.articulations["robot"]
                p = robot.data.root_pos_w[:5, 2].detach().cpu().numpy()
                origins = self.env.scene.env_origins[:5].cpu().numpy()
                print(f"[RESET-DBG] planner post-reset root_z[:5]={p} env_origins[:5]={origins}", flush=True)
                try:
                    if getattr(self, "force_spawn_z", None) is not None:
                        desired_z = float(self.force_spawn_z)
                        world_pos = robot.data.root_pos_w.clone()
                        origins = self.env.scene.env_origins.to(world_pos.device)
                        world_pos[:, 2] = origins[:, 2] + desired_z
                        orientations = robot.data.root_quat_w.clone()
                        env_ids = torch.arange(world_pos.shape[0], device=world_pos.device)
                        robot.write_root_pose_to_sim(torch.cat([world_pos, orientations], dim=-1), env_ids=env_ids)
                        try:
                            zero_vel = torch.zeros((world_pos.shape[0], 6), device=world_pos.device)
                            robot.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
                        except Exception:
                            pass
                        print(f"[FORCE-DBG] forced world_z_offset={desired_z} wrote world_z[:5]={world_pos[:5,2].cpu().numpy()}", flush=True)
                except Exception as e:
                    print(f"[FORCE-DBG] failed to force spawn z: {e}", flush=True)
            except Exception:
                pass
        # reset agent
        actions = self.agent.reset(obs)
        assert actions is not None, "agent.reset(obs) returned None"
        assert torch.is_tensor(actions), f"actions type is {type(actions)}"


        if hasattr(self.args_cli, "max_actions") and self.args_cli.max_actions:
            max_actions = torch.zeros_like(actions)
            max_actions[:, 0] = VEL_RANGE_X[1]
            max_actions[:, 1] = VEL_RANGE_Y[1]

        # make actions equal in the case of an equal agent
        if self.args_cli.equal_actions:
            actions[:] = actions[0]
        elif hasattr(self.args_cli, "max_actions") and self.args_cli.max_actions:
            actions[:] = max_actions

        # reset feet contact and counter
        self.feet_contact[:] = False
        self.feet_non_contact_counter[:] = 0

        # all predictions
        safe_predictions_start = []
        safe_predictions_end = []
        collision_predictions_start = []
        collision_predictions_end = []
        perfect_velocity_predictions_start = []
        perfect_velocity_predictions_end = []
        predicted_envs = torch.zeros(self.env.num_envs, dtype=torch.bool)
        reset_envs = torch.zeros(self.env.num_envs, dtype=torch.bool)

        # setup planner
        if use_planner:
            planner_cfg_dict = get_planner_cfg(
                num_envs=self.env.num_envs,
                traj_dim=self.cfg.model_cfg.prediction_horizon,
                debug=False,
                device=self.device,
            )
            planner_cfg = omegaconf.OmegaConf.create(planner_cfg_dict)
            planner: SimpleSE2TrajectoryOptimizer = hydra.utils.instantiate(planner_cfg.to)
            planner.set_fdm_classes(fdm_model=self.model, env=self.env)

        # extra imports when images should be recorded and define save paths
        if cameras is not None:
            # setup image save path
            resume_path = get_checkpoint_path(
                self.trainer.log_root_path, self.trainer.cfg.load_run, self.trainer.cfg.load_checkpoint
            )
            directory_path = os.path.dirname(resume_path)
            if hasattr(self.args_cli, "paper_platform_figure") and self.args_cli.paper_platform_figure:
                render_path = os.path.join(directory_path, "platform_render")
            else:
                render_path = os.path.join(directory_path, "dynamics_render")
            os.makedirs(render_path, exist_ok=True)
            cam_save_path = []
            for idx, camera in enumerate(cameras):
                cam_save_path.append(os.path.join(render_path, f"camera_{idx}"))
                os.makedirs(cam_save_path[-1], exist_ok=True)

            # counter for rendering
            render_counter = torch.zeros(self.env.num_envs, device=self.env.device, dtype=torch.int)

        while not self.replay_buffer.is_filled:
            robot = self.env.scene.articulations["robot"]
            p = robot.data.root_pos_w[0].detach().cpu()
            v = robot.data.root_lin_vel_w[0].detach().cpu()
            print("[DEBUG] root_z =", float(p[2]), "vz =", float(v[2]), flush=True)
            pelvis_z = float(robot.data.root_pos_w[0,2].item())
            # 用 ankle_roll_link 的刚体高度（需要你确认 body index）
            # 临时做法：直接打印所有 body z 的 min/max
            z_all = robot.data.body_pos_w[0,:,2].detach().cpu()
            print("[DEBUG] pelvis_z", pelvis_z, "body_z min/max", float(z_all.min()), float(z_all.max()), flush=True)

            with torch.inference_mode():
                obs, _, dones, _, _ = self.env.step(actions.clone())
                
            # --- DEBUG: 本步实际生效的顶层动作（SE2 cmd） ---
            if self.env.common_step_counter < 20:
                print("[DEBUG] step =", int(self.env.common_step_counter),
                    "actions[0][:3] =", actions[0, :3].detach().cpu().tolist(),
                    flush=True)
            robot = self.env.scene.articulations["robot"]
            v = robot.data.root_lin_vel_w[0].detach().cpu()
            w = robot.data.root_ang_vel_w[0].detach().cpu()
            print(f"[DEBUG] root v_w={v.tolist()}  w_w={w.tolist()}", flush=True)



            # also mark every env as done where the replay buffer is filled
            dones = dones | self.replay_buffer.env_buffer_filled.to(self.device)

            ###
            # Determine feet contact
            ###
            # Note: only start recording and changing actions when all feet have touched the ground
            feet_all_contact, dones, obs_new = self._feet_contact_handler(dones)
            obs = obs_new if obs_new is not None else obs

            ###
            # Get the actions
            ###
            # set plan to zero if in collision
            combined_condition = torch.logical_and(obs["fdm_state"][:, 7], feet_all_contact)
            combined_condition = torch.logical_and(combined_condition, predicted_envs.to(self.device))
            self.agent._plan[combined_condition] *= 0.0
            # plan actions
            actions = self.agent.act(obs, dones.to(torch.bool).clone(), feet_contact=feet_all_contact)
            # update reset envs, when done and already an prediction has been made
            reset_envs[torch.logical_and(self.agent._plan_step.cpu() == 0, predicted_envs)] = True
            actions[reset_envs] = 0.0

            if self.args_cli.equal_actions:
                actions[:] = actions[0]
            elif hasattr(self.args_cli, "max_actions") and self.args_cli.max_actions:
                actions[:] = max_actions

            ###
            # update replay buffer and get completed predictions
            ###
            dones = dones.to(self.replay_buffer.device)
            self.replay_buffer.add(
                states=obs["fdm_state"].clone(),
                obersevations_proprioceptive=obs["fdm_obs_proprioception"].clone(),
                obersevations_exteroceptive=(
                    obs["fdm_obs_exteroceptive"].clone() if "fdm_obs_exteroceptive" in obs else None
                ),
                actions=actions.clone(),
                dones=dones.to(torch.bool).clone(),
                feet_contact=feet_all_contact,
                add_observation_exteroceptive=(
                    obs["fdm_add_obs_exteroceptive"] if "fdm_add_obs_exteroceptive" in obs else None
                ),
            )
            # if torch.any(dones):
            #     # for done environments reset replay_buffer
            #     self.replay_buffer.reset(env_ids=self.replay_buffer._ALL_INDICES[dones])


            # decide for which environments to do a new prediction
            # note: first time prediction can be done when fill idx is increased to 1, next when it reaches 1+prediction_horizon, etc.
            env_new_prediction = self.replay_buffer._ALL_INDICES[
                self.replay_buffer.env_step_counter % int(self.replay_buffer.data_collection_interval) == 0
            ]
            env_new_prediction = env_new_prediction[self.replay_buffer.fill_idx[env_new_prediction] > 0]
            # only predict each environment once
            env_new_prediction = env_new_prediction[~predicted_envs[env_new_prediction]]

            if len(env_new_prediction) > 0:
                # mark environments as predicted
                predicted_envs[env_new_prediction] = True

                # update predictions and save them for plotting
                model_out = self._eval_predict(env_new_prediction)

                # append the initial state to the predictions for visualization
                # NOTE: the yaw component is not used, here we augment to fit to the sin, cos encoding of the model_out
                model_out[0] = torch.cat(
                    [obs["planner_obs"]["start"][env_new_prediction][:, [0, 1, 2, 2]].unsqueeze(1), model_out[0]], dim=1
                )

                if not self.model.cfg.unified_failure_prediction or (
                    self.args_cli.env == "baseline" and self.model.cfg.unified_failure_prediction
                ):
                    model_out[1] = model_out[1].max(dim=1)[0]

                # separate into collision and non-collision predictions
                states_coll = model_out[0][model_out[1] > self.cfg.model_cfg.collision_threshold]
                states_safe = model_out[0][model_out[1] <= self.cfg.model_cfg.collision_threshold]

                # separate into start and end points and flatten
                states_coll_start = states_coll[..., :-1, :3].reshape(-1, 3).cpu()
                states_coll_end = states_coll[..., 1:, :3].reshape(-1, 3).cpu()
                states_safe_start = states_safe[..., :-1, :3].reshape(-1, 3).cpu()
                states_safe_end = states_safe[..., 1:, :3].reshape(-1, 3).cpu()

                # elevate the prediction to the terrain height
                if use_planner:
                    states_coll_start[:, 2] = planner.terrain_analysis.get_height(states_coll_start[:, :2]) + 0.5
                    states_coll_end[:, 2] = planner.terrain_analysis.get_height(states_coll_end[:, :2]) + 0.5
                    states_safe_start[:, 2] = planner.terrain_analysis.get_height(states_safe_start[:, :2]) + 0.5
                    states_safe_end[:, 2] = planner.terrain_analysis.get_height(states_safe_end[:, :2]) + 0.5
                else:
                    states_coll_start[:, 2] = 0.5
                    states_coll_end[:, 2] = 0.5
                    states_safe_start[:, 2] = 0.5
                    states_safe_end[:, 2] = 0.5

                # save predictions
                safe_predictions_start.extend(states_safe_start.tolist())
                safe_predictions_end.extend(states_safe_end.tolist())
                collision_predictions_start.extend(states_coll_start.tolist())
                collision_predictions_end.extend(states_coll_end.tolist())

                # calculate perfect velocity predictions
                if use_planner:
                    # -- get the future actions
                    if self.args_cli.equal_actions:
                        plan = self.agent._plan
                        ph = self.cfg.model_cfg.prediction_horizon
                        H = plan.shape[1]

                        base_env = torch.zeros(env_new_prediction.shape[0], dtype=torch.long, device=plan.device)
                        ps = self.agent._plan_step[base_env] - 1
                        idx_seq = torch.stack([ps + k for k in range(ph)], dim=1).clamp(0, H - 1)
                        print("plan_step min/max:", ps.min().item(), ps.max().item(), "H:", H, "ph:", ph,
                        "idx_min:", (ps).min().item(), "idx_max:", (ps + ph - 1).max().item(), flush=True)

                        future_actions = plan[base_env][:, idx_seq].to(self.device)

                    elif hasattr(self.args_cli, "max_actions") and self.args_cli.max_actions:
                        future_actions = torch.concatenate(
                            [
                                max_actions[env_new_prediction][:, None, :]
                                for _ in range(self.cfg.model_cfg.prediction_horizon)
                            ],
                            dim=1,
                        ).to(self.device)
                    else:
                        # plan shape: (num_envs, plan_horizon, act_dim)
                        plan = self.agent._plan
                        ps = self.agent._plan_step[env_new_prediction] - 1  # (n,)
                        H = plan.shape[1]
                        ph = self.cfg.model_cfg.prediction_horizon

                        # (n, ph) 每个 env 要取的 action index
                        idx_seq = torch.stack([ps + k for k in range(ph)], dim=1)

                        # 防止越界（关键！）
                        idx_seq = idx_seq.clamp(0, H - 1)

                        # 高级索引：先选 env，再按每行 idx_seq 取
                        future_actions = plan[env_new_prediction][:, idx_seq]  # (n, ph, act_dim)
                        future_actions = future_actions.to(self.device)


                    # -- collect the start and goal observations
                    planner.obs = obs["planner_obs"]
                    # -- get the perfect velocity estimate
                    new_perfect_vel_pred = planner.b_obj_func_N_step(
                        future_actions.unsqueeze(0),
                        only_rollout=True,
                        control_mode="velocity_control",
                        env_ids=env_new_prediction,
                    ).squeeze(1)
                    # -- for visualization, append the starting state
                    new_perfect_vel_pred = torch.cat(
                        [obs["planner_obs"]["start"][env_new_prediction].unsqueeze(1), new_perfect_vel_pred], dim=1
                    )
                    # -- separate into start and end points and flatten
                    new_perfect_vel_pred_start = new_perfect_vel_pred[..., :-1, :3].reshape(-1, 3).cpu()
                    new_perfect_vel_pred_end = new_perfect_vel_pred[..., 1:, :3].reshape(-1, 3).cpu()
                    # -- elevate the prediction to the terrain height
                    new_perfect_vel_pred_start[:, 2] = (
                        planner.terrain_analysis.get_height(new_perfect_vel_pred_start[:, :2]) + 0.5
                    )
                    new_perfect_vel_pred_end[:, 2] = (
                        planner.terrain_analysis.get_height(new_perfect_vel_pred_end[:, :2]) + 0.5
                    )
                    # -- save predictions
                    perfect_velocity_predictions_start.extend(new_perfect_vel_pred_start.tolist())
                    perfect_velocity_predictions_end.extend(new_perfect_vel_pred_end.tolist())
            # Drawing
            ###
            if hasattr(self, "draw_interface") and self.draw_interface is not None:
                # draw predictions
                self.draw_interface.clear_lines()
                for env_idx in range(self.env.num_envs):
                    if self.replay_buffer.fill_idx[env_idx] > 2:
                        # plot trajectories from replay buffer
                        self.draw_interface.draw_lines(
                            self.replay_buffer.states[env_idx, 1 : self.replay_buffer.fill_idx[env_idx], 0, :3][:-1].tolist(),
                            self.replay_buffer.states[env_idx, 1 : self.replay_buffer.fill_idx[env_idx], 0, :3][1:].tolist(),
                            self.trajectory_color * (self.replay_buffer.fill_idx[env_idx] - 2),
                            [25.0] * (self.replay_buffer.fill_idx[env_idx] - 2),
                        )

                self.draw_interface.draw_lines(
                    safe_predictions_start,
                    safe_predictions_end,
                    [self.safe_colors[0]] * len(safe_predictions_start),
                    [25.0] * len(safe_predictions_start),
                )
                self.draw_interface.draw_lines(
                    collision_predictions_start,
                    collision_predictions_end,
                    [self.collision_colors[0]] * len(collision_predictions_start),
                    [25.0] * len(collision_predictions_start),
                )

                # draw perfect velocity predictions
                if use_planner:
                    self.draw_interface.draw_lines(
                        perfect_velocity_predictions_start,
                        perfect_velocity_predictions_end,
                        self.perfect_velocity_color * len(perfect_velocity_predictions_start),
                        [25.0] * len(perfect_velocity_predictions_start),
                    )
            ###
            # Drawing
            ###
            if hasattr(self, "draw_interface") and self.draw_interface is not None:
                # draw predictions
                self.draw_interface.clear_lines()
                for env_idx in range(self.env.num_envs):
                    if self.replay_buffer.fill_idx[env_idx] > 2:
                        # plot trajectories from replay buffer
                        self.draw_interface.draw_lines(
                            self.replay_buffer.states[env_idx, 1 : self.replay_buffer.fill_idx[env_idx], 0, :3][:-1].tolist(),
                            self.replay_buffer.states[env_idx, 1 : self.replay_buffer.fill_idx[env_idx], 0, :3][1:].tolist(),
                            self.trajectory_color * (self.replay_buffer.fill_idx[env_idx] - 2),
                            [25.0] * (self.replay_buffer.fill_idx[env_idx] - 2),
                        )

                self.draw_interface.draw_lines(
                    safe_predictions_start,
                    safe_predictions_end,
                    [self.safe_colors[0]] * len(safe_predictions_start),
                    [25.0] * len(safe_predictions_start),
                )
                self.draw_interface.draw_lines(
                    collision_predictions_start,
                    collision_predictions_end,
                    [self.collision_colors[0]] * len(collision_predictions_start),
                    [25.0] * len(collision_predictions_start),
                )

                # draw perfect velocity predictions
                if use_planner:
                    self.draw_interface.draw_lines(
                        perfect_velocity_predictions_start,
                        perfect_velocity_predictions_end,
                        self.perfect_velocity_color * len(perfect_velocity_predictions_start),
                        [25.0] * len(perfect_velocity_predictions_start),
                    )

            # save the images if cameras are provided
            if cameras is not None:
                for idx, camera in enumerate(cameras):
                    
                    if hasattr(self.args_cli, "paper_platform_figure") and self.args_cli.paper_platform_figure:
                        if idx == 1:
                            robot_pos_1 = self.env.scene.articulations["robot"].data.root_pos_w[idx] + torch.tensor(
                                [-11.0, -1.0, 10], device=self.env.device
                            )
                            camera.set_world_pose(position=robot_pos_1, orientation=[0.9250441, 0.0, 0.3798598, 0.0])
                        elif idx == 2:
                            robot_pos_2 = self.env.scene.articulations["robot"].data.root_pos_w[idx] + torch.tensor(
                                [-10.0, 0.0, 9], device=self.env.device
                            )
                            camera.set_world_pose(position=robot_pos_2, orientation=[0.9250441, 0.0, 0.3798598, 0.0])
                    robot_pos = self.env.scene.articulations["robot"].data.root_pos_w[0]

                    camera.set_world_pose(
                        position=(
                            robot_pos
                            + torch.tensor([-5.0, 0.0, 5.0], device=self.env.device)
                        ).tolist(),
                        orientation=[0.9250441, 0.0, 0.3798598, 0.0],
                    )

                    for i in range(2):
                        self.env.sim.render()
                    camera.get_current_frame()
                    # Convert RGB to BGR for OpenCV
                    image_bgr = cv2.cvtColor(camera.get_rgba()[:, :, :3], cv2.COLOR_RGB2BGR)
                    # Save the image as PNG
                    assert cv2.imwrite(
                        f"{cam_save_path[idx]}/img_{str(render_counter[idx].item()).zfill(4)}.png", image_bgr
                    )

                    render_counter[idx] += 1

                # break if all environments are done
                if torch.any(render_counter >= 500):
                    break

        # save the images as a video
        if cameras is not None:
            print(f"[INFO]: Images saved to {cam_save_path}. Generating video.")
            for idx, path in enumerate(cam_save_path):
                os.system(
                    f"ffmpeg -r {int(1 / self.env.step_dt)} -f image2 -s 1920x1080 -i"
                    f" '{path}/img_%04d.png' -vcodec libx264 -profile:v high -crf 25 -pix_fmt yuv420p"
                    f" '{path}/video.mp4'"
                )

    def close(self):
        self.env.close()
        del self.model
        del self.replay_buffer
        del self.trainer
        if self.agent is not MISSING and self.agent is not None:
            del self.agent

    """
    Helper functions
    """

    @torch.inference_mode()
    def _collect(self, eval: bool = False):
        """Collect data from the environment and store it in the trainer's storage."""
        print("[INFO]: Collecting data...")
        if self.cfg.env_cfg.curriculum is not None:
            # store env reset counter value from previous rounds
            prev_env_reset_counter = self.env.curriculum_manager._term_cfgs[0].func.env_reset_counter
        # reset environment
        with torch.inference_mode():
            obs, _ = self.env.reset(random.randint(0, 1000000))
            try:
                robot = self.env.scene.articulations["robot"]
                p = robot.data.root_pos_w[:5, 2].detach().cpu().numpy()
                origins = self.env.scene.env_origins[:5].cpu().numpy()
                print(f"[RESET-DBG] random post-reset root_z[:5]={p} env_origins[:5]={origins}", flush=True)
                try:
                    if getattr(self, "force_spawn_z", None) is not None:
                        desired_z = float(self.force_spawn_z)
                        world_pos = robot.data.root_pos_w.clone()
                        origins = self.env.scene.env_origins.to(world_pos.device)
                        world_pos[:, 2] = origins[:, 2] + desired_z
                        orientations = robot.data.root_quat_w.clone()
                        env_ids = torch.arange(world_pos.shape[0], device=world_pos.device)
                        robot.write_root_pose_to_sim(torch.cat([world_pos, orientations], dim=-1), env_ids=env_ids)
                        try:
                            zero_vel = torch.zeros((world_pos.shape[0], 6), device=world_pos.device)
                            robot.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
                        except Exception:
                            pass
                        print(f"[FORCE-DBG] forced world_z_offset={desired_z} wrote world_z[:5]={world_pos[:5,2].cpu().numpy()}", flush=True)
                except Exception as e:
                    print(f"[FORCE-DBG] failed to force spawn z: {e}", flush=True)
            except Exception:
                pass
            # the contact sensor is delayed, execute delay+1 steps to reset all environments correctly
            if torch.any(obs["fdm_state"][..., 7]):
                for _ in range(self.cfg.env_cfg.scene.contact_forces.history_length - self.cfg.env_cfg.decimation + 1):
                    obs, _, dones, _, _ = self.env.step(torch.zeros(self.env.num_envs, 3, device=self.env.device))
                if dones.sum() != 0:
                    carb.log_warn("Environments should not be done after reset.")
            # if curriuclum activated, update counter got increased by number of all environments due to reset
            # balance the counter which will set the updated flag to False
            if self.cfg.env_cfg.curriculum is not None:
                self.env.curriculum_manager._term_cfgs[0].func.env_reset_counter = prev_env_reset_counter
                if self.env.curriculum_manager._term_cfgs[0].func.cfg.update_interval < self.env.num_envs:
                    print(
                        "[WARNING]: Update interval is smaller than number of environments. Ratio are already modified"
                        " even if not intended."
                    )
            # if planner is used, set resample_population to True
            if "planner_obs" in obs:
                obs["planner_obs"]["resample_population"] = torch.ones(
                    self.env.num_envs, dtype=torch.bool, device=self.env.device
                )

        # reset replay buffer
        self.replay_buffer.reset()
        # reset agent
        actions = self.agent.reset(obs)

        # reset feet contact amd counter
        self.feet_contact[:] = False
        self.feet_non_contact_counter[:] = 0

        # collect data
        sim_time = 0.0
        process_time = 0.0
        plan_time = 0.0
        collect_time = []
        info_counter = 1
        step_counter = 0

        while not self.replay_buffer.is_filled:
            ###
            # Step the environment
            ###
            sim_start = time.time()
            with torch.inference_mode():
                obs, _, dones, _, _ = self.env.step(actions.clone())
            sim_time += time.time() - sim_start

            ###
            # Determine feet contact
            ###
            # Note: only start recording and changing actions when all feet have touched the ground
            feet_all_contact, dones, obs_new = self._feet_contact_handler(dones)
            obs = obs_new if obs_new is not None else obs

            ###
            # Plan the actions for the current state
            ###
            # Note: has to be done before updating the replay buffer, as these actions are given to the current state
            # and FDM actions are the current state and the executed actions for that state
            plan_start = time.time()

            # apply curriculum to action generation if enabled (only during training data collection)
            if (
                not eval
                and self.cfg.env_cfg.curriculum is not None
                and self.env.curriculum_manager._term_cfgs[0].func.updated
            ):
                with torch.inference_mode():
                    assert isinstance(self.agent, MixedAgent), "Curriculum can only be applied to MixedAgent."
                    self.agent.update_ratios(self.env.curriculum_manager._term_cfgs[0].func.ratios)

            # get actions
            actions = self.agent.act(obs, dones.to(torch.bool).clone(), feet_contact=feet_all_contact)

            # debug viz
            self.agent.debug_viz()

            plan_time += time.time() - plan_start

            ###
            # Update replay buffer
            ###
            update_buffer_start = time.time()
            self.replay_buffer.add(
                states=obs["fdm_state"].clone(),
                obersevations_proprioceptive=obs["fdm_obs_proprioception"].clone(),
                obersevations_exteroceptive=(
                    obs["fdm_obs_exteroceptive"].clone() if "fdm_obs_exteroceptive" in obs else None
                ),
                actions=actions.clone(),
                dones=dones.to(torch.bool).clone(),
                feet_contact=feet_all_contact,
                add_observation_exteroceptive=(
                    obs["fdm_add_obs_exteroceptive"] if "fdm_add_obs_exteroceptive" in obs else None
                ),
            )
            process_time += time.time() - update_buffer_start

            ###
            # Update timers
            ###

            # print fill ratio information
            if self.replay_buffer.fill_ratio > 0.1 * info_counter:
                print(
                    f"[INFO] Fill ratio: {self.replay_buffer.fill_ratio:.2f} \tPlan time: \t{plan_time:.2f}s \tSim"
                    f" time: \t{sim_time:.2f}s \tUpdate time: \t{process_time:.2f}s"
                )
                # save overall time
                collect_time.append(plan_time + sim_time + process_time)
                # reset times
                plan_time = 0.0
                sim_time = 0.0
                process_time = 0.0
                info_counter += 1

            step_counter += 1
            if step_counter % 1000 == 0:
                print(f"[INFO] Step {step_counter} completed.")

            ###
            # Break if some environments take too long to be filled
            ###

            if (
                not self.replay_buffer.is_filled
                and self.replay_buffer.fill_ratio > 0.95
                and plan_time + sim_time + process_time > 1.5 * np.mean(collect_time)
            ):
                print("[WARNING]: Collection took too long for some environments. Stopping collection.")
                self.replay_buffer.fill_leftover_envs()
                break

        # slice into samples and populate storage of trainer
        if eval:
            _, max_vel, max_acc = self.trainer.val_dataset.populate(replay_buffer=self.replay_buffer)

            # # debug, visualize the initial observations
            # import omni.isaac.debug_draw._debug_draw as omni_debug_draw
            # draw_interface = omni_debug_draw.acquire_debug_draw_interface()
            # draw_interface.draw_points(
            #     initial_pos[:, 0, :3].tolist(),
            #     [(1.0, 0.5, 0, 1)] * initial_pos.shape[0],
            #     [5] * initial_pos.shape[0],
            # )
            # # render simulation
            # for i in range(100):
            #     self.env.render()

            self.model.set_velocity_limits(max_vel)
            if hasattr(self.model, "set_acceleration_limits"):
                self.model.set_acceleration_limits(max_acc)
            if hasattr(self.model, "set_hard_contact_obs_limits"):
                self.model.set_hard_contact_obs_limits(
                    min_hard_contact_obs=self.trainer.val_dataset.min_hard_contact_obs.to(self.device),
                    max_hard_contact_obs=self.trainer.val_dataset.max_hard_contact_obs.to(self.device),
                )

            # run evaluation on all new collected samples
            self.trainer.evaluate()

            # reset the curriculum term to not update the ratios
            if self.cfg.env_cfg.curriculum is not None:
                with torch.inference_mode():
                    self.env.curriculum_manager._term_cfgs[0].func.reset(self.env)

        else:
            _, max_vel, max_acc = self.trainer.train_dataset.populate(replay_buffer=self.replay_buffer)
            self.model.set_velocity_limits(max_vel)
            if hasattr(self.model, "set_acceleration_limits"):
                self.model.set_acceleration_limits(max_acc)
            if hasattr(self.model, "set_hard_contact_obs_limits"):
                self.model.set_hard_contact_obs_limits(
                    min_hard_contact_obs=self.trainer.train_dataset.min_hard_contact_obs.to(self.device),
                    max_hard_contact_obs=self.trainer.train_dataset.max_hard_contact_obs.to(self.device),
                )
            # run evaluation on all new collected samples
            self.trainer.evaluate(dataloader=self.trainer.dataloader)

        # save depth images as debug info
        if self.args_cli.debug and self.args_cli.env == "depth" and not eval:
            os.makedirs(self.trainer.log_dir + "/debug", exist_ok=True)
            for idx in range(min(self.trainer.train_dataset.num_samples, 100)):
                depth_img = self.trainer.train_dataset.obs_exteroceptive[idx, :, :, 0].cpu().numpy() * 1000
                depth_img = depth_img.astype(np.uint16)
                cv2.imwrite(self.trainer.log_dir + f"/debug/depth_{idx}.png", depth_img)

        print("[INFO]: Data collection complete.")

    @torch.inference_mode()
        
    def _feet_contact_handler(
        self, dones: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | dict[str, torch.Tensor]] | None]:
        # 1) done 的 env 先清零计数（保持语义）
        self.feet_non_contact_counter[dones] = 0

        # 2) per-foot 接触判断
        forces = torch.norm(
            self.env.scene.sensors["contact_forces"].data.net_forces_w[:, self.feet_idx],
            dim=-1,
        )  # (N, nfeet)

        # 阈值先别太大：两足脚底接触力可能小于你想象，建议先用 1~5
        contact_th = 5.0
        self.feet_contact = forces > contact_th  # (N, nfeet)

        feet_any_contact = torch.any(self.feet_contact, dim=-1)  # (N,)
        feet_all_contact = torch.all(self.feet_contact, dim=-1)  # (N,)

        # 3) 可选：只在“起步阶段”要求 all_contact 达成一次
        if not hasattr(self, "feet_contact_initialized"):
            self.feet_contact_initialized = torch.zeros(
                self.env.num_envs, dtype=torch.bool, device=self.env.device
            )

        # dones 的 env 重新初始化
        self.feet_contact_initialized[dones] = False

        # 一旦某 env 达到过 all_contact，则认为起步完成
        self.feet_contact_initialized |= feet_all_contact

        # 起步前：要求双脚都触地；起步后：允许单脚支撑
        feet_ok = torch.where(
            self.feet_contact_initialized,
            feet_any_contact,
            feet_all_contact,
        )

        # 4) 计数器：统计“完全没脚接触”的持续时间（用 any_contact 做更合理）
        self.feet_non_contact_counter[feet_any_contact] = 0
        self.feet_non_contact_counter[~feet_any_contact] += 1

        # 5) 长时间没接触则 reset
        obs = None
        reset_envs = self.feet_non_contact_counter > 200
        if torch.any(reset_envs):
            print("[WARNING]: Resetting environments that have not touched the ground for a while.")
            self.env._reset_idx(self.agent._ALL_INDICES[reset_envs])
            obs = self.env.observation_manager.compute()
            dones[reset_envs] = True
            self.feet_non_contact_counter[reset_envs] = 0
            self.feet_contact_initialized[reset_envs] = False

        # 6) debug 打印（别引用未定义变量）
        if torch.rand(1).item() < 0.01:
            e = 0
            print(
                "forces:", forces[e].tolist(),
                "contact:", self.feet_contact[e].tolist(),
                "all:", feet_all_contact[e].item(),
                "any:", feet_any_contact[e].item(),
                "init:", self.feet_contact_initialized[e].item(),
                "feet_ok:", feet_ok[e].item(),
                flush=True,
            )

        return feet_ok, dones, obs


    def _eval_predict(self, env_ids: torch.Tensor, model: FDMModel | None = None):
        """Make predictions based on the current states and the planned actions"""

        # get initial states
        initial_states = self.replay_buffer.states[env_ids, self.replay_buffer.fill_idx[env_ids] - 1, 0][:, None, :7]
        initial_states_SE3 = pp.SE3(
            initial_states.repeat(1, self.cfg.model_cfg.prediction_horizon, 1).reshape(-1, 7)
        ).to(self.device)

        # get state history transformed into local frame
        state_history = self.trainer.train_dataset.state_history_transformer(
            self.replay_buffer,
            torch.vstack([env_ids, self.replay_buffer.fill_idx[env_ids] - 1]).T,
            initial_states,
            self.model.cfg.history_length,
            self.model.cfg.exclude_state_idx_from_input,
        ).to(self.device)
        if self.args_cli.env != "baseline":
            if model is None:
                state_history[..., 5] = (state_history[..., 5] - self.model.hard_contact_obs_limits[0]) / (
                    self.model.hard_contact_obs_limits[1] - self.model.hard_contact_obs_limits[0]
                )
            else:
                state_history[..., 5] = (state_history[..., 5] - model.hard_contact_obs_limits[0]) / (
                    model.hard_contact_obs_limits[1] - model.hard_contact_obs_limits[0]
                )

               # collect future actions  (SAFE indexing)
        plan = self.agent._plan                       # (num_envs, H_plan, act_dim)
        plan_steps = self.agent._plan_step            # (num_envs,)
        H_plan = plan.shape[1]
        ph = int(self.cfg.model_cfg.prediction_horizon)

        if self.args_cli.equal_actions:
            # use env 0's plan for all envs, but still safe w.r.t. plan_step
            base_env = torch.zeros(env_ids.shape[0], dtype=torch.long, device=plan.device)
            ps = plan_steps[base_env] - 1  # (n,)
            idx_seq = torch.stack([ps + k for k in range(ph)], dim=1)  # (n, ph)
            idx_seq = idx_seq.clamp(0, H_plan - 1)

            # first pick base_env rows, then gather per-row indices
            future_actions = plan[base_env]  # (n, H_plan, act_dim)
            future_actions = future_actions.gather(
                dim=1,
                index=idx_seq[..., None].expand(-1, -1, future_actions.shape[-1]),
            )

        elif hasattr(self.args_cli, "max_actions") and self.args_cli.max_actions:
            future_actions = torch.zeros(env_ids.shape[0], ph, 3, device=self.device)
            future_actions[:, :, 0] = VEL_RANGE_X[1]
            future_actions[:, :, 1] = VEL_RANGE_Y[1]

        else:
            ps = plan_steps[env_ids] - 1  # (n,)
            idx_seq = torch.stack([ps + k for k in range(ph)], dim=1)  # (n, ph)
            idx_seq = idx_seq.clamp(0, H_plan - 1)

            # pick env rows then gather
            plan_sel = plan[env_ids]  # (n, H_plan, act_dim)
            future_actions = plan_sel.gather(
                dim=1,
                index=idx_seq[..., None].expand(-1, -1, plan_sel.shape[-1]),
            )

        future_actions = future_actions.to(self.device)


        # make predictions
        model_in = (
            state_history,
            self.replay_buffer.observations_proprioceptive[env_ids, self.replay_buffer.fill_idx[env_ids] - 1, :].to(
                self.device
            ),
            (
                self.replay_buffer.observations_exteroceptive[env_ids, self.replay_buffer.fill_idx[env_ids] - 1, :]
                .type(torch.float32)
                .to(self.device)
                if self.replay_buffer.observations_exteroceptive is not None
                else torch.zeros(1)
            ),
            future_actions,
            (
                self.replay_buffer.add_observations_exteroceptive[env_ids, self.replay_buffer.fill_idx[env_ids] - 1, :]
                .type(torch.float32)
                .to(self.device)
                if self.replay_buffer.add_observations_exteroceptive is not None
                else torch.zeros(1)
            ),
        )
        if model:
            with torch.no_grad():
                model_out = list(model.forward(model_in))
        else:
            model_out = list(self.trainer.predict(model_in))

        # IMPORTANT: assume that the first output of the model is the future states
        future_states = model_out[0]
        if self.args_cli.env == "baseline":
            # attach a zero yaw angle to the future states
            future_states = torch.cat(
                [
                    future_states,
                    torch.sin(torch.zeros_like(future_states[..., 0])).unsqueeze(2),
                    torch.cos(torch.zeros_like(future_states[..., 0])).unsqueeze(2),
                ],
                dim=-1,
            )

        # transform future states in global frame
        future_states = torch.concatenate(
            [
                future_states[..., :2],
                torch.zeros_like(future_states[..., 2]).unsqueeze(2),
                math_utils.convert_quat(
                    math_utils.quat_from_euler_xyz(
                        roll=torch.zeros_like(future_states[..., 2]),
                        pitch=torch.zeros_like(future_states[..., 2]),
                        yaw=torch.atan2(future_states[..., 2], future_states[..., 3]),
                    ),
                    to="xyzw",
                ),
            ],
            dim=-1,
        )
        future_states = (initial_states_SE3 * pp.SE3(future_states.reshape(-1, 7))).tensor()
        future_states_yaw = math_utils.euler_xyz_from_quat(future_states[..., [6, 3, 4, 5]])[2]
        future_states = torch.concatenate([future_states[..., :3], future_states_yaw.unsqueeze(1)], dim=-1).reshape(
            -1, self.cfg.model_cfg.prediction_horizon, 4
        )

        # overwrite future states in base frame with the ones in the global frame
        model_out[0] = future_states

        return model_out

    def _eval_loss_fdm(
        self,
        env_new_prediction: torch.Tensor,
        model_out: tuple[torch.Tensor, torch.Tensor],
        pred_trajectories: dict[int, list[torch.Tensor]],
        pred_collision: dict[int, list[torch.Tensor]],
        final_pred_error: dict[int, list[torch.Tensor]],
        meta_eval: dict[str, list[float]],
        counter: int,
    ) -> tuple[dict[int, list[torch.Tensor]], dict[int, list[torch.Tensor]], dict[str, list[float]], int]:
        # extract the output quantities of the model
        if self.args_cli.env == "baseline":
            future_states, collision_pred = model_out
        else:
            future_states, collision_pred, energy_pred = model_out
        # loss for world frame coordinates
        for env_idx in env_new_prediction:
            # get previous predictions
            prev_pred = pred_trajectories[env_idx.item()]
            prev_coll = pred_collision[env_idx.item()]
            if len(prev_pred) <= self.cfg.model_cfg.prediction_horizon:
                continue
            # NOTE: the predicitions also include the initial position for the transformation, has to be removed here
            prev_pred = prev_pred[-self.cfg.model_cfg.prediction_horizon][-self.cfg.model_cfg.prediction_horizon :]
            if self.cfg.model_cfg.unified_failure_prediction:
                prev_coll = prev_coll[-self.cfg.model_cfg.prediction_horizon]
            else:
                prev_coll = prev_coll[-self.cfg.model_cfg.prediction_horizon][-self.cfg.model_cfg.prediction_horizon :]

            # get loss values
            future_states_yaw = math_utils.euler_xyz_from_quat(
                self.replay_buffer.states[
                    env_idx,
                    self.replay_buffer.fill_idx[env_idx]
                    - self.cfg.model_cfg.prediction_horizon : self.replay_buffer.fill_idx[env_idx],
                    0,
                    [6, 3, 4, 5],
                ].reshape(-1, 4)
            )[2].reshape(-1, 1)
            # encode yaw with sin and cos
            future_states_yaw = torch.cat((torch.sin(future_states_yaw), torch.cos(future_states_yaw)), dim=1)
            target = (
                torch.hstack((
                    self.replay_buffer.states[
                        env_idx,
                        self.replay_buffer.fill_idx[env_idx]
                        - self.cfg.model_cfg.prediction_horizon : self.replay_buffer.fill_idx[env_idx],
                        0,
                        :2,
                    ],
                    future_states_yaw,
                    self.replay_buffer.states[
                        env_idx,
                        self.replay_buffer.fill_idx[env_idx]
                        - self.cfg.model_cfg.prediction_horizon : self.replay_buffer.fill_idx[env_idx],
                        0,
                        -1,
                    ].unsqueeze(1),
                    # FIXME: quick fix for energy trajectory
                    torch.zeros((self.cfg.model_cfg.prediction_horizon, 1)),
                ))
                .unsqueeze(0)
                .to(self.device)
            )

            # compute loss
            if self.args_cli.env == "baseline":
                model_out = [prev_pred.unsqueeze(0)[..., :2], prev_coll.unsqueeze(0)]
            else:
                # FIXME: quick fix for energy trajectory
                model_out = [prev_pred.unsqueeze(0), prev_coll.unsqueeze(0), torch.zeros_like(prev_coll).unsqueeze(0)]
            _, meta = self.model.loss(model_out, target, mode="eval")
            if len(meta_eval) == 0:
                for key, value in meta.items():
                    meta_eval[key] = [value]
            else:
                for key, value in meta.items():
                    meta_eval[key].append(value)

            # save final prediction error
            final_pred_error[env_idx.item()].append(
                torch.vstack((prev_pred[-1, :3], torch.cat((target[0, -1, :2], prev_pred[-1, 2].unsqueeze(0)))))
            )

            # increase counter
            counter += target.shape[0]

        # append initial position to future states
        future_states = torch.cat(
            (
                self.replay_buffer.states[env_new_prediction, self.replay_buffer.fill_idx[env_new_prediction] - 1, 0][
                    :, None, [0, 1, 2, 7]
                ].to(self.device),
                future_states,
            ),
            dim=1,
        )
        [
            pred_trajectories[env_id.item()].append(future_states[pred_idx].clone())
            for pred_idx, env_id in enumerate(env_new_prediction)
        ]
        [
            pred_collision[env_id.item()].append(collision_pred[pred_idx].clone())
            for pred_idx, env_id in enumerate(env_new_prediction)
        ]
        return pred_trajectories, pred_collision, final_pred_error, meta_eval, counter

    def _eval_loss_proprioception(
        self,
        env_new_prediction: torch.Tensor,
        model_out: tuple[torch.Tensor, torch.Tensor],
        pred_trajectories: dict[int, list[torch.Tensor]],
        pred_friction: dict[int, list[torch.Tensor]],
        final_pred_error: dict[int, list[torch.Tensor]],
        meta_eval: dict[str, list[float]],
        counter: int,
    ) -> dict[int, list[torch.Tensor]]:
        # extract the output quantities of the model
        future_states, friction = model_out
        # loss for world frame coordinates
        for env_idx in env_new_prediction:
            # get previous predictions
            prev_pred = pred_trajectories[env_idx.item()]
            prev_fric = pred_friction[env_idx.item()]
            if len(prev_pred) <= self.cfg.model_cfg.prediction_horizon:
                continue
            # NOTE: the predicitions also include the initial position for the transformation, has to be removed here
            prev_pred = prev_pred[-self.cfg.model_cfg.prediction_horizon][-self.cfg.model_cfg.prediction_horizon :]
            prev_fric = prev_fric[-self.cfg.model_cfg.prediction_horizon]

            # get loss values
            model_out = [prev_pred.unsqueeze(0), prev_fric.unsqueeze(0)]
            future_states_yaw = math_utils.euler_xyz_from_quat(
                self.replay_buffer.states[
                    env_idx,
                    self.replay_buffer.fill_idx[env_idx]
                    - self.cfg.model_cfg.prediction_horizon : self.replay_buffer.fill_idx[env_idx],
                    0,
                    [6, 3, 4, 5],
                ].reshape(-1, 4)
            )[2].reshape(-1, 1)
            # encode yaw with sin and cos
            future_states_yaw = torch.cat((torch.sin(future_states_yaw), torch.cos(future_states_yaw)), dim=1)
            target = (
                torch.hstack((
                    # POSITION
                    self.replay_buffer.states[
                        env_idx,
                        self.replay_buffer.fill_idx[env_idx]
                        - self.cfg.model_cfg.prediction_horizon : self.replay_buffer.fill_idx[env_idx],
                        0,
                        :2,
                    ],
                    # ORIENTATION
                    future_states_yaw,
                    # COLLISION
                    self.replay_buffer.states[
                        env_idx,
                        self.replay_buffer.fill_idx[env_idx]
                        - self.cfg.model_cfg.prediction_horizon : self.replay_buffer.fill_idx[env_idx],
                        0,
                        4,  # TODO: check that is this is collision
                    ].unsqueeze(1),
                    # FRICTION
                    self.replay_buffer.states[
                        env_idx,
                        self.replay_buffer.fill_idx[env_idx]
                        - self.cfg.model_cfg.prediction_horizon : self.replay_buffer.fill_idx[env_idx],
                        0,
                        -4,  # TODO: check that is this is friction
                    ].unsqueeze(1),
                ))
                .unsqueeze(0)
                .to(self.device)
            )

            # compute loss
            _, meta = self.model.loss(model_out, target, mode="eval")
            if len(meta_eval) == 0:
                for key, value in meta.items():
                    meta_eval[key] = [value]
            else:
                for key, value in meta.items():
                    meta_eval[key].append(value)

            # save final prediction error
            final_pred_error[env_idx.item()].append(
                torch.vstack((prev_pred[-1, :3], torch.cat((target[0, -1, :2], prev_pred[-1, 2].unsqueeze(0)))))
            )

            # increase counter
            counter += target.shape[0]

        # append initial position to future states
        future_states = torch.cat(
            (
                self.replay_buffer.states[env_new_prediction, self.replay_buffer.fill_idx[env_new_prediction] - 1, 0][
                    :, None, [0, 1, 2, 7]
                ].to(self.device),
                future_states,
            ),
            dim=1,
        )
        [
            pred_trajectories[env_id.item()].append(future_states[pred_idx].clone())
            for pred_idx, env_id in enumerate(env_new_prediction)
        ]
        [
            pred_friction[env_id.item()].append(friction[pred_idx].clone())
            for pred_idx, env_id in enumerate(env_new_prediction)
        ]
        return pred_trajectories, pred_friction, final_pred_error, meta_eval, counter

    def _draw_trajectories(
        self,
        pred_trajectories: dict[int, list[torch.Tensor]],
        pred_collision: dict[int, list[torch.Tensor]],
        final_pred_error: dict[int, list[torch.Tensor]],
        safe_colors: list[tuple[float]],
        collision_colors: list[tuple[float]],
        draw_trajectory: bool = True,
    ):
        if not hasattr(self, "draw_interface"):
            raise RuntimeError("Draw interface not initialized. Set `eval=True` in the runner constructor.")

        for env_idx in range(self.env.num_envs):
            if draw_trajectory:
                start_idx = (
                    0 if self.replay_buffer.fill_idx[env_idx] < 10 else self.replay_buffer.fill_idx[env_idx] - 10
                )
                # plot trajectories from replay buffer
                self.draw_interface.draw_lines(
                    self.replay_buffer.states[
                        env_idx, start_idx : self.replay_buffer.fill_idx[env_idx], 0, :3
                    ].tolist()[:-1],
                    self.replay_buffer.states[
                        env_idx, start_idx : self.replay_buffer.fill_idx[env_idx], 0, :3
                    ].tolist()[1:],
                    self.trajectory_color * (self.replay_buffer.fill_idx[env_idx] - 1 - start_idx),
                    [5.0] * (self.replay_buffer.fill_idx[env_idx] - 1 - start_idx),
                )

            # plot every prediction made for the environment
            if len(pred_trajectories[env_idx]) > self.nb_draw_traj * self.step_draw_traj:
                pred_iter = pred_trajectories[env_idx][-self.nb_draw_traj * self.step_draw_traj :]
                pred_iter = pred_iter[::2] if len(pred_trajectories[env_idx]) % 2 == 0 else pred_iter[1::2]
                if pred_collision is not None:
                    coll_iter = pred_collision[env_idx][-self.nb_draw_traj * self.step_draw_traj :]
                    coll_iter = coll_iter[::2] if len(pred_trajectories[env_idx]) % 2 == 0 else coll_iter[1::2]
                error_iter = final_pred_error[env_idx][-self.nb_draw_traj * (self.step_draw_traj - 1) :]
                error_iter = error_iter[::2] if len(pred_trajectories[env_idx]) % 2 == 0 else error_iter[1::2]
            else:
                pred_iter = pred_trajectories[env_idx][::2]
                if pred_collision is not None:
                    coll_iter = pred_collision[env_idx][::2]
                error_iter = final_pred_error[env_idx][::2]

            for pred_idx, pred in enumerate(pred_iter):
                # in collision color is red otherwise green
                if pred_collision is not None:
                    color = collision_colors if torch.any(coll_iter[pred_idx] > 0.5) else safe_colors
                else:
                    color = safe_colors
                # draw line
                self.draw_interface.draw_lines(
                    pred[:-1, :3].tolist(),
                    pred[1:, :3].tolist(),
                    color,
                    [5.0] * (pred.shape[0] - 1),
                )

            if len(error_iter) > 0:
                for curr_error in error_iter:
                    self.draw_interface.draw_lines(
                        [curr_error[0].tolist()],
                        [curr_error[1].tolist()],
                        [(1.0, 1.0, 0, 1.0)],
                        [5.0],
                    )
