# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import carb

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.envs.mdp import bad_orientation, root_height_below_minimum

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _delayed_terminal_mask(
    env: ManagerBasedRLEnv,
    raw_terminal: torch.Tensor,
    name: str,
    hold_steps: int,
) -> torch.Tensor:
    """Hold terminal envs alive for a few physics/control calls before returning done."""
    counter_name = f"_fdm_{name}_hold_counter"
    if not hasattr(env, counter_name):
        setattr(env, counter_name, torch.zeros(env.num_envs, device=env.device, dtype=torch.long))

    counter = getattr(env, counter_name)
    raw_terminal = raw_terminal.to(device=env.device, dtype=torch.bool)
    hold_steps = int(hold_steps)
    if hold_steps <= 0:
        counter[:] = 0
        return raw_terminal

    active = raw_terminal | (counter > 0)
    counter[active] += 1
    counter[~active] = 0

    done = counter > hold_steps
    counter[done] = 0
    return done


def delayed_root_height_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    hold_steps: int = 15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Delay root-height termination so FDM can record real post-failure states."""
    raw_terminal = root_height_below_minimum(
        env=env,
        minimum_height=minimum_height,
        asset_cfg=asset_cfg,
    )
    return _delayed_terminal_mask(env, raw_terminal, "root_height", hold_steps)


def delayed_bad_orientation(
    env: ManagerBasedRLEnv,
    limit_angle: float,
    hold_steps: int = 15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Delay bad-orientation termination so FDM can record real post-failure states."""
    raw_terminal = bad_orientation(
        env=env,
        limit_angle=limit_angle,
        asset_cfg=asset_cfg,
    )
    return _delayed_terminal_mask(env, raw_terminal, "bad_orientation", hold_steps)


def illegal_contact_delayed(
    env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg, delay: int = 1
) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold after a certain delay.

    This allows to record the observations when the robot is in contact and do not directly terminate.
    Useful in navigation tasks/ forward dynamics model learning.
    The delay is multiplied by the decimation of the low-level policy to make sure that the obstacles
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # make sure that delay is not larger than the history length
    delay_physics_timestep = delay * env.cfg.decimation
    if delay_physics_timestep >= net_contact_forces.shape[1]:
        carb.log_warn(
            f"Delay {delay} requires a history length of {delay_physics_timestep} but current length is only"
            f" {net_contact_forces.shape[1]}.Setting delay to {net_contact_forces.shape[1] - 1}"
        )
        delay_physics_timestep = net_contact_forces.shape[1] - 1
    # check if any contact force exceeds the threshold
    # newest contact force is at history idx 0, so delay is removing the newest contact force
    return torch.any(
        torch.max(torch.norm(net_contact_forces[:, delay_physics_timestep:, sensor_cfg.body_ids], dim=-1), dim=1)[0]
        > threshold,
        dim=1,
    )
