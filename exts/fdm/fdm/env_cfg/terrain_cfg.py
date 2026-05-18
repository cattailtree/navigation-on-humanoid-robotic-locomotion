# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.terrains as terrain_gen

import nav_tasks.terrains as fdm_terrain_gen


def _flat_reference(proportion: float = 0.10):
    return terrain_gen.HfRandomUniformTerrainCfg(
        proportion=proportion,
        noise_range=(0.0, 0.004),
        noise_step=0.004,
        border_width=0.25,
        vertical_scale=0.001,
    )


def _rough_floor(proportion: float = 0.10, noise_range: tuple[float, float] = (-0.035, 0.045)):
    return terrain_gen.HfRandomUniformTerrainCfg(
        proportion=proportion,
        noise_range=noise_range,
        noise_step=0.012,
        border_width=0.25,
        vertical_scale=0.004,
    )


def _offset_pillar_field(
    proportion: float,
    box_count: tuple[int, int],
    cylinder_count: tuple[int, int],
    rough: bool = True,
):
    return fdm_terrain_gen.MeshPillarTerrainCfg(
        proportion=proportion,
        platform_width=1.15,
        max_height_noise=0.12,
        rough_terrain=(
            terrain_gen.HfRandomUniformTerrainCfg(
                noise_range=(-0.025, 0.040),
                noise_step=0.010,
                border_width=0.25,
                vertical_scale=0.004,
            )
            if rough
            else None
        ),
        box_objects=fdm_terrain_gen.MeshPillarTerrainCfg.BoxCfg(
            width=(0.22, 0.42),
            length=(0.55, 0.95),
            max_yx_angle=(0, 10),
            height=(1.6, 2.4),
            num_objects=box_count,
        ),
        cylinder_cfg=fdm_terrain_gen.MeshPillarTerrainCfg.CylinderCfg(
            radius=(0.12, 0.22),
            max_yx_angle=(0, 5),
            height=(1.6, 2.4),
            num_objects=cylinder_count,
        ),
    )


def _ramp_gate(proportion: float, slope_range: tuple[float, float], wall_probability: float):
    return fdm_terrain_gen.StairsRampEvalTerrainCfg(
        proportion=proportion,
        modify_ramp_slope=True,
        ramp_slope_range=slope_range,
        step_width=0.30,
        platform_width=1.05,
        center_platform_width=1.05,
        border_width=0.25,
        width_randomization=0.35,
        random_stairs_ramp_position_flipping=True,
        free_space_front=True,
        no_free_space_front=False,
        random_wall_probability=wall_probability,
        all_wall=False,
        max_height=0.45,
    )


def _stairs_gate(proportion: float, step_height_range: tuple[float, float], wall_probability: float):
    return fdm_terrain_gen.StairsRampEvalTerrainCfg(
        proportion=proportion,
        modify_step_height=True,
        step_height_range=step_height_range,
        step_width=0.30,
        platform_width=1.00,
        center_platform_width=1.00,
        border_width=0.25,
        width_randomization=0.30,
        random_stairs_ramp_position_flipping=True,
        free_space_front=True,
        no_free_space_front=False,
        random_wall_probability=wall_probability,
        all_wall=False,
        max_height=0.35,
    )


def _wall_gate(proportion: float, dim_range: list[float], width: float = 0.12):
    return fdm_terrain_gen.SingleObjectTerrainCfg(
        proportion=proportion,
        object_type="wall",
        dim_range=dim_range,
        height_range=[1.4, 2.2],
        wall_width=width,
    )


def _box_gate(proportion: float, dim_range: list[float]):
    return fdm_terrain_gen.SingleObjectTerrainCfg(
        proportion=proportion,
        object_type="box",
        dim_range=dim_range,
        height_range=[0.8, 1.6],
    )


def _cylinder_gate(proportion: float, dim_range: list[float]):
    return fdm_terrain_gen.SingleObjectTerrainCfg(
        proportion=proportion,
        object_type="cylinder",
        dim_range=dim_range,
        height_range=[1.0, 1.8],
    )


def _cross_gate(proportion: float, dim_range: list[float]):
    return fdm_terrain_gen.SingleObjectTerrainCfg(
        proportion=proportion,
        object_type="box",
        dim_range=dim_range,
        height_range=[0.9, 1.6],
        position_pattern=fdm_terrain_gen.cross_object_pattern,
    )


def _maze_with_steps(proportion: float, resolution: float = 1.55, wall_width: float = 0.16, num_stairs: int = 1):
    return fdm_terrain_gen.RandomMazeTerrainCfg(
        proportion=proportion,
        resolution=resolution,
        maze_height=1.8,
        wall_width=wall_width,
        max_increase=0.12,
        max_decrease=0.10,
        width_range=(0.85, 1.15),
        length_range=(0.85, 1.20),
        height_range=(0.8, 1.3),
        num_stairs=num_stairs,
        step_height_range=(0.06, 0.11) if num_stairs > 0 else None,
        step_width_range=(0.30, 0.42) if num_stairs > 0 else None,
        stairs_platform_width=0.85,
    )


def _advantage_sub_terrains(profile: str = "train"):
    """Terrains where geometry and terrain-induced execution error both matter."""
    if profile == "2d":
        return {
            "flat_reference": _flat_reference(0.08),
            "offset_pillar_slalom": _offset_pillar_field(0.30, (2, 4), (2, 4), rough=False),
            "thin_wall_gate": _wall_gate(0.18, [0.9, 1.5], width=0.12),
            "single_box_choke": _box_gate(0.12, [0.55, 0.95]),
            "single_cylinder_choke": _cylinder_gate(0.10, [0.18, 0.34]),
            "cross_box_offset_gate": _cross_gate(0.12, [0.38, 0.70]),
            "wide_maze": _maze_with_steps(0.10, resolution=1.75, wall_width=0.14, num_stairs=0),
        }

    if profile == "eval":
        return {
            "flat_reference": _flat_reference(0.05),
            "rough_offset_pillar_slalom": _offset_pillar_field(0.24, (3, 5), (3, 5), rough=True),
            "ramp_offset_gate": _ramp_gate(0.17, (9, 17), wall_probability=0.35),
            "stairs_offset_gate": _stairs_gate(0.17, (0.07, 0.12), wall_probability=0.30),
            "thin_wall_gate": _wall_gate(0.10, [1.05, 1.70], width=0.12),
            "cross_box_offset_gate": _cross_gate(0.10, [0.42, 0.75]),
            "maze_with_low_steps": _maze_with_steps(0.10, resolution=1.50, wall_width=0.16, num_stairs=1),
            "rough_floor_reference": _rough_floor(0.07, (-0.04, 0.055)),
        }

    if profile == "rough":
        return {
            "rough_floor_reference": _rough_floor(0.16, (-0.045, 0.060)),
            "rough_offset_pillar_slalom": _offset_pillar_field(0.28, (2, 4), (2, 4), rough=True),
            "rough_dense_pillar_gate": _offset_pillar_field(0.20, (4, 6), (3, 5), rough=True),
            "ramp_offset_gate": _ramp_gate(0.14, (10, 18), wall_probability=0.30),
            "stairs_offset_gate": _stairs_gate(0.14, (0.07, 0.12), wall_probability=0.25),
            "cross_box_offset_gate": _cross_gate(0.08, [0.42, 0.80]),
        }

    return {
        "flat_reference": _flat_reference(0.07),
        "rough_floor_reference": _rough_floor(0.09, (-0.025, 0.040)),
        "rough_offset_pillar_slalom": _offset_pillar_field(0.22, (2, 4), (2, 4), rough=True),
        "ramp_offset_gate": _ramp_gate(0.14, (8, 16), wall_probability=0.25),
        "stairs_offset_gate": _stairs_gate(0.14, (0.06, 0.11), wall_probability=0.25),
        "thin_wall_gate": _wall_gate(0.09, [0.85, 1.45], width=0.12),
        "single_box_choke": _box_gate(0.06, [0.50, 0.90]),
        "single_cylinder_choke": _cylinder_gate(0.05, [0.16, 0.30]),
        "cross_box_offset_gate": _cross_gate(0.07, [0.36, 0.70]),
        "maze_with_low_steps": _maze_with_steps(0.07, resolution=1.60, wall_width=0.15, num_stairs=1),
    }


def _terrain_generator(
    profile: str,
    *,
    size: tuple[float, float] = (10.0, 10.0),
    border_width: float = 1.0,
    border_height: float = 2.0,
    num_rows: int = 8,
    num_cols: int = 8,
    curriculum: bool = False,
):
    return terrain_gen.TerrainGeneratorCfg(
        size=size,
        border_width=border_width,
        border_height=border_height,
        num_rows=num_rows,
        num_cols=num_cols,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=curriculum,
        sub_terrains=_advantage_sub_terrains(profile),
    )


BASELINE_2D_TERRAIN_CFG = _terrain_generator(
    "2d",
    border_width=1.0,
    num_rows=8,
    num_cols=8,
)

FDM_TERRAINS_CFG = _terrain_generator(
    "train",
    border_width=1.0,
    num_rows=8,
    num_cols=8,
)

FDM_EVAL_EXTEROCEPTIVE_TERRAINS_CFG = _terrain_generator(
    "eval",
    border_width=1.0,
    num_rows=8,
    num_cols=8,
)

FDM_EXTEROCEPTIVE_TERRAINS_CFG = _terrain_generator(
    "train",
    border_width=1.0,
    num_rows=8,
    num_cols=8,
)

FDM_ROUGH_TERRAINS_CFG = _terrain_generator(
    "rough",
    border_width=1.0,
    num_rows=8,
    num_cols=8,
)

PLANNER_EVAL_CFG = _terrain_generator(
    "eval",
    border_width=2.0,
    num_rows=15,
    num_cols=20,
)

PLANNER_EVAL_2D_CFG = _terrain_generator(
    "2d",
    border_width=2.0,
    num_rows=15,
    num_cols=20,
)

PAPER_FIGURE_TERRAIN_CFG = _terrain_generator(
    "eval",
    border_width=1.0,
    num_rows=6,
    num_cols=6,
)
