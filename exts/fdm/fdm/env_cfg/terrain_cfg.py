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


def _rough_floor(proportion: float = 0.10, noise_range: tuple[float, float] = (-0.020, 0.030)):
    return terrain_gen.HfRandomUniformTerrainCfg(
        proportion=proportion,
        noise_range=noise_range,
        noise_step=0.015,
        border_width=0.25,
        vertical_scale=0.004,
    )


def _grass_floor(proportion: float):
    return terrain_gen.HfRandomUniformTerrainCfg(
        proportion=proportion,
        noise_range=(-0.012, 0.020),
        noise_step=0.008,
        border_width=0.25,
        vertical_scale=0.003,
    )


def _snow_floor(proportion: float):
    return terrain_gen.HfRandomUniformTerrainCfg(
        proportion=proportion,
        noise_range=(-0.025, 0.040),
        noise_step=0.035,
        border_width=0.25,
        vertical_scale=0.005,
    )


def _mud_floor(proportion: float):
    return terrain_gen.HfRandomUniformTerrainCfg(
        proportion=proportion,
        noise_range=(-0.035, 0.030),
        noise_step=0.022,
        border_width=0.25,
        vertical_scale=0.006,
    )


def _offset_pillar_field(
    proportion: float,
    box_count: tuple[int, int],
    cylinder_count: tuple[int, int],
    rough: bool = True,
    platform_width: float = 1.8,
):
    return fdm_terrain_gen.MeshPillarTerrainCfg(
        proportion=proportion,
        platform_width=platform_width,
        max_height_noise=0.08,
        rough_terrain=(
            terrain_gen.HfRandomUniformTerrainCfg(
                noise_range=(-0.015, 0.025),
                noise_step=0.015,
                border_width=0.25,
                vertical_scale=0.004,
            )
            if rough
            else None
        ),
        box_objects=fdm_terrain_gen.MeshPillarTerrainCfg.BoxCfg(
            width=(0.18, 0.34),
            length=(0.42, 0.78),
            max_yx_angle=(0, 6),
            height=(1.6, 2.4),
            num_objects=box_count,
        ),
        cylinder_cfg=fdm_terrain_gen.MeshPillarTerrainCfg.CylinderCfg(
            radius=(0.10, 0.18),
            max_yx_angle=(0, 3),
            height=(1.6, 2.4),
            num_objects=cylinder_count,
        ),
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


def _advantage_sub_terrains(profile: str = "train"):
    """Terrains where geometry and terrain-induced execution error both matter."""
    if profile == "2d":
        return {
            "flat_reference": _flat_reference(0.35),
            "grass_mild_reference": _grass_floor(0.25),
            "snow_mild_reference": _snow_floor(0.15),
            "offset_pillar_sparse": _offset_pillar_field(0.12, (1, 2), (1, 2), rough=False, platform_width=2.2),
            "single_box_easy": _box_gate(0.06, [0.32, 0.58]),
            "single_cylinder_easy": _cylinder_gate(0.04, [0.10, 0.20]),
            "cross_box_easy": _cross_gate(0.03, [0.24, 0.44]),
        }

    if profile == "eval":
        return {
            "flat_reference": _flat_reference(0.12),
            "grass_reference": _grass_floor(0.18),
            "snow_reference": _snow_floor(0.14),
            "mud_reference": _mud_floor(0.12),
            "rough_offset_pillar_slalom": _offset_pillar_field(0.18, (1, 3), (1, 3), rough=True, platform_width=2.0),
            "rough_dense_pillar_gate": _offset_pillar_field(0.08, (2, 4), (2, 4), rough=True, platform_width=1.8),
            "single_box_choke": _box_gate(0.08, [0.38, 0.68]),
            "single_cylinder_choke": _cylinder_gate(0.06, [0.12, 0.24]),
            "cross_box_offset_gate": _cross_gate(0.04, [0.28, 0.52]),
        }

    if profile == "rough":
        return {
            "grass_reference": _grass_floor(0.22),
            "snow_reference": _snow_floor(0.18),
            "mud_reference": _mud_floor(0.16),
            "rough_floor_reference": _rough_floor(0.20, (-0.030, 0.045)),
            "rough_offset_pillar_slalom": _offset_pillar_field(0.16, (1, 3), (1, 3), rough=True, platform_width=2.0),
            "rough_dense_pillar_gate": _offset_pillar_field(0.06, (2, 4), (2, 4), rough=True, platform_width=1.8),
            "cross_box_offset_gate": _cross_gate(0.02, [0.30, 0.56]),
        }

    return {
        "flat_reference": _flat_reference(0.25),
        "grass_reference": _grass_floor(0.20),
        "snow_reference": _snow_floor(0.14),
        "mud_reference": _mud_floor(0.10),
        "rough_floor_reference": _rough_floor(0.11, (-0.020, 0.030)),
        "rough_offset_pillar_sparse": _offset_pillar_field(0.10, (1, 2), (1, 2), rough=True, platform_width=2.2),
        "single_box_easy": _box_gate(0.04, [0.32, 0.58]),
        "single_cylinder_easy": _cylinder_gate(0.03, [0.10, 0.20]),
        "cross_box_easy": _cross_gate(0.03, [0.24, 0.44]),
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
