# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.terrains as terrain_gen

import nav_tasks.terrains as fdm_terrain_gen


def _flat_floor(proportion: float):
    return terrain_gen.HfRandomUniformTerrainCfg(
        proportion=proportion,
        noise_range=(0.0, 0.0),
        noise_step=0.005,
        border_width=0.25,
        vertical_scale=0.001,
    )


def _flat_reference(proportion: float = 0.10):
    return terrain_gen.HfRandomUniformTerrainCfg(
        proportion=proportion,
        noise_range=(0.0, 0.004),
        noise_step=0.005,
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
        noise_step=0.005,
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
        noise_step=0.025,
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


def _floor_or_flat(kind: str, proportion: float, *, enable_floor_terrain: bool):
    if not enable_floor_terrain:
        return _flat_floor(proportion)
    if kind == "flat":
        return _flat_reference(proportion)
    if kind == "grass":
        return _grass_floor(proportion)
    if kind == "snow":
        return _snow_floor(proportion)
    if kind == "mud":
        return _mud_floor(proportion)
    if kind == "rough":
        return _rough_floor(proportion)
    raise ValueError(f"Unknown floor terrain kind: {kind}")


def _advantage_sub_terrains(profile: str = "train", *, enable_floor_terrain: bool = True):
    """Terrains where geometry and terrain-induced execution error both matter."""
    if profile == "2d":
        return {
            "flat_reference": _floor_or_flat("flat", 0.35, enable_floor_terrain=enable_floor_terrain),
            "grass_mild_reference": _floor_or_flat("grass", 0.25, enable_floor_terrain=enable_floor_terrain),
            "snow_mild_reference": _floor_or_flat("snow", 0.15, enable_floor_terrain=enable_floor_terrain),
            "offset_pillar_sparse": _offset_pillar_field(0.12, (1, 2), (1, 2), rough=False, platform_width=2.2),
            "single_box_easy": _box_gate(0.06, [0.32, 0.58]),
            "single_cylinder_easy": _cylinder_gate(0.04, [0.10, 0.20]),
            "cross_box_easy": _cross_gate(0.03, [0.24, 0.44]),
        }

    if profile == "eval":
        return {
            "flat_reference": _floor_or_flat("flat", 0.12, enable_floor_terrain=enable_floor_terrain),
            "grass_reference": _floor_or_flat("grass", 0.18, enable_floor_terrain=enable_floor_terrain),
            "snow_reference": _floor_or_flat("snow", 0.14, enable_floor_terrain=enable_floor_terrain),
            "mud_reference": _floor_or_flat("mud", 0.12, enable_floor_terrain=enable_floor_terrain),
            "rough_offset_pillar_slalom": _offset_pillar_field(
                0.18, (1, 3), (1, 3), rough=enable_floor_terrain, platform_width=2.0
            ),
            "rough_dense_pillar_gate": _offset_pillar_field(
                0.08, (2, 4), (2, 4), rough=enable_floor_terrain, platform_width=1.8
            ),
            "single_box_choke": _box_gate(0.08, [0.38, 0.68]),
            "single_cylinder_choke": _cylinder_gate(0.06, [0.12, 0.24]),
            "cross_box_offset_gate": _cross_gate(0.04, [0.28, 0.52]),
        }

    if profile == "rough":
        return {
            "grass_reference": _floor_or_flat("grass", 0.22, enable_floor_terrain=enable_floor_terrain),
            "snow_reference": _floor_or_flat("snow", 0.18, enable_floor_terrain=enable_floor_terrain),
            "mud_reference": _floor_or_flat("mud", 0.16, enable_floor_terrain=enable_floor_terrain),
            "rough_floor_reference": (
                _rough_floor(0.20, (-0.030, 0.045)) if enable_floor_terrain else _flat_floor(0.20)
            ),
            "rough_offset_pillar_slalom": _offset_pillar_field(
                0.16, (1, 3), (1, 3), rough=enable_floor_terrain, platform_width=2.0
            ),
            "rough_dense_pillar_gate": _offset_pillar_field(
                0.06, (2, 4), (2, 4), rough=enable_floor_terrain, platform_width=1.8
            ),
            "cross_box_offset_gate": _cross_gate(0.02, [0.30, 0.56]),
        }

    return {
        "flat_reference": _floor_or_flat("flat", 0.25, enable_floor_terrain=enable_floor_terrain),
        "grass_reference": _floor_or_flat("grass", 0.20, enable_floor_terrain=enable_floor_terrain),
        "snow_reference": _floor_or_flat("snow", 0.14, enable_floor_terrain=enable_floor_terrain),
        "mud_reference": _floor_or_flat("mud", 0.10, enable_floor_terrain=enable_floor_terrain),
        "rough_floor_reference": (
            _rough_floor(0.11, (-0.020, 0.030)) if enable_floor_terrain else _flat_floor(0.11)
        ),
        "rough_offset_pillar_sparse": _offset_pillar_field(
            0.10, (1, 2), (1, 2), rough=enable_floor_terrain, platform_width=2.2
        ),
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
    enable_floor_terrain: bool = True,
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
        sub_terrains=_advantage_sub_terrains(profile, enable_floor_terrain=enable_floor_terrain),
    )


def _fdm_training_generator():
    return terrain_gen.TerrainGeneratorCfg(
        size=(10.0, 10.0),
        border_width=2.0,
        border_height=1.0,
        num_rows=15,
        num_cols=20,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=False,
        sub_terrains={
            "outdoor": fdm_terrain_gen.MeshPillarTerrainCfg(
                proportion=0.45,
                rough_terrain=None,
                box_objects=fdm_terrain_gen.MeshPillarTerrainCfg.BoxCfg(
                    width=(0.35, 0.9),
                    length=(0.2, 0.45),
                    max_yx_angle=(0, 10),
                    degrees=True,
                    num_objects=(1, 2),
                    height=(2.5, 2.5),
                ),
                cylinder_cfg=fdm_terrain_gen.MeshPillarTerrainCfg.CylinderCfg(
                    radius=(0.25, 0.45),
                    max_yx_angle=(0, 5),
                    degrees=True,
                    num_objects=(1, 2),
                    height=(2.5, 2.5),
                ),
                max_height_noise=0.0,
                platform_width=1.0,
            ),
            "single_box": fdm_terrain_gen.SingleObjectTerrainCfg(
                proportion=0.22,
                object_type="box",
                dim_range=[1.0, 1.8],
                height_range=[0.5, 1.5],
            ),
            "single_cylinder": fdm_terrain_gen.SingleObjectTerrainCfg(
                proportion=0.13,
                object_type="cylinder",
                dim_range=[0.5, 0.9],
                height_range=[0.5, 1.5],
            ),
            "single_wall": fdm_terrain_gen.SingleObjectTerrainCfg(
                proportion=0.01,
                object_type="wall",
                dim_range=[1.0, 1.6],
                height_range=[0.5, 1.5],
            ),
            "box_cross_pattern": fdm_terrain_gen.SingleObjectTerrainCfg(
                proportion=0.07,
                object_type="box",
                dim_range=[0.5, 0.9],
                height_range=[0.5, 1.5],
                position_pattern=fdm_terrain_gen.cross_object_pattern,
            ),
            "maze": fdm_terrain_gen.RandomMazeTerrainCfg(
                proportion=0.01,
                resolution=2.0,
                maze_height=3.0,
                wall_width=0.2,
                max_increase=0.0,
                max_decrease=0.0,
                width_range=(1.0, 1.0),
                length_range=(1.0, 1.0),
                height_range=(1.0, 1.0),
                num_stairs=0,
                step_height_range=(0.08, 0.12),
                step_width_range=(0.35, 0.45),
                stairs_platform_width=1.0,
            ),
        },
    )


BASELINE_2D_TERRAIN_CFG = _terrain_generator(
    "2d",
    border_width=1.0,
    num_rows=8,
    num_cols=8,
)

FDM_TRAINING_TERRAINS_CFG = _fdm_training_generator()

FDM_TERRAINS_CFG = _terrain_generator(
    "train",
    border_width=1.0,
    num_rows=8,
    num_cols=8,
)

FDM_TERRAINS_NO_TERRAIN_CFG = _terrain_generator(
    "train",
    border_width=1.0,
    num_rows=8,
    num_cols=8,
    enable_floor_terrain=False,
)

FDM_EVAL_EXTEROCEPTIVE_TERRAINS_CFG = _terrain_generator(
    "eval",
    border_width=1.0,
    num_rows=8,
    num_cols=8,
)

FDM_EVAL_EXTEROCEPTIVE_TERRAINS_NO_TERRAIN_CFG = _terrain_generator(
    "eval",
    border_width=1.0,
    num_rows=8,
    num_cols=8,
    enable_floor_terrain=False,
)

FDM_EXTEROCEPTIVE_TERRAINS_CFG = _terrain_generator(
    "train",
    border_width=1.0,
    num_rows=8,
    num_cols=8,
)

FDM_EXTEROCEPTIVE_TERRAINS_NO_TERRAIN_CFG = _terrain_generator(
    "train",
    border_width=1.0,
    num_rows=8,
    num_cols=8,
    enable_floor_terrain=False,
)

FDM_ROUGH_TERRAINS_CFG = _terrain_generator(
    "rough",
    border_width=1.0,
    num_rows=8,
    num_cols=8,
)

FDM_ROUGH_TERRAINS_NO_TERRAIN_CFG = _terrain_generator(
    "rough",
    border_width=1.0,
    num_rows=8,
    num_cols=8,
    enable_floor_terrain=False,
)

PLANNER_EVAL_CFG = _terrain_generator(
    "eval",
    border_width=2.0,
    num_rows=15,
    num_cols=20,
)

PLANNER_EVAL_NO_TERRAIN_CFG = _terrain_generator(
    "eval",
    border_width=2.0,
    num_rows=15,
    num_cols=20,
    enable_floor_terrain=False,
)

PLANNER_EVAL_2D_CFG = _terrain_generator(
    "2d",
    border_width=2.0,
    num_rows=15,
    num_cols=20,
)

PLANNER_EVAL_2D_NO_TERRAIN_CFG = _terrain_generator(
    "2d",
    border_width=2.0,
    num_rows=15,
    num_cols=20,
    enable_floor_terrain=False,
)

PAPER_FIGURE_TERRAIN_CFG = _terrain_generator(
    "eval",
    border_width=1.0,
    num_rows=6,
    num_cols=6,
)

PAPER_FIGURE_NO_TERRAIN_CFG = _terrain_generator(
    "eval",
    border_width=1.0,
    num_rows=6,
    num_cols=6,
    enable_floor_terrain=False,
)
