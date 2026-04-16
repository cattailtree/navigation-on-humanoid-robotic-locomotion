# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import os

import isaaclab.terrains as terrain_gen
import nav_tasks.terrains as fdm_terrain_gen

from fdm import FDM_DATA_DIR

from .terrains import RslStairsCfg


###
# Baseline 2D Environment
# 统一成简单避障 baseline
###

BASELINE_2D_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=2.0,
    num_rows=20,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.HfRandomUniformTerrainCfg(
            noise_range=(5e-3, 1e-2),
            noise_step=1e-2,
            border_width=0.25,
            vertical_scale=1e-3,
            proportion=0.35,
        ),
        "flat_pillar": fdm_terrain_gen.MeshPillarTerrainCfg(
            proportion=0.25,
            box_objects=fdm_terrain_gen.MeshPillarTerrainCfg.BoxCfg(
                width=(0.4, 0.8),
                length=(0.2, 0.4),
                max_yx_angle=(0, 8),
                height=(2.5, 2.5),
                num_objects=(1, 2),
            ),
            cylinder_cfg=fdm_terrain_gen.MeshPillarTerrainCfg.CylinderCfg(
                radius=(0.25, 0.4),
                max_yx_angle=(0, 5),
                height=(2.5, 2.5),
                num_objects=(1, 2),
            ),
        ),
        "single_box": fdm_terrain_gen.SingleObjectTerrainCfg(
            object_type="box",
            dim_range=[1.0, 1.5],
            proportion=0.15,
        ),
        "single_cylinder": fdm_terrain_gen.SingleObjectTerrainCfg(
            object_type="cylinder",
            dim_range=[0.5, 0.8],
            proportion=0.10,
        ),
        "single_wall": fdm_terrain_gen.SingleObjectTerrainCfg(
            object_type="wall",
            dim_range=[1.0, 1.5],
            proportion=0.05,
        ),
        "box_cross_pattern": fdm_terrain_gen.SingleObjectTerrainCfg(
            object_type="box",
            dim_range=[0.5, 0.8],
            position_pattern=fdm_terrain_gen.cross_object_pattern,
            proportion=0.06,
        ),
        "cylinder_cross_pattern": fdm_terrain_gen.SingleObjectTerrainCfg(
            object_type="cylinder",
            dim_range=[0.25, 0.4],
            position_pattern=fdm_terrain_gen.cross_object_pattern,
            proportion=0.03,
        ),
        "wall_cross_pattern": fdm_terrain_gen.SingleObjectTerrainCfg(
            object_type="wall",
            dim_range=[1.0, 1.5],
            position_pattern=fdm_terrain_gen.cross_object_pattern,
            proportion=0.01,
            wall_width=0.2,
        ),
    },
    border_height=2.5,
)


##
# Terrain Generator
# 主训练：先做对人形更友好的避障
##
FDM_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(12.0, 12.0),
    border_width=2.0,
    num_rows=15,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # 主力：稍微加密一点的 pillar
        "outdoor": fdm_terrain_gen.MeshPillarTerrainCfg(
            proportion=0.45,
            box_objects=fdm_terrain_gen.MeshPillarTerrainCfg.BoxCfg(
                width=(0.35, 0.9),
                length=(0.2, 0.45),
                max_yx_angle=(0, 10),
                height=(2.5, 2.5),
                num_objects=(1, 2),
            ),
            cylinder_cfg=fdm_terrain_gen.MeshPillarTerrainCfg.CylinderCfg(
                radius=(0.25, 0.45),
                max_yx_angle=(0, 5),
                height=(2.5, 2.5),
                num_objects=(1, 2),
            ),
        ),

        # 单个 box
        "single_box": fdm_terrain_gen.SingleObjectTerrainCfg(
            object_type="box",
            dim_range=[1.0, 1.8],
            proportion=0.22,
        ),

        # 单个 cylinder
        "single_cylinder": fdm_terrain_gen.SingleObjectTerrainCfg(
            object_type="cylinder",
            dim_range=[0.5, 0.9],
            proportion=0.13,
        ),

        # 少量 wall，但别太多
        "single_wall": fdm_terrain_gen.SingleObjectTerrainCfg(
            object_type="wall",
            dim_range=[1.0, 1.6],
            proportion=0.01,
        ),

        # 少量 cross pattern，增加转向避障
        "box_cross_pattern": fdm_terrain_gen.SingleObjectTerrainCfg(
            object_type="box",
            dim_range=[0.5, 0.9],
            position_pattern=fdm_terrain_gen.cross_object_pattern,
            proportion=0.07,
        ),

        # 极少量宽松 maze，作为 harder sample
        "maze": fdm_terrain_gen.RandomMazeTerrainCfg(
            proportion=0.01,
            resolution=2.0,
            maze_height=3.0,
            step_height_range=(0.08, 0.12),
            step_width_range=(0.35, 0.45),
            num_stairs=0,
        ),
    },
)


# Exteroceptive training terrains：仍然以避障为主，少量 rough
FDM_EXTEROCEPTIVE_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=2.5,
    num_rows=10,
    num_cols=3,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    border_height=3.0,
    sub_terrains={
        "outdoor": fdm_terrain_gen.MeshPillarTerrainCfg(
            proportion=0.7,
            box_objects=fdm_terrain_gen.MeshPillarTerrainCfg.BoxCfg(
                width=(0.5, 0.9),
                length=(0.2, 0.4),
                max_yx_angle=(0, 8),
                height=(0.8, 2.5),
                num_objects=(1, 2),
            ),
            cylinder_cfg=fdm_terrain_gen.MeshPillarTerrainCfg.CylinderCfg(
                radius=(0.25, 0.4),
                max_yx_angle=(0, 5),
                height=(0.8, 2.5),
                num_objects=(1, 2),
            ),
        ),
        "outdoor_rough": fdm_terrain_gen.MeshPillarTerrainCfg(
            proportion=0.1,
            box_objects=fdm_terrain_gen.MeshPillarTerrainCfg.BoxCfg(
                width=(0.5, 0.9),
                length=(0.2, 0.4),
                max_yx_angle=(0, 8),
                height=(0.8, 2.5),
                num_objects=(1, 2),
            ),
            cylinder_cfg=fdm_terrain_gen.MeshPillarTerrainCfg.CylinderCfg(
                radius=(0.25, 0.4),
                max_yx_angle=(0, 5),
                height=(0.8, 2.5),
                num_objects=(1, 2),
            ),
            rough_terrain=terrain_gen.HfRandomUniformTerrainCfg(
                noise_range=(0.01, 0.03),
                noise_step=0.01,
                border_width=0.25,
            ),
        ),
        "flat": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2,
            noise_range=(5e-3, 1e-2),
            noise_step=1e-2,
            border_width=0.25,
            vertical_scale=1e-3,
        ),
    },
)


# Eval exteroceptive：只保留平地 / pillar / 轻微 rough
FDM_EVAL_EXTEROCEPTIVE_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(20.0, 20.0),
    border_width=2.5,
    num_rows=2,
    num_cols=2,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.3,
            noise_range=(5e-3, 1e-2),
            noise_step=1e-2,
            border_width=0.25,
            vertical_scale=1e-3,
        ),
        "flat_pillar": fdm_terrain_gen.MeshPillarTerrainCfg(
            proportion=0.4,
            box_objects=fdm_terrain_gen.MeshPillarTerrainCfg.BoxCfg(
                width=(0.5, 0.9),
                length=(0.2, 0.4),
                max_yx_angle=(0, 8),
                height=(0.8, 2.5),
                num_objects=(1, 2),
            ),
            cylinder_cfg=fdm_terrain_gen.MeshPillarTerrainCfg.CylinderCfg(
                radius=(0.25, 0.4),
                max_yx_angle=(0, 5),
                height=(0.8, 2.5),
                num_objects=(1, 2),
            ),
        ),
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15,
            noise_range=(0.01, 0.03),
            noise_step=0.01,
            border_width=0.25,
        ),
        "rough_pillar": fdm_terrain_gen.MeshPillarTerrainCfg(
            proportion=0.15,
            box_objects=fdm_terrain_gen.MeshPillarTerrainCfg.BoxCfg(
                width=(0.5, 0.9),
                length=(0.2, 0.4),
                max_yx_angle=(0, 8),
                height=(0.8, 2.5),
                num_objects=(1, 2),
            ),
            cylinder_cfg=fdm_terrain_gen.MeshPillarTerrainCfg.CylinderCfg(
                radius=(0.25, 0.4),
                max_yx_angle=(0, 5),
                height=(0.8, 2.5),
                num_objects=(1, 2),
            ),
            rough_terrain=terrain_gen.HfRandomUniformTerrainCfg(
                noise_range=(0.01, 0.03),
                noise_step=0.01,
                border_width=0.25,
            ),
        ),
    },
)


# Maze：保留，但降权、降难度，只做辅助
MAZE_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(20.0, 20.0),
    border_width=1.0,
    border_height=3.0,
    num_cols=8,
    num_rows=6,
    use_cache=False,
    sub_terrains={
        "maze": fdm_terrain_gen.RandomMazeTerrainCfg(
            proportion=0.1,
            resolution=1.75,
            maze_height=3.0,
            step_height_range=(0.08, 0.12),
            step_width_range=(0.35, 0.45),
            num_stairs=0,
        ),
    },
)
MAZE_MERGE_TERRAIN_CFG = MAZE_TERRAIN_CFG.replace(num_cols=3, num_rows=6)


BASELINE_FLAT_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=1.0,
    num_rows=10,
    num_cols=3,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "semi_flat": terrain_gen.MeshRandomGridTerrainCfg(
            grid_width=6.75,
            grid_height_range=(0.05, 0.05),
            platform_width=12.0,
            holes=False,
        ),
    },
    border_height=2.0,
)

###
# FDM ACCURACY EVALUATION TERRAINS
# 统一成避障评估
###

PILLAR_EVAL_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=2.5,
    num_rows=10,
    num_cols=8,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "random": fdm_terrain_gen.MeshPillarTerrainCfg(
            box_objects=fdm_terrain_gen.MeshPillarTerrainCfg.BoxCfg(
                width=(0.5, 0.9),
                length=(0.2, 0.4),
                max_yx_angle=(0, 8),
                height=(1.5, 3.0),
                num_objects=(1, 2),
            ),
            cylinder_cfg=fdm_terrain_gen.MeshPillarTerrainCfg.CylinderCfg(
                radius=(0.25, 0.4),
                max_yx_angle=(0, 5),
                height=(1.5, 3.0),
                num_objects=(1, 2),
            ),
            proportion=1.0,
        ),
    },
    border_height=3.0,
)


# 保留名字，语义改为轻量避障 rough
FDM_ROUGH_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=2.5,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "outdoor_rough": fdm_terrain_gen.MeshPillarTerrainCfg(
            proportion=1.0,
            box_objects=fdm_terrain_gen.MeshPillarTerrainCfg.BoxCfg(
                width=(0.5, 0.9),
                length=(0.2, 0.4),
                max_yx_angle=(0, 8),
                height=(0.8, 2.5),
                num_objects=(1, 2),
            ),
            cylinder_cfg=fdm_terrain_gen.MeshPillarTerrainCfg.CylinderCfg(
                radius=(0.25, 0.4),
                max_yx_angle=(0, 5),
                height=(0.8, 2.5),
                num_objects=(1, 2),
            ),
            rough_terrain=terrain_gen.HfRandomUniformTerrainCfg(
                noise_range=(0.01, 0.03),
                noise_step=0.01,
                border_width=0.25,
            ),
        ),
    },
)


# 旧名字保留，统一映射为避障评估语义
STAIRS_WALL_EVAL_CFG = PILLAR_EVAL_CFG
STAIRS_EVAL_CFG = PILLAR_EVAL_CFG
STAIRS_RAMP_EVAL_CFG = PILLAR_EVAL_CFG
STAIRS_RAMP_LARGE_EVAL_CFG = PILLAR_EVAL_CFG


###
# FDM PLANNING EVAL TERRAINS
# 统一为避障 planning eval
###

PLANNER_EVAL_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=2.0,
    num_rows=2,
    num_cols=12,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
    sub_terrains={
        "single_box": fdm_terrain_gen.SingleObjectTerrainCfg(
            object_type="box",
            dim_range=[1.0, 1.8],
            proportion=0.30,
        ),
        "single_cylinder": fdm_terrain_gen.SingleObjectTerrainCfg(
            object_type="cylinder",
            dim_range=[0.5, 0.9],
            proportion=0.20,
        ),
        "single_wall": fdm_terrain_gen.SingleObjectTerrainCfg(
            object_type="wall",
            dim_range=[1.0, 1.8],
            proportion=0.10,
        ),
        "box_cross_pattern": fdm_terrain_gen.SingleObjectTerrainCfg(
            object_type="box",
            dim_range=[0.5, 1.0],
            position_pattern=fdm_terrain_gen.cross_object_pattern,
            proportion=0.15,
        ),
        "cylinder_cross_pattern": fdm_terrain_gen.SingleObjectTerrainCfg(
            object_type="cylinder",
            dim_range=[0.25, 0.5],
            position_pattern=fdm_terrain_gen.cross_object_pattern,
            proportion=0.10,
        ),
        "wall_cross_pattern": fdm_terrain_gen.SingleObjectTerrainCfg(
            object_type="wall",
            dim_range=[1.0, 1.8],
            position_pattern=fdm_terrain_gen.cross_object_pattern,
            proportion=0.05,
        ),
        "pillar": fdm_terrain_gen.MeshPillarPlannerTestTerrainCfg(
            proportion=0.10,
            box_objects=fdm_terrain_gen.MeshPillarTerrainCfg.BoxCfg(
                width=(0.4, 0.7),
                length=(0.2, 0.4),
                max_yx_angle=(0, 8),
                height=(1.5, 2.0),
                num_objects=(1, 2),
            ),
            cylinder_cfg=fdm_terrain_gen.MeshPillarTerrainCfg.CylinderCfg(
                radius=(0.25, 0.4),
                max_yx_angle=(0, 5),
                height=(1.5, 2.0),
                num_objects=(1, 2),
            ),
            platform_width=2.5,
        ),
    },
    border_height=3.0,
)


PLANNER_EVAL_2D_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=1.5,
    num_rows=12,
    num_cols=8,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "pillar": fdm_terrain_gen.MeshPillarPlannerTestTerrainCfg(
            box_objects=fdm_terrain_gen.MeshPillarTerrainCfg.BoxCfg(
                width=(0.4, 0.7),
                length=(0.2, 0.4),
                max_yx_angle=(0, 8),
                height=(1.5, 2.0),
                num_objects=(1, 2),
            ),
            cylinder_cfg=fdm_terrain_gen.MeshPillarTerrainCfg.CylinderCfg(
                radius=(0.25, 0.4),
                max_yx_angle=(0, 5),
                height=(1.5, 2.0),
                num_objects=(1, 2),
            ),
            proportion=0.35,
            platform_width=2.5,
        ),
        "single_box": fdm_terrain_gen.SingleObjectTerrainCfg(
            object_type="box",
            dim_range=[1.0, 1.8],
            proportion=0.20,
        ),
        "single_cylinder": fdm_terrain_gen.SingleObjectTerrainCfg(
            object_type="cylinder",
            dim_range=[0.5, 0.9],
            proportion=0.15,
        ),
        "single_wall": fdm_terrain_gen.SingleObjectTerrainCfg(
            object_type="wall",
            dim_range=[1.0, 1.8],
            proportion=0.05,
        ),
        "box_cross_pattern": fdm_terrain_gen.SingleObjectTerrainCfg(
            object_type="box",
            dim_range=[0.5, 1.0],
            position_pattern=fdm_terrain_gen.cross_object_pattern,
            proportion=0.15,
        ),
        "cylinder_cross_pattern": fdm_terrain_gen.SingleObjectTerrainCfg(
            object_type="cylinder",
            dim_range=[0.25, 0.5],
            position_pattern=fdm_terrain_gen.cross_object_pattern,
            proportion=0.07,
        ),
        "wall_cross_pattern": fdm_terrain_gen.SingleObjectTerrainCfg(
            object_type="wall",
            dim_range=[1.0, 1.8],
            position_pattern=fdm_terrain_gen.cross_object_pattern,
            proportion=0.03,
            wall_width=0.2,
        ),
    },
    border_height=0.0,
)


PLANNER_EVAL_3D_CFG = PLANNER_EVAL_CFG


###
# PAPER PLOT TERRAINS
# 当前阶段也统一成避障语义
###

PAPER_FIGURE_TERRAIN_CFG = PLANNER_EVAL_CFG
PAPER_PLATFORM_FIGURE_TERRAIN_CFG = BASELINE_FLAT_TERRAIN_CFG
PAPER_PLANNER_FIGURE_TERRAIN_CFG = PLANNER_EVAL_CFG