# fdm/env_cfg/robot_cfg_base.py

from __future__ import annotations
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import UsdFileCfg

class UnitreeUsdFileCfg(UsdFileCfg):
    """Unitree 自定义 Usd 加载器，兼容性调整（如需）"""
    pass

class UnitreeArticulationCfg(ArticulationCfg):
    """Unitree 通用配置器，用于加载机器人模型"""
    pass
