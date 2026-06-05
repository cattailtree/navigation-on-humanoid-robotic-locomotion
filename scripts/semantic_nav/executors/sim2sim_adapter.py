from __future__ import annotations

from pathlib import Path
import sys

from executors.robot_adapter import VelocityCommand


SIM2SIM_ROOT = Path(__file__).resolve().parents[2] / "mujoco_sim2sim"
if str(SIM2SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM2SIM_ROOT))

from low_level_controller import LowLevelCommand  # noqa: E402


def to_low_level_command(command: VelocityCommand) -> LowLevelCommand:
    return LowLevelCommand(vx=command.vx, vy=command.vy, wz=command.wz)
