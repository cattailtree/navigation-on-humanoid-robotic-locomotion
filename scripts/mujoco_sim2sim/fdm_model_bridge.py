from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

try:
    from .config import Sim2SimConfig
except ImportError:
    from config import Sim2SimConfig


DEFAULT_RUN_DIR = Path(r"C:\Users\Admin\fdm\logs\fdm\fdm_se2_prediction_depth\May12_14-21-45_fdm_train")

torch = None
yaml = None


def _ensure_torch():
    global torch
    if torch is None:
        import torch as torch_module

        torch = torch_module
    return torch


def _ensure_yaml():
    global yaml
    if yaml is None:
        import yaml as yaml_module

        yaml = yaml_module
    return yaml


def install_omni_log_stub() -> None:
    """Allow importing IsaacLab math utilities outside a full Isaac Sim runtime."""
    if "omni.log" in sys.modules:
        return

    omni_module = sys.modules.get("omni")
    if omni_module is None:
        omni_module = types.ModuleType("omni")
        sys.modules["omni"] = omni_module

    log_module = types.ModuleType("omni.log")
    log_module.warn = lambda *args, **kwargs: None
    log_module.warning = lambda *args, **kwargs: None
    log_module.info = lambda *args, **kwargs: None
    log_module.error = lambda *args, **kwargs: None
    setattr(omni_module, "log", log_module)
    sys.modules["omni.log"] = log_module


def ensure_fdm_on_path(repo_root: Path | None = None) -> None:
    repo_root = repo_root or Path(__file__).resolve().parents[2]
    fdm_ext = repo_root / "exts" / "fdm"
    if str(fdm_ext) not in sys.path:
        sys.path.insert(0, str(fdm_ext))
    install_omni_log_stub()


def load_saved_model_cfg(run_dir: Path = DEFAULT_RUN_DIR):
    yaml_module = _ensure_yaml()
    cfg_path = run_dir / "params" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing saved FDM config: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as cfg_file:
        cfg = yaml_module.load(cfg_file, Loader=yaml_module.UnsafeLoader)
    return cfg["model_cfg"]


def cfg_get(cfg, key: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def resolve_class(path_or_class):
    if not isinstance(path_or_class, str):
        return path_or_class
    module_name, class_name = path_or_class.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def apply_cfg_dict(target, source: dict) -> None:
    for key, value in source.items():
        if key == "class_type":
            setattr(target, key, resolve_class(value))
            continue
        if not hasattr(target, key):
            continue
        current = getattr(target, key)
        if isinstance(value, dict) and current is not None:
            apply_cfg_dict(current, value)
        else:
            setattr(target, key, value)


def build_model_cfg(saved_cfg):
    if not isinstance(saved_cfg, dict):
        return saved_cfg

    ensure_fdm_on_path()
    from fdm.model.fdm_model_cfg import FDMHeightModelMultiStepCfg

    model_cfg = FDMHeightModelMultiStepCfg()
    apply_cfg_dict(model_cfg, saved_cfg)
    return model_cfg


def infer_dims(model_cfg, state_dict: dict) -> tuple[int, int, int, int, tuple[int, int]]:
    history = int(cfg_get(model_cfg, "history_length"))
    horizon = int(cfg_get(model_cfg, "prediction_horizon"))
    proprio_dim = int(state_dict["proprioceptive_normalizer._mean"].numel())
    state_plus_proprio = int(state_dict["state_obs_proprioceptive_encoder.weight_ih_l0"].shape[1])
    state_dim = state_plus_proprio - proprio_dim
    height_shape = tuple(cfg_get(model_cfg, "height_scan_shape", Sim2SimConfig.height_scan_shape))
    if height_shape == (120, 92):
        height_shape = Sim2SimConfig.height_scan_shape
    return history, horizon, state_dim, proprio_dim, height_shape


def _checkpoint_has_geometric_collision_head(state_dict: dict) -> bool:
    return any(str(key).startswith("geometric_collision_") for key in state_dict.keys())


def load_fdm_model(
    checkpoint: Path,
    run_dir: Path = DEFAULT_RUN_DIR,
    device: str = "cpu",
):
    torch_module = _ensure_torch()
    ensure_fdm_on_path()
    model_cfg = build_model_cfg(load_saved_model_cfg(run_dir))
    state_dict = torch_module.load(checkpoint, map_location=device, weights_only=True)
    has_geometric_head = _checkpoint_has_geometric_collision_head(state_dict)
    if hasattr(model_cfg, "use_geometric_collision_head"):
        model_cfg.use_geometric_collision_head = has_geometric_head
    if hasattr(model_cfg, "geometric_collision_loss_weight") and not has_geometric_head:
        model_cfg.geometric_collision_loss_weight = 0.0
    history, horizon, state_dim, proprio_dim, height_shape = infer_dims(model_cfg, state_dict)
    model = model_cfg.class_type(cfg=model_cfg, device=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    dims = {
        "history": history,
        "horizon": horizon,
        "state_dim": state_dim,
        "proprio_dim": proprio_dim,
        "height_shape": height_shape,
    }
    return model, dims
