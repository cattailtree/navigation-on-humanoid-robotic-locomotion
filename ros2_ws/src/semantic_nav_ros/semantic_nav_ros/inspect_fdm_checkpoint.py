from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect the FDM checkpoint contract used by semantic_nav_ros.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="FDM run directory. Defaults to the newest *_fdm_train run under logs/fdm/fdm_se2_prediction_depth.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("ERROR: torch is required to inspect a checkpoint.", file=sys.stderr)
        return 2

    run_dir = _resolve(args.run_dir) if args.run_dir is not None else _latest_run_dir()
    if run_dir is None:
        print("ERROR: no *_fdm_train run found under logs/fdm/fdm_se2_prediction_depth.", file=sys.stderr)
        return 1
    checkpoint = _resolve(args.checkpoint) if args.checkpoint is not None else _latest_checkpoint(run_dir)
    if checkpoint is None:
        print(f"ERROR: no model_collection_round_*.pth under {run_dir}", file=sys.stderr)
        return 1
    config_path = run_dir / "params" / "config.yaml"
    if not config_path.exists():
        print(f"ERROR: missing saved config {config_path}", file=sys.stderr)
        return 1

    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    proprio_dim = int(state_dict["proprioceptive_normalizer._mean"].numel())
    state_plus_proprio = int(state_dict["state_obs_proprioceptive_encoder.weight_ih_l0"].shape[1])
    state_dim = state_plus_proprio - proprio_dim
    has_geometry = any(str(key).startswith("geometric_collision_") for key in state_dict)
    config_text = config_path.read_text(encoding="utf-8", errors="replace")

    print(f"run_dir: {run_dir}")
    print(f"checkpoint: {checkpoint}")
    print(f"checkpoint_round: {_checkpoint_round(checkpoint)}")
    print(f"robot_hint_g1_29dof: {_contains_any(config_text, ('G1/29dof', 'g1_29dof', 'robot_cfg_g1'))}")
    print(f"has_geometric_collision_head: {has_geometry}")
    print(f"state_dim: {state_dim}")
    print(f"proprio_dim: {proprio_dim}")
    print(f"height_scan_shape: {_find_scalar(config_text, 'height_scan_shape') or '[60, 46]'}")
    print(f"history_length: {_find_scalar(config_text, 'history_length') or 'unknown'}")
    print(f"prediction_horizon: {_find_scalar(config_text, 'prediction_horizon') or 'unknown'}")
    print(f"base_lin_vel_in_config: {'base_lin_vel:' in config_text}")

    errors: list[str] = []
    if not _contains_any(config_text, ("G1/29dof", "g1_29dof", "robot_cfg_g1")):
        errors.append("saved config does not look like the G1 29DOF config")
    if not has_geometry:
        errors.append("checkpoint does not contain geometric_collision_* weights")
    if state_dim != 8:
        errors.append(f"expected state_dim=8 for current G1 FDM, got {state_dim}")
    if proprio_dim != 157:
        errors.append(f"expected proprio_dim=157 for current G1 FDM, got {proprio_dim}")
    if "base_lin_vel:" not in config_text:
        errors.append("saved config does not include base_lin_vel")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("FDM checkpoint contract OK")
    return 0


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "logs" / "fdm").exists() and (parent / "exts" / "fdm").exists():
            return parent
    return current.parents[4]


def _resolve(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.is_absolute():
        return path
    return _repo_root() / path


def _latest_run_dir() -> Path | None:
    root = _repo_root() / "logs" / "fdm" / "fdm_se2_prediction_depth"
    candidates = [
        path
        for path in root.glob("*_fdm_train")
        if path.is_dir() and (path / "params" / "config.yaml").exists() and _latest_checkpoint(path) is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = sorted(run_dir.glob("model_collection_round_*.pth"))
    if not checkpoints:
        return None
    return max(checkpoints, key=_checkpoint_round)


def _checkpoint_round(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])



def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _find_scalar(text: str, key: str) -> str | None:
    prefix = f"  {key}:"
    for line in text.splitlines():
        if line.startswith(prefix):
            value = line.split(":", 1)[1].strip()
            return value or None
    return None


if __name__ == "__main__":
    sys.exit(main())
