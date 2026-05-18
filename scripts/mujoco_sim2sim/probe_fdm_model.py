from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import Sim2SimConfig
from fdm_model_bridge import DEFAULT_RUN_DIR, load_fdm_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe the saved FDM model with synthetic MuJoCo-side tensors.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint", type=Path, default=Sim2SimConfig.fdm_checkpoint)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, dims = load_fdm_model(checkpoint=args.checkpoint, run_dir=args.run_dir, device=args.device)

    batch_size = args.batch_size
    state = torch.zeros(batch_size, dims["history"], dims["state_dim"], device=args.device)
    obs_proprioceptive = torch.zeros(batch_size, dims["history"], dims["proprio_dim"], device=args.device)
    obs_exteroceptive = torch.zeros(
        batch_size, 1, dims["height_shape"][0], dims["height_shape"][1], device=args.device
    )
    actions = torch.zeros(batch_size, dims["horizon"], 3, device=args.device)
    add_obs_exteroceptive = torch.zeros(batch_size, 1, device=args.device)

    with torch.no_grad():
        model_out = model((state, obs_proprioceptive, obs_exteroceptive, actions, add_obs_exteroceptive))
        state_traj, collision_prob, energy = model_out[0], model_out[1], model_out[2]
        geometric_collision_prob = model_out[3] if len(model_out) > 3 else None

    print("[FDM] checkpoint loaded")
    print(
        f"[FDM] history={dims['history']} horizon={dims['horizon']} state_dim={dims['state_dim']} "
        f"proprio_dim={dims['proprio_dim']}"
    )
    print(f"[FDM] height_shape={dims['height_shape']}")
    print(f"[FDM] state_traj_shape={tuple(state_traj.shape)}")
    print(f"[FDM] collision_prob_shape={tuple(collision_prob.shape)}")
    if geometric_collision_prob is not None:
        print(f"[FDM] geometric_collision_prob_shape={tuple(geometric_collision_prob.shape)}")
    print(f"[FDM] energy_shape={tuple(energy.shape)}")
    print(
        "[FDM] finite="
        f"{torch.isfinite(state_traj).all().item() and torch.isfinite(collision_prob).all().item() and torch.isfinite(energy).all().item()}"
    )


if __name__ == "__main__":
    main()
