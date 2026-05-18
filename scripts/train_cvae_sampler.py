# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# SPDX-License-Identifier: BSD-3-Clause

"""Train a CVAE sampler for MPPI trajectory generation.

Dataset format (.pt):
- mean_actions: (N, H, A) MPPI mean trajectories used as condition.
- target_actions: (N, H, A) selected candidate trajectories used as targets.
- context: optional (N, C) planner context vector.
- sample_weight: optional (N,) reconstruction weights.

Newer v1.1 datasets may also include goal/risk labels. They are loaded and
validated by the collection pipeline, but this script keeps the deployed sampler
compatible by training the same reconstruction + KL CVAE objective.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from fdm.planner.sampling_planner.cvae_action_sampler import ActionCVAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CVAE action sampler.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to .pt dataset with mean_actions/target_actions.")
    parser.add_argument("--out", type=str, required=True, help="Output checkpoint path (.pt).")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--beta-kl", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--context-key",
        type=str,
        default="context",
        help="Optional key for context tensor. Set to an empty string to disable context.",
    )
    return parser.parse_args()


def load_dataset(path: str, context_key: str = "context") -> TensorDataset:
    data = torch.load(path, map_location="cpu")
    if "mean_actions" not in data or "target_actions" not in data:
        raise KeyError("Dataset must contain keys: mean_actions and target_actions.")

    mean_actions = data["mean_actions"].float()
    target_actions = data["target_actions"].float()
    if mean_actions.shape != target_actions.shape:
        raise ValueError(f"Shape mismatch: {mean_actions.shape=} {target_actions.shape=}")

    tensors: list[torch.Tensor] = [mean_actions, target_actions]
    if len(context_key) > 0 and context_key in data:
        context = data[context_key].float()
        if context.shape[0] != mean_actions.shape[0]:
            raise ValueError(f"Context length mismatch: {context.shape[0]=}, {mean_actions.shape[0]=}")
        tensors.append(context)

    if "sample_weight" in data:
        sample_weight = data["sample_weight"].float().reshape(-1)
        if sample_weight.shape[0] != mean_actions.shape[0]:
            raise ValueError(f"sample_weight length mismatch: {sample_weight.shape[0]=}, {mean_actions.shape[0]=}")
        tensors.append(sample_weight)

    return TensorDataset(*tensors)


def cvae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta_kl: float,
    sample_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    recon_per_sample = torch.mean((recon - target) ** 2, dim=(1, 2))
    if sample_weight is not None:
        sample_weight = sample_weight.to(recon_per_sample.device)
        recon_loss = torch.sum(recon_per_sample * sample_weight) / torch.clamp(sample_weight.sum(), min=1e-8)
    else:
        recon_loss = recon_per_sample.mean()
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + beta_kl * kl
    return total, {"loss": total.detach(), "recon_loss": recon_loss.detach(), "kl_loss": kl.detach()}


def split_batch(batch: tuple[torch.Tensor, ...], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    cond = batch[0].to(device)
    target = batch[1].to(device)
    context = None
    sample_weight = None
    if len(batch) == 3:
        if batch[2].ndim == 1:
            sample_weight = batch[2].to(device)
        else:
            context = batch[2].to(device)
    elif len(batch) == 4:
        context = batch[2].to(device)
        sample_weight = batch[3].to(device)
    return cond, target, context, sample_weight


def run_epoch(
    model: ActionCVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    beta_kl: float,
    device: torch.device,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)

    totals = {"loss": 0.0, "recon_loss": 0.0, "kl_loss": 0.0}
    n = 0
    for batch in loader:
        cond, target, context, sample_weight = split_batch(batch, device)

        with torch.set_grad_enabled(train):
            recon, mu, logvar = model(target=target, cond=cond, context=context)
            loss, stats = cvae_loss(
                recon=recon,
                target=target,
                mu=mu,
                logvar=logvar,
                beta_kl=beta_kl,
                sample_weight=sample_weight,
            )

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        bs = cond.shape[0]
        n += bs
        for key in totals:
            totals[key] += stats[key].item() * bs

    return {key: value / max(1, n) for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    dataset = load_dataset(args.dataset, context_key=args.context_key)
    n_val = int(len(dataset) * args.val_ratio)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False)

    sample_cond = dataset[0][0]
    horizon, action_dim = sample_cond.shape
    device = torch.device(args.device)

    model = ActionCVAE(action_dim=action_dim, planning_horizon=horizon, latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model, train_loader, optimizer, args.beta_kl, device)
        val_stats = run_epoch(model, val_loader, None, args.beta_kl, device)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_stats['loss']:.6f} train_recon={train_stats['recon_loss']:.6f} "
            f"train_kl={train_stats['kl_loss']:.6f} "
            f"val_loss={val_stats['loss']:.6f} val_recon={val_stats['recon_loss']:.6f} "
            f"val_kl={val_stats['kl_loss']:.6f}"
        )

        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if best_state is None:
        best_state = model.state_dict()

    torch.save(
        {
            "state_dict": best_state,
            "action_dim": action_dim,
            "planning_horizon": horizon,
            "latent_dim": args.latent_dim,
            "beta_kl": args.beta_kl,
        },
        out_path,
    )
    print(f"Saved best checkpoint to: {out_path}")


if __name__ == "__main__":
    main()
