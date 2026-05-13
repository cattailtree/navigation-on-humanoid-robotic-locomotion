# CVAE-MPPI End-to-End Pipeline

## 1) Online dataset collection (from MPPI planning)
Enable dataset dumping in `TrajectoryOptimizerCfg`:

- `cvae_dataset_dump_path`: output `.pt` file path.
- `cvae_dataset_topk`: top-k trajectories per env used as target actions.
- `cvae_dataset_max_samples`: retention cap.

When enabled, each planning step stores:
- `mean_actions`  `(N, H, A)` from current MPPI mean trajectory.
- `target_actions` `(N, H, A)` from top-k scoring sampled trajectories.
- optional `context` `(N, C)` if perception/state/history context exists.

## 2) Train CVAE sampler

```bash
python scripts/train_cvae_sampler.py \
  --dataset /tmp/cvae_dataset.pt \
  --out /tmp/cvae_sampler.pt \
  --epochs 50 \
  --batch-size 256 \
  --latent-dim 16 \
  --beta-kl 1e-3
```

If your dataset contains context under key `context`, this script loads it automatically.
Set `--context-key ""` to disable context.

## 3) Deploy in planner
Set MPPI optimizer config:

- `sampling_strategy="cvae"`
- `cvae_checkpoint="/tmp/cvae_sampler.pt"`
- `cvae_latent_dim=16`
- `cvae_temperature=1.0`

During inference, CVAE samples trajectory population conditioned on:
- `mean_actions`
- optional context vector assembled from planner obs (`goal`, `proprio`, `proprioception`, `history`, `height_scan`, `state`),
  or directly provided by `obs["cvae_context"]`.

## 4) I/O contract summary

### Training
- Input: `mean_actions`, optional `context`, and `target_actions` supervision.
- Output: reconstructed trajectory and latent params (`mu`, `logvar`).
- Loss: `MSE(recon, target) + beta_kl * KL`.

### Deployment
- Input: current `mean_actions` + optional context.
- Output: sampled candidate trajectory population for MPPI scoring.
