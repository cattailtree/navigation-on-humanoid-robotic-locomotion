# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn


class ActionCVAE(nn.Module):
    """CVAE for trajectory-action generation conditioned on MPPI mean actions."""

    def __init__(
        self,
        action_dim: int,
        planning_horizon: int,
        latent_dim: int = 16,
        encoder_hidden_dim: int = 128,
        cond_hidden_dim: int = 128,
        decoder_hidden_dim: int = 128,
        context_hidden_dim: int = 128,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.planning_horizon = planning_horizon
        self.latent_dim = latent_dim
        self.cond_dim = action_dim * planning_horizon
        self.output_dim = action_dim * planning_horizon

        self.cond_encoder = nn.Sequential(
            nn.Linear(self.cond_dim, cond_hidden_dim),
            nn.LayerNorm(cond_hidden_dim),
            nn.SiLU(),
        )
        self.context_encoder = nn.Sequential(
            nn.LazyLinear(context_hidden_dim),
            nn.LayerNorm(context_hidden_dim),
            nn.SiLU(),
            nn.Linear(context_hidden_dim, cond_hidden_dim),
        )
        self.encoder = nn.Sequential(
            nn.Linear(self.output_dim + cond_hidden_dim, encoder_hidden_dim),
            nn.LayerNorm(encoder_hidden_dim),
            nn.SiLU(),
            nn.Linear(encoder_hidden_dim, encoder_hidden_dim),
            nn.SiLU(),
        )
        self.encoder_mu = nn.Linear(encoder_hidden_dim, latent_dim)
        self.encoder_logvar = nn.Linear(encoder_hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(cond_hidden_dim + latent_dim, decoder_hidden_dim),
            nn.SiLU(),
            nn.Linear(decoder_hidden_dim, decoder_hidden_dim),
            nn.SiLU(),
            nn.Linear(decoder_hidden_dim, self.output_dim),
        )

    def _condition_features(self, cond: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        cond_flat = cond.reshape(cond.shape[0], -1)
        cond_feat = self.cond_encoder(cond_flat)
        if context is not None:
            context = context.reshape(context.shape[0], -1)
            cond_feat = cond_feat + self.context_encoder(context)
        return cond_feat

    def encode(
        self, target: torch.Tensor, cond: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode target trajectory with condition into latent Gaussian parameters."""
        bs, horizon, action_dim = cond.shape
        if target.shape != cond.shape:
            raise ValueError(f"Target shape {target.shape} must equal cond shape {cond.shape}.")
        if horizon != self.planning_horizon or action_dim != self.action_dim:
            raise ValueError(
                "Condition shape mismatch. "
                f"Expected (*, {self.planning_horizon}, {self.action_dim}), got (*, {horizon}, {action_dim})."
            )
        cond_feat = self._condition_features(cond, context=context)
        enc_in = torch.cat((target.reshape(bs, -1), cond_feat), dim=-1)
        h = self.encoder(enc_in)
        return self.encoder_mu(h), self.encoder_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, cond: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        bs, horizon, action_dim = cond.shape
        cond_feat = self._condition_features(cond, context=context)
        decoder_in = torch.cat([cond_feat, z], dim=-1)
        out = self.decoder(decoder_in)
        return out.view(bs, horizon, action_dim)

    def forward(
        self, target: torch.Tensor, cond: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Train-time forward pass for reconstruction + KL loss."""
        mu, logvar = self.encode(target, cond, context=context)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, cond, context=context)
        return recon, mu, logvar

    @staticmethod
    def loss(
        recon: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta_kl: float = 1e-3,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        recon_loss = torch.nn.functional.mse_loss(recon, target)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total = recon_loss + beta_kl * kl
        return total, {"loss": total.detach(), "recon_loss": recon_loss.detach(), "kl_loss": kl.detach()}

    @torch.inference_mode()
    def sample(
        self, cond: torch.Tensor, num_samples: int, temperature: float = 1.0, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Sample action trajectories from the CVAE decoder.

        Args:
            cond: (BS, H, A) tensor, typically MPPI mean trajectory.
            num_samples: number of trajectories to sample per environment.
            temperature: latent noise scale.

        Returns:
            Tensor with shape (N, BS, H, A).
        """
        bs, horizon, action_dim = cond.shape
        if horizon != self.planning_horizon or action_dim != self.action_dim:
            raise ValueError(
                "Condition shape mismatch. "
                f"Expected (*, {self.planning_horizon}, {self.action_dim}), got (*, {horizon}, {action_dim})."
            )

        z = torch.randn(num_samples, bs, self.latent_dim, device=cond.device, dtype=cond.dtype) * temperature
        cond_expanded = cond.unsqueeze(0).expand(num_samples, -1, -1, -1).reshape(-1, horizon, action_dim)
        if context is not None:
            context = context.unsqueeze(0).expand(num_samples, -1, -1).reshape(-1, context.shape[-1])
        z = z.reshape(-1, self.latent_dim)
        out = self.decode(z, cond_expanded, context=context)
        out = out.view(num_samples, bs, horizon, action_dim)
        return out.view(num_samples, bs, horizon, action_dim)


class CVAEActionSampler:
    """Wrapper for CVAE-based action trajectory sampling inside MPPI."""

    def __init__(
        self,
        action_dim: int,
        planning_horizon: int,
        latent_dim: int = 16,
        temperature: float = 1.0,
        checkpoint: str | None = None,
        device: torch.device | str = "cpu",
    ):
        self.device = torch.device(device)
        self.temperature = temperature
        self.model = ActionCVAE(
            action_dim=action_dim,
            planning_horizon=planning_horizon,
            latent_dim=latent_dim,
        ).to(self.device)
        self.model.eval()

        if checkpoint is not None and len(checkpoint) > 0:
            state = torch.load(checkpoint, map_location=self.device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.model.load_state_dict(state, strict=False)

    @torch.inference_mode()
    def sample_population(
        self, mean: torch.Tensor, population_size: int, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Sample action trajectories around current mean.

        Returns:
            (N, BS, H, A) sampled actions.
        """
        sampled = self.model.sample(mean, num_samples=population_size, temperature=self.temperature, context=context)
        return sampled
