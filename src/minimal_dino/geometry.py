from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class GeometryConfig:
    """Parameters of paired reliability equalization (PRE), not baseline centering."""

    momentum: float = 0.96
    strength: float = 1.0
    ridge: float = 0.01
    min_gain: float = 0.5
    max_gain: float = 2.0
    warmup_steps: int = 50
    update_interval: int = 10
    reliability: str = "paired"

    def __post_init__(self) -> None:
        for name in ("momentum", "strength", "ridge", "min_gain", "max_gain"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"target_geometry.{name} must be a finite number")
            if not math.isfinite(value):
                raise ValueError(f"target_geometry.{name} must be a finite number")
        if not 0 <= self.momentum < 1 or not 0 <= self.strength <= 1:
            raise ValueError("target_geometry momentum must be in [0, 1), strength in [0, 1]")
        if self.ridge <= 0 or not 0 < self.min_gain <= 1 <= self.max_gain:
            raise ValueError("target_geometry requires ridge > 0 and 0 < min_gain <= 1 <= max_gain")
        for name in ("warmup_steps", "update_interval"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"target_geometry.{name} must be a positive integer")
        if self.reliability not in {"paired", "none", "shuffled", "diagonal"}:
            raise ValueError(
                "target_geometry.reliability must be paired, none, shuffled, or diagonal"
            )


class PairedTargetGeometry(nn.Module):
    """Causal EMA covariance and paired-noise statistics, applied before the target head.

    For h_v = s + e_v with conditionally independent, zero-mean view noise,
    E[(h_1-h_2)(h_1-h_2)^T]/2 estimates augmentation noise covariance N.
    R = clip_PSD((C+ridge I)^(-1/2) (C-N) (C+ridge I)^(-1/2), 0, 1)
    gates bounded variance equalization via a symmetric matrix sandwich. It retains
    off-diagonal reliability within degenerate eigenspaces of C. This is not a guarantee
    that augmentation preserves semantics. No labels or evaluation statistics enter here.
    """

    def __init__(self, embedding_dim: int, config: GeometryConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("mean", torch.zeros(1, embedding_dim))
        self.register_buffer("covariance", torch.zeros(embedding_dim, embedding_dim))
        self.register_buffer("noise_covariance", torch.zeros(embedding_dim, embedding_dim))
        self.register_buffer("correction", torch.zeros(embedding_dim, embedding_dim))
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("reliability", torch.zeros(embedding_dim))
        self.register_buffer("gains", torch.ones(embedding_dim))
        self.register_buffer("eigenvalues", torch.zeros(embedding_dim))
        self.register_buffer("reliability_operator", torch.zeros(embedding_dim, embedding_dim))

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        # Statistics are detached buffers. Online use is prohibited by SentenceBYOL.
        # A zero-strength ablation is bitwise identical to scalar-centered BYOL.
        if self.config.strength == 0:
            return torch.zeros_like(embedding, dtype=torch.float32)
        with torch.autocast(device_type=embedding.device.type, enabled=False):
            return (embedding.float() - self.mean) @ self.correction

    @torch.no_grad()
    def update(self, views: tuple[torch.Tensor, torch.Tensor]) -> None:
        first, second = views
        if first.shape != second.shape or first.ndim != 2:
            raise ValueError("Paired teacher embeddings must have matching [batch, dim] shapes")
        if first.shape[0] < 1 or first.shape[1] != self.mean.shape[1]:
            raise ValueError(
                "Paired teacher embeddings have an invalid batch or embedding dimension"
            )
        with torch.autocast(device_type=first.device.type, enabled=False):
            first, second = first.detach().float(), second.detach().float()
            joined = torch.cat((first, second))
            batch_mean = joined.mean(0, keepdim=True)
            centered = joined - batch_mean
            batch_covariance = centered.T @ centered / joined.shape[0]
            # Deterministic wrong-pair control: preserves marginals and consumes no RNG.
            noise_second = second.roll(1, 0) if self.config.reliability == "shuffled" else second
            difference = first - noise_second
            batch_noise = difference.T @ difference / (2 * first.shape[0])
            if self.updates.item() == 0:
                # Initialize from observations, avoiding fictitious zero-centered samples.
                self.mean.copy_(batch_mean)
                self.covariance.copy_(batch_covariance)
                self.noise_covariance.copy_(batch_noise)
            else:
                momentum = self.config.momentum
                delta = batch_mean - self.mean
                # Exact covariance of the EMA mixture, including between-batch mean drift.
                self.covariance.mul_(momentum).add_(batch_covariance, alpha=1 - momentum)
                self.covariance.add_(delta.T @ delta, alpha=momentum * (1 - momentum))
                self.mean.lerp_(batch_mean, 1 - momentum)
                self.noise_covariance.lerp_(batch_noise, 1 - momentum)
            self.updates.add_(1)
            step = int(self.updates.item())
            if step >= self.config.warmup_steps and (
                (step - self.config.warmup_steps) % self.config.update_interval == 0
            ):
                self.refresh()

    @torch.no_grad()
    def refresh(self) -> None:
        """Refresh a bounded operator; used only after the current batch's loss."""
        with torch.autocast(device_type=self.mean.device.type, enabled=False):
            covariance = (self.covariance + self.covariance.T) * 0.5
            values, vectors = torch.linalg.eigh(covariance)
            values = values.clamp_min(0)
            scale = values.mean().clamp_min(1e-12)
            denominator = values + self.config.ridge * scale
            equalization = (
                (scale / denominator).sqrt().clamp(self.config.min_gain, self.config.max_gain)
            )
            whitening_delta = (vectors * (equalization - 1)) @ vectors.T
            if self.config.reliability == "diagonal":
                # Discard off-diagonal reliability only; keep the same ridge as the full method.
                noise = (vectors * (self.noise_covariance @ vectors)).sum(0).clamp_min(0)
                diagonal = ((values - noise) / denominator).clamp(0, 1)
                root = (vectors * diagonal.sqrt()) @ vectors.T
                reliability = diagonal.sort().values
            elif self.config.reliability == "none":
                root = torch.eye(len(values), device=values.device)
                reliability = torch.ones_like(values)
            else:
                inverse_root = (vectors * denominator.rsqrt()) @ vectors.T
                reliable_covariance = (
                    inverse_root @ (covariance - self.noise_covariance) @ inverse_root
                )
                reliable_covariance = (reliable_covariance + reliable_covariance.T) * 0.5
                reliability, basis = torch.linalg.eigh(reliable_covariance)
                reliability = reliability.clamp(0, 1)
                root = (basis * reliability.sqrt()) @ basis.T
            correction = self.config.strength * (root @ whitening_delta @ root)
            correction = (correction + correction.T) * 0.5
            self.correction.copy_(correction)
            gains = 1 + torch.linalg.eigvalsh(correction)
            self.reliability_operator.copy_(root @ root)
            self.reliability.copy_(reliability)
            self.gains.copy_(gains)
            self.eigenvalues.copy_(values)

    def metrics(self) -> dict[str, torch.Tensor]:
        return {
            "geometry_updates": self.updates.clone(),
            "geometry_reliability_mean": self.reliability.mean(),
            "geometry_gain_min": self.gains.min(),
            "geometry_gain_max": self.gains.max(),
            "geometry_correction_norm": self.correction.norm(),
            "geometry_noise_trace_ratio": self.noise_covariance.trace()
            / self.covariance.trace().clamp_min(1e-12),
        }
