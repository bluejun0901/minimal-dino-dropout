from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class UniformityLoss(nn.Module):
    """Wang & Isola (2020): log mean exp(-t ||z_i - z_j||²), for i < j.

    Inputs are pooled sentence embeddings; normalization and pair distances use
    FP32 even during mixed-precision training. Batches with fewer than two
    samples contribute a differentiable zero because they contain no pairs.
    """

    def __init__(self, t: float = 2.0) -> None:
        super().__init__()
        if (
            isinstance(t, bool)
            or not isinstance(t, (int, float))
            or not math.isfinite(t)
            or t <= 0
        ):
            raise ValueError("uniformity t must be a finite positive number")
        self.t = t

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=embeddings.device.type, enabled=False):
            embeddings = embeddings.float()
            if embeddings.shape[0] < 2:
                return embeddings.sum() * 0.0
            normalized = F.normalize(embeddings, dim=-1)
            squared_distances = torch.pdist(normalized, p=2).square()
            # logsumexp avoids underflow for large t without changing the loss.
            return torch.logsumexp(-self.t * squared_distances, dim=0) - math.log(
                squared_distances.numel()
            )


class BYOLLoss(nn.Module):
    """Symmetric BYOL regression with an EMA center for target embeddings."""

    representation = "prediction"
    uses_teacher = True

    def __init__(self, embedding_dim: int, center_momentum: float = 0.9) -> None:
        super().__init__()
        if isinstance(embedding_dim, bool) or not isinstance(embedding_dim, int):
            raise ValueError("embedding_dim must be an integer")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        if not 0.0 <= center_momentum < 1.0:
            raise ValueError("center_momentum must be in [0, 1)")
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, embedding_dim))

    def forward(
        self,
        student_predictions: tuple[torch.Tensor, torch.Tensor],
        teacher_projections: tuple[torch.Tensor, torch.Tensor],
        *,
        compute_metrics: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        student1, student2 = student_predictions
        teacher1, teacher2 = (view.float().detach() for view in teacher_projections)
        cosine_12 = F.cosine_similarity(student1.float(), teacher2.float(), dim=-1)
        cosine_21 = F.cosine_similarity(student2.float(), teacher1.float(), dim=-1)
        loss = (2.0 - 2.0 * cosine_12.mean() + 2.0 - 2.0 * cosine_21.mean()) / 2.0

        if not compute_metrics:
            return loss, {}
        return loss, {
            "byol_cosine": torch.cat((cosine_12.detach(), cosine_21.detach())).mean(),
            "student_prediction_std": torch.cat((student1, student2))
            .detach()
            .float()
            .std(dim=0, unbiased=False)
            .mean(),
            "teacher_projection_std": torch.cat((teacher1, teacher2))
            .float()
            .std(dim=0, unbiased=False)
            .mean(),
            "center_norm": self.center.norm(),
        }

    @torch.no_grad()
    def update_center(self, teacher_embeddings: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Update the center from raw target embeddings after the current loss."""
        batch_center = torch.cat(teacher_embeddings).float().mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(
            batch_center, alpha=1.0 - self.center_momentum
        )


class InfoNCELoss(nn.Module):
    """Symmetric in-batch InfoNCE over two augmented sentence views."""

    representation = "embedding"
    uses_teacher = False

    def __init__(self, temperature: float = 0.05) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(
        self, views: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        view1, view2 = (F.normalize(view, dim=-1) for view in views)
        logits = view1 @ view2.T / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

        similarities = logits.detach() * self.temperature
        positive_cosine = similarities.diagonal().mean()
        if similarities.shape[0] > 1:
            negative_mask = ~torch.eye(
                similarities.shape[0], dtype=torch.bool, device=similarities.device
            )
            negative_cosine = similarities[negative_mask].mean()
        else:
            negative_cosine = similarities.new_tensor(float("nan"))
        accuracy = (
            (logits.argmax(dim=1) == labels).float().mean()
            + (logits.argmax(dim=0) == labels).float().mean()
        ) / 2
        return loss, {
            "positive_cosine": positive_cosine,
            "negative_cosine": negative_cosine,
            "contrastive_accuracy": accuracy,
        }


def build_objective(
    name: str,
    *,
    projection_dim: int,
    embedding_dim: int | None = None,
    center_momentum: float,
    infonce_temp: float,
) -> BYOLLoss | InfoNCELoss:
    """Construct an objective while keeping objective-specific settings local."""
    if name == "byol":
        return BYOLLoss(
            projection_dim if embedding_dim is None else embedding_dim,
            center_momentum,
        )
    if name == "infonce":
        return InfoNCELoss(infonce_temp)
    raise ValueError(f"Unknown objective: {name}")
