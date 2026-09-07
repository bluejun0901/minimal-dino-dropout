from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from minimal_dino.geometry import GeometryConfig, PairedTargetGeometry


class BYOLLoss(nn.Module):
    """Symmetric BYOL regression with an EMA center for target embeddings."""

    representation = "prediction"
    uses_teacher = True

    def __init__(
        self,
        embedding_dim: int,
        center_momentum: float = 0.9,
        *,
        target_geometry: dict | None = None,
    ) -> None:
        super().__init__()
        if isinstance(embedding_dim, bool) or not isinstance(embedding_dim, int):
            raise ValueError("embedding_dim must be an integer")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        if not 0.0 <= center_momentum < 1.0:
            raise ValueError("center_momentum must be in [0, 1)")
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, embedding_dim))
        self.geometry = (
            PairedTargetGeometry(embedding_dim, GeometryConfig(**target_geometry))
            if target_geometry is not None
            else None
        )

    def transform_target(self, embedding: torch.Tensor, center_scale: float) -> torch.Tensor:
        centered = embedding.float() - self.center * center_scale
        if self.geometry is None or self.geometry.config.strength == 0:
            return centered
        return centered + self.geometry(embedding)

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
            **(self.geometry.metrics() if self.geometry is not None else {}),
        }

    @torch.no_grad()
    def update_center(self, teacher_embeddings: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Update the center from raw target embeddings after the current loss."""
        batch_center = torch.cat(teacher_embeddings).float().mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(
            batch_center, alpha=1.0 - self.center_momentum
        )
        if self.geometry is not None:
            self.geometry.update(teacher_embeddings)


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
    target_geometry: dict | None = None,
) -> BYOLLoss | InfoNCELoss:
    """Construct an objective while keeping objective-specific settings local."""
    if name == "byol":
        return BYOLLoss(
            projection_dim if embedding_dim is None else embedding_dim,
            center_momentum,
            target_geometry=target_geometry,
        )
    if name == "infonce":
        return InfoNCELoss(infonce_temp)
    raise ValueError(f"Unknown objective: {name}")
