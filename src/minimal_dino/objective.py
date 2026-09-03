from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DINOLoss(nn.Module):
    """Two-view DINO cross-entropy with teacher centering and sharpening."""

    representation = "logits"
    uses_teacher = True

    def __init__(self, output_dim: int, student_temp: float = 0.1, center_momentum: float = 0.9):
        super().__init__()
        if student_temp <= 0:
            raise ValueError("student_temp must be positive")
        if not 0 <= center_momentum < 1:
            raise ValueError("center_momentum must be in [0, 1)")
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, output_dim))

    def forward(
        self,
        student_views: tuple[torch.Tensor, torch.Tensor],
        teacher_views: tuple[torch.Tensor, torch.Tensor],
        teacher_temp: float,
        *,
        compute_metrics: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if teacher_temp <= 0:
            raise ValueError("teacher_temp must be positive")

        # Keep the large-temperature-scaled softmaxes in FP32 even when the encoders and
        # projection heads run under BF16 autocast.
        student_log_probs = tuple(
            F.log_softmax(view.float() / self.student_temp, dim=-1) for view in student_views
        )
        if teacher_views[0] is teacher_views[1]:
            # Dropout augmentation gives the deterministic EMA teacher the same input
            # twice. Reuse its distribution instead of launching a duplicate softmax.
            teacher_prob = F.softmax(
                (teacher_views[0].float() - self.center) / teacher_temp, dim=-1
            ).detach()
            teacher_probs = (teacher_prob, teacher_prob)
        else:
            teacher_probs = tuple(
                F.softmax((view.float() - self.center) / teacher_temp, dim=-1).detach()
                for view in teacher_views
            )

        # Match only opposite dropout views, as in DINO's two-global-view pseudocode.
        loss_12 = -(teacher_probs[0] * student_log_probs[1]).sum(dim=-1).mean()
        loss_21 = -(teacher_probs[1] * student_log_probs[0]).sum(dim=-1).mean()
        loss = (loss_12 + loss_21) / 2

        if not compute_metrics:
            return loss, {}

        teacher_prob = torch.cat(teacher_probs)
        student_prob = torch.cat(tuple(log_prob.detach().exp() for log_prob in student_log_probs))
        eps = torch.finfo(teacher_prob.dtype).eps
        metrics = {
            "teacher_entropy": -(teacher_prob * teacher_prob.clamp_min(eps).log())
            .sum(dim=-1)
            .mean(),
            "teacher_batch_entropy": -(
                teacher_prob.mean(dim=0) * teacher_prob.mean(dim=0).clamp_min(eps).log()
            ).sum(),
            "student_entropy": -(student_prob * student_prob.clamp_min(eps).log())
            .sum(dim=-1)
            .mean(),
            "center_norm": self.center.norm(),
        }
        return loss, metrics

    @torch.no_grad()
    def update_center(self, teacher_views: tuple[torch.Tensor, torch.Tensor]) -> None:
        # Update after computing the loss, so the current batch uses the previous center.
        if teacher_views[0] is teacher_views[1]:
            batch_center = teacher_views[0].float().mean(dim=0, keepdim=True)
        else:
            batch_center = torch.cat(teacher_views).float().mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1 - self.center_momentum)




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
    output_dim: int,
    student_temp: float,
    center_momentum: float,
    infonce_temp: float,
) -> DINOLoss | InfoNCELoss:
    """Construct an objective while keeping objective-specific settings local."""
    if name == "dino":
        return DINOLoss(output_dim, student_temp, center_momentum)
    if name == "infonce":
        return InfoNCELoss(infonce_temp)
    raise ValueError(f"Unknown objective: {name}")
