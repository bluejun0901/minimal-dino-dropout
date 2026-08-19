from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DINOLoss(nn.Module):
    """Two-view DINO cross-entropy with teacher centering and sharpening."""

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
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if teacher_temp <= 0:
            raise ValueError("teacher_temp must be positive")

        student_log_probs = tuple(
            F.log_softmax(view / self.student_temp, dim=-1) for view in student_views
        )
        teacher_probs = tuple(
            F.softmax((view - self.center) / teacher_temp, dim=-1).detach()
            for view in teacher_views
        )

        # Match only opposite dropout views, as in DINO's two-global-view pseudocode.
        loss_12 = -(teacher_probs[0] * student_log_probs[1]).sum(dim=-1).mean()
        loss_21 = -(teacher_probs[1] * student_log_probs[0]).sum(dim=-1).mean()
        loss = (loss_12 + loss_21) / 2

        teacher_prob = torch.cat(teacher_probs)
        student_prob = torch.cat(tuple(log_prob.exp() for log_prob in student_log_probs))
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
        batch_center = torch.cat(teacher_views).mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1 - self.center_momentum)
