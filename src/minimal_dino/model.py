from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel
from transformers.utils import logging as transformers_logging
transformers_logging.set_verbosity_error()

@contextmanager
def dropout_mode(module: nn.Module, enabled: bool):
    """Temporarily control only dropout modules, independent of model train/eval mode."""
    dropouts = [child for child in module.modules() if isinstance(child, nn.Dropout)]
    previous = [child.training for child in dropouts]
    for child in dropouts:
        child.train(enabled)
    try:
        yield
    finally:
        for child, was_training in zip(dropouts, previous):
            child.train(was_training)


class UniformityLoss(nn.Module):
    def __init__(self, t: float = 2.0):
        super().__init__()
        if t <= 0:
            raise ValueError("t must be positive")
        self.t = t

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.shape[0] < 2:
            return embeddings.sum() * 0.0
        embeddings = F.normalize(embeddings, dim=-1)
        similarity_matrix = embeddings @ embeddings.T
        # Exclude diagonal elements (self-similarity) from the loss.
        mask = ~torch.eye(
            similarity_matrix.shape[0],
            dtype=torch.bool,
            device=similarity_matrix.device,
        )
        similarity_matrix = similarity_matrix[mask].view(similarity_matrix.shape[0], -1)
        distance_square_matrix = 2 - 2 * similarity_matrix
        return torch.logsumexp(-distance_square_matrix * self.t, dim=-1).mean()


class DINOHead(nn.Module):
    """The BN-free three-layer projection head used by DINO.

    The final linear layer uses unit-normalized weight vectors. This is equivalent to
    DINO's weight-normalized layer with its scale fixed to one (``norm_last_layer=True``).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 65_536,
        hidden_dim: int = 2_048,
        bottleneck_dim: int = 256,
        use_mlp: bool = True,
        uniformity_step_size: float = 0.01,
    ) -> None:
        super().__init__()
        if not isinstance(use_mlp, bool):
            raise ValueError("use_mlp must be a boolean")
        if uniformity_step_size < 0:
            raise ValueError("uniformity_step_size must be non-negative")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.use_mlp = use_mlp
        self.uniformity_step_size = uniformity_step_size
        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, bottleneck_dim),
            )
            projection_dim = bottleneck_dim
        else:
            self.mlp = nn.Identity()
            projection_dim = input_dim
        self.last_weight = nn.Parameter(torch.empty(output_dim, projection_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.mlp.modules():
            if isinstance(layer, nn.Linear):
                nn.init.trunc_normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)
        nn.init.trunc_normal_(self.last_weight, std=0.02)

    def uniformity_grad(self, z: torch.Tensor, t: float = 2.0) -> torch.Tensor:
        if z.shape[0] < 2:
            return torch.zeros_like(z)
        diff = z[:, None, :] - z[None, :, :]
        dist2 = diff.square().sum(dim=-1)
        weight = torch.exp(-t * dist2)
        mask = ~torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
        weight = weight * mask
        denominator = weight.sum() / 2
        grad = -2 * t * (weight[..., None] * diff).sum(dim=1) / denominator.clamp_min(
            torch.finfo(z.dtype).tiny
        )
        return grad

    def uniformize_embedding(
        self, embedding: torch.Tensor, step_size: float
    ) -> torch.Tensor:
        normalized_embedding = F.normalize(embedding, dim=-1)
        embedding_norm = embedding.norm(dim=-1, keepdim=True).clamp_min(
            torch.finfo(normalized_embedding.dtype).tiny
        )
        grad = self.uniformity_grad(normalized_embedding, t=2.0)
        return F.normalize(normalized_embedding - step_size * grad, dim=-1) * embedding_norm

    def forward(self, embedding: torch.Tensor, is_teacher: bool = False) -> torch.Tensor:
        projection = F.normalize(self.mlp(embedding), dim=-1)
        weight = F.normalize(self.last_weight, dim=-1)
        if not self.training and is_teacher and self.uniformity_step_size > 0:
            projection = self.uniformize_embedding(
                projection.float(), step_size=self.uniformity_step_size
            ).to(projection.dtype)

        return F.linear(projection, weight)


@dataclass
class DINOOutput:
    embedding: torch.Tensor
    logits: torch.Tensor


class SentenceDINO(nn.Module):
    """Configurable BERT sentence pooling followed by a DINO projection head."""

    def __init__(
        self,
        encoder: nn.Module,
        output_dim: int = 65_536,
        head_hidden_dim: int = 2_048,
        bottleneck_dim: int = 256,
        pooling: str = "mean",
        use_mlp: bool = True,
        uniformity_step_size: float = 0.01,
    ) -> None:
        super().__init__()
        if pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be 'cls' or 'mean'")
        self.encoder = encoder
        self.pooling = pooling
        self.use_mlp = use_mlp
        hidden_size = encoder.config.hidden_size
        self.head = DINOHead(
            hidden_size,
            output_dim,
            head_hidden_dim,
            bottleneck_dim,
            use_mlp=use_mlp,
            uniformity_step_size=uniformity_step_size,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "bert-base-uncased",
        *,
        revision: str | None = None,
        dropout: float | None = None,
        **head_kwargs: Any,
    ) -> SentenceDINO:
        model_kwargs = {}
        if dropout is not None:
            if not 0.0 <= dropout < 1.0:
                raise ValueError("dropout must be in [0, 1)")
            # BERT uses separate dropout probabilities for hidden states and attention.
            # Keep them tied so one sweep value describes the complete encoder setup.
            model_kwargs.update(
                hidden_dropout_prob=dropout,
                attention_probs_dropout_prob=dropout,
            )
        encoder = AutoModel.from_pretrained(model_name, revision=revision, **model_kwargs)
        return cls(encoder, **head_kwargs)

    @classmethod
    def from_random_init(
        cls,
        model_name: str = "bert-base-uncased",
        *,
        revision: str | None = None,
        dropout: float | None = None,
        **head_kwargs: Any,
    ) -> SentenceDINO:
        """Build the requested encoder architecture without loading pretrained weights."""
        config_kwargs = {}
        if dropout is not None:
            if not 0.0 <= dropout < 1.0:
                raise ValueError("dropout must be in [0, 1)")
            config_kwargs.update(
                hidden_dropout_prob=dropout,
                attention_probs_dropout_prob=dropout,
            )
        config = AutoConfig.from_pretrained(model_name, revision=revision, **config_kwargs)
        encoder = AutoModel.from_config(config)
        return cls(encoder, **head_kwargs)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        use_dropout: bool,
        is_teacher: bool = False,
    ) -> DINOOutput:
        # Calling the same encoder twice gives independent masks. The context is needed
        # for the EMA teacher, which otherwise stays in eval mode and disables dropout.
        with dropout_mode(self.encoder, use_dropout):
            hidden = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            ).last_hidden_state
        if self.pooling == "cls":
            embedding = hidden[:, 0, :]
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            embedding = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        logits = self.head(embedding, is_teacher=is_teacher)
        return DINOOutput(embedding=embedding, logits=logits)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return the configured pre-projection sentence representation without dropout."""
        return self(input_ids, attention_mask, use_dropout=False, is_teacher=False).embedding


def model_config(model: SentenceDINO) -> dict[str, Any]:
    """Return the architecture choices needed to reconstruct a sentence model."""
    return {
        "output_dim": model.head.output_dim,
        "head_hidden_dim": model.head.hidden_dim,
        "bottleneck_dim": model.head.bottleneck_dim,
        "pooling": model.pooling,
        "use_mlp": model.use_mlp,
        "uniformity_step_size": model.head.uniformity_step_size,
    }


def checkpoint_model_config(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Read model choices, including checkpoints created before explicit switches."""
    config = dict(checkpoint["head_config"])
    student_state = checkpoint["student"]
    config.setdefault("pooling", checkpoint.get("pooling", "mean"))
    config.setdefault(
        "use_mlp", any(name.startswith("head.mlp.") for name in student_state)
    )
    config.setdefault("head_hidden_dim", 2_048)
    config.setdefault("bottleneck_dim", 256)
    config.setdefault("uniformity_step_size", 0.01)
    return config
