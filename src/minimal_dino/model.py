from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel


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
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_weight = nn.Parameter(torch.empty(output_dim, bottleneck_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.trunc_normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)
        nn.init.trunc_normal_(self.last_weight, std=0.02)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        bottleneck = F.normalize(self.mlp(embedding), dim=-1)
        weight = F.normalize(self.last_weight, dim=-1)
        return F.linear(bottleneck, weight)


@dataclass
class DINOOutput:
    embedding: torch.Tensor
    logits: torch.Tensor


class SentenceDINO(nn.Module):
    """Mean-pooled BERT encoder followed by a DINO projection head."""

    def __init__(
        self,
        encoder: nn.Module,
        output_dim: int = 65_536,
        head_hidden_dim: int = 2_048,
        bottleneck_dim: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        hidden_size = encoder.config.hidden_size
        self.head = DINOHead(hidden_size, output_dim, head_hidden_dim, bottleneck_dim)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "bert-base-uncased",
        *,
        revision: str | None = None,
        dropout: float | None = None,
        **head_kwargs: int,
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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        use_dropout: bool,
    ) -> DINOOutput:
        # Calling the same encoder twice gives independent masks. The context is needed
        # for the EMA teacher, which otherwise stays in eval mode and disables dropout.
        with dropout_mode(self.encoder, use_dropout):
            hidden = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            ).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        embedding = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return DINOOutput(embedding=embedding, logits=self.head(embedding))

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return the pre-projection masked-mean sentence representation without dropout."""
        return self(input_ids, attention_mask, use_dropout=False).embedding
