from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn
from transformers import AutoConfig, AutoModel
from transformers.utils import logging as transformers_logging

transformers_logging.set_verbosity_error()


@contextmanager
def dropout_mode(
    module: nn.Module,
    enabled: bool,
    probability: float | None = None,
):
    """Temporarily control dropout state and probability independently of model mode."""
    if probability is not None and not 0.0 <= probability < 1.0:
        raise ValueError("dropout probability must be in [0, 1)")
    dropouts = [child for child in module.modules() if isinstance(child, nn.Dropout)]
    previous = [(child.training, child.p) for child in dropouts]
    for child in dropouts:
        child.train(enabled)
        if probability is not None:
            child.p = probability
    try:
        yield
    finally:
        for child, (was_training, previous_probability) in zip(dropouts, previous):
            child.train(was_training)
            child.p = previous_probability


class BYOLHead(nn.Module):
    """BYOL projector and online predictor.

    LayerNorm replaces BYOL's batch normalization so sentence training also supports a
    singleton final minibatch. The target branch uses only :meth:`project`; its predictor
    parameters are never part of the regression target.
    """

    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 256,
        projector_hidden_dim: int = 4_096,
        predictor_hidden_dim: int = 4_096,
    ) -> None:
        super().__init__()
        for name, value in {
            "projection_dim": projection_dim,
            "projector_hidden_dim": projector_hidden_dim,
            "predictor_hidden_dim": predictor_hidden_dim,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.projector_hidden_dim = projector_hidden_dim
        self.predictor_hidden_dim = predictor_hidden_dim
        self.projector = self._mlp(input_dim, projector_hidden_dim, projection_dim)
        self.predictor = self._mlp(projection_dim, predictor_hidden_dim, projection_dim)
        self.reset_parameters()

    @staticmethod
    def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def project(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.projector(embedding)

    def predict(self, projection: torch.Tensor) -> torch.Tensor:
        return self.predictor(projection)


@dataclass
class BYOLOutput:
    embedding: torch.Tensor
    projection: torch.Tensor
    prediction: torch.Tensor | None


class SentenceBYOL(nn.Module):
    """Sentence encoder followed by a BYOL projector and online predictor."""

    def __init__(
        self,
        encoder: nn.Module,
        projection_dim: int = 256,
        projector_hidden_dim: int = 4_096,
        predictor_hidden_dim: int = 4_096,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        if pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be 'cls' or 'mean'")
        self.encoder = encoder
        self.pooling = pooling
        self.head = BYOLHead(
            encoder.config.hidden_size,
            projection_dim=projection_dim,
            projector_hidden_dim=projector_hidden_dim,
            predictor_hidden_dim=predictor_hidden_dim,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "bert-base-uncased",
        *,
        revision: str | None = None,
        dropout: float | None = None,
        **head_kwargs: Any,
    ) -> SentenceBYOL:
        encoder = AutoModel.from_pretrained(
            model_name, revision=revision, **cls._dropout_kwargs(dropout)
        )
        return cls(encoder, **head_kwargs)

    @classmethod
    def from_random_init(
        cls,
        model_name: str = "bert-base-uncased",
        *,
        revision: str | None = None,
        dropout: float | None = None,
        **head_kwargs: Any,
    ) -> SentenceBYOL:
        """Build the requested encoder architecture without loading pretrained weights."""
        config = AutoConfig.from_pretrained(
            model_name, revision=revision, **cls._dropout_kwargs(dropout)
        )
        return cls(AutoModel.from_config(config), **head_kwargs)

    @staticmethod
    def _dropout_kwargs(dropout: float | None) -> dict[str, float]:
        if dropout is None:
            return {}
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        return {
            "hidden_dropout_prob": dropout,
            "attention_probs_dropout_prob": dropout,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        use_dropout: bool,
        dropout_probability: float | None = None,
        target: bool = False,
        center: torch.Tensor | None = None,
        center_scale: float = 0.5,
    ) -> BYOLOutput:
        with dropout_mode(self.encoder, use_dropout, dropout_probability):
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
        if center is not None:
            if not target:
                raise ValueError("center can only be applied to the target branch")
            if center.shape != (1, self.head.input_dim):
                raise ValueError(
                    f"center must have shape (1, {self.head.input_dim}), got {tuple(center.shape)}"
                )
            projection_input = embedding.float() - center * center_scale
        else:
            projection_input = embedding
        projection = self.head.project(projection_input)
        prediction = None if target else self.head.predict(projection)
        return BYOLOutput(embedding, projection, prediction)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return the configured pre-projection sentence representation without dropout."""
        return self(input_ids, attention_mask, use_dropout=False, target=True).embedding


def model_config(model: SentenceBYOL) -> dict[str, Any]:
    """Return the architecture choices needed to reconstruct a sentence model."""
    return {
        "projection_dim": model.head.projection_dim,
        "projector_hidden_dim": model.head.projector_hidden_dim,
        "predictor_hidden_dim": model.head.predictor_hidden_dim,
        "pooling": model.pooling,
    }


def checkpoint_model_config(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Read the BYOL model configuration from a checkpoint."""
    return dict(checkpoint["head_config"])
