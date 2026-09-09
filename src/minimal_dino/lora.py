from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import nn


def lora_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    values = dict(enabled=False, r=8, alpha=16.0, target_modules=["query", "value"])
    if config is not None:
        unknown = set(config) - set(values)
        if unknown:
            raise ValueError(f"Unknown model.lora options: {sorted(unknown)}")
        values.update(config)
    if not isinstance(values["enabled"], bool):
        raise ValueError("model.lora.enabled must be a boolean")
    rank = values["r"]
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("model.lora.r must be a positive integer")
    alpha = values["alpha"]
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(alpha)
        or alpha <= 0
    ):
        raise ValueError("model.lora.alpha must be a finite positive number")
    targets = values["target_modules"]
    if (
        not isinstance(targets, list)
        or not targets
        or any(not isinstance(name, str) or not name for name in targets)
        or len(set(targets)) != len(targets)
    ):
        raise ValueError("model.lora.target_modules must be a non-empty list of unique names")
    values["target_modules"] = list(targets)
    return values


class LoRALinear(nn.Module):
    """Frozen linear layer with a zero-initialized low-rank weight update."""

    def __init__(self, base: nn.Linear, r: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.scale = alpha / r
        self.lora_A = nn.Parameter(base.weight.new_empty(r, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base(inputs) + ((inputs @ self.lora_A.T) @ self.lora_B.T) * self.scale


def apply_lora(encoder: nn.Module, config: Mapping[str, Any]) -> None:
    if getattr(encoder.config, "model_type", None) != "bert":
        raise ValueError("model.lora requires a BERT backbone")
    targets = config["target_modules"]
    matches = {target: [] for target in targets}
    for name, module in encoder.named_modules():
        if isinstance(module, nn.Linear):
            for target in targets:
                if name == target or name.endswith("." + target):
                    matches[target].append((name, module))
    missing = [target for target, modules in matches.items() if not modules]
    if missing:
        raise ValueError(f"LoRA targets did not match any BERT linear layers: {missing}")
    encoder.requires_grad_(False)
    for name, module in dict(item for modules in matches.values() for item in modules).items():
        parent_name, _, child_name = name.rpartition(".")
        parent = encoder.get_submodule(parent_name)
        setattr(parent, child_name, LoRALinear(module, config["r"], config["alpha"]))
