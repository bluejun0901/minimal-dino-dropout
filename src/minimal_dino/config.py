from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf


def to_train_args(config: DictConfig) -> SimpleNamespace:
    """Translate the grouped Hydra config at the training-loop boundary."""
    values = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    if not isinstance(values, dict):
        raise TypeError("The composed training config must be a mapping")

    data = _section(values, "data")
    model = _section(values, "model")
    augmentation = _section(values, "augmentation")
    objective = _section(values, "objective")
    optimization = _section(values, "optimization")
    teacher = _section(values, "teacher")
    evaluation = _section(values, "evaluation")
    checkpoint = _section(values, "checkpoint")
    runtime = _section(values, "runtime")
    logging = _section(values, "logging")

    augmentations = augmentation["names"]
    if (
        not isinstance(augmentations, list)
        or not augmentations
        or any(
            not isinstance(name, str) or name not in {"dropout", "word"}
            for name in augmentations
        )
        or len(set(augmentations)) != len(augmentations)
    ):
        raise ValueError(
            "augmentation.names must be a non-empty list containing unique "
            "'dropout' and/or 'word' values"
        )
    if objective["name"] not in {"byol", "infonce"}:
        raise ValueError("objective.name must be 'byol' or 'infonce'")
    if model["pooling"] not in {"cls", "mean"}:
        raise ValueError("model.pooling must be 'cls' or 'mean'")
    for name in ("projection_dim", "projector_hidden_dim", "predictor_hidden_dim"):
        value = model[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"model.{name} must be a positive integer")
    encoder_freeze_steps = optimization["encoder_freeze_steps"]
    if (
        isinstance(encoder_freeze_steps, bool)
        or not isinstance(encoder_freeze_steps, int)
        or encoder_freeze_steps < 0
    ):
        raise ValueError("optimization.encoder_freeze_steps must be a non-negative integer")
    for name in ("encoder_learning_rate", "head_learning_rate"):
        value = optimization[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"optimization.{name} must be positive")
    target_dropout = teacher["dropout"]
    if not isinstance(target_dropout, (int, float)) or isinstance(target_dropout, bool):
        raise ValueError("teacher.dropout must be a number")
    if not 0.0 < target_dropout < model["dropout"]:
        raise ValueError("teacher.dropout must be positive and lower than model.dropout")
    center_momentum = objective.get("center_momentum", 0.9)
    if (
        isinstance(center_momentum, bool)
        or not isinstance(center_momentum, (int, float))
        or not 0.0 <= center_momentum < 1.0
    ):
        raise ValueError("objective.center_momentum must be in [0, 1)")
    center_scale = objective.get("center_scale", 0.5)
    center_scale_start = objective.get("center_scale_start")
    if center_scale_start is None:
        center_scale_start = center_scale
    center_scale_end = objective.get("center_scale_end")
    if center_scale_end is None:
        center_scale_end = center_scale_start
    for name, value in (
        ("center_scale", center_scale),
        ("center_scale_start", center_scale_start),
        ("center_scale_end", center_scale_end),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0.0
        ):
            raise ValueError(f"objective.{name} must be a finite non-negative number")

    uniformity_weight = objective.get("uniformity_weight", 0.0)
    uniformity_t = objective.get("uniformity_t", 2.0)
    for name, value, allow_zero in (
        ("uniformity_weight", uniformity_weight, True),
        ("uniformity_t", uniformity_t, False),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or (value < 0 if allow_zero else value <= 0)
        ):
            bound = "non-negative" if allow_zero else "positive"
            raise ValueError(f"objective.{name} must be a finite {bound} number")

    device = runtime["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    byol_precision = runtime["byol_precision"]
    if byol_precision not in {"fp32", "bf16"}:
        raise ValueError("runtime.byol_precision must be 'fp32' or 'bf16'")

    return SimpleNamespace(
        train_file=data["train_file"],
        max_length=data["max_length"],
        num_workers=data["num_workers"],
        model_name=model["name"],
        model_revision=model["revision"],
        random_init=model["random_init"],
        dropout=model["dropout"],
        pooling=model["pooling"],
        projection_dim=model["projection_dim"],
        projector_hidden_dim=model["projector_hidden_dim"],
        predictor_hidden_dim=model["predictor_hidden_dim"],
        augmentation=augmentations,
        augmentation_strength=augmentation["strength"],
        objective=objective["name"],
        center_momentum=center_momentum,
        center_scale=center_scale,
        center_scale_start=center_scale_start,
        center_scale_end=center_scale_end,
        infonce_temp=objective.get("temperature", 0.05),
        uniformity_weight=uniformity_weight,
        uniformity_t=uniformity_t,
        epochs=optimization["epochs"],
        max_steps=optimization.get("max_steps"),
        batch_size=optimization["batch_size"],
        encoder_learning_rate=optimization["encoder_learning_rate"],
        head_learning_rate=optimization["head_learning_rate"],
        encoder_freeze_steps=encoder_freeze_steps,
        weight_decay=optimization["weight_decay"],
        warmup_ratio=optimization["warmup_ratio"],
        max_grad_norm=optimization["max_grad_norm"],
        target_dropout=target_dropout,
        teacher_momentum=teacher["momentum"],
        eval_steps=evaluation["steps"],
        eval_batch_size=evaluation["batch_size"],
        eval_limit=evaluation["limit"],
        stsb_dir=evaluation["stsb_dir"],
        collapse_warning_after=evaluation["collapse_warning_after"],
        resume_from_checkpoint=checkpoint["resume_from"],
        save_steps=checkpoint["save_steps"],
        keep_last_checkpoints=checkpoint["keep_last"],
        output_dir=runtime["output_dir"],
        seed=runtime["seed"],
        device=device,
        byol_precision=byol_precision,
        log_steps=logging["steps"],
        quiet=logging["quiet"],
        tensorboard=logging["tensorboard"],
    )


def config_to_container(config: Any) -> Any:
    """Return plain serializable values for run artifacts."""
    if isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    if isinstance(config, SimpleNamespace) or hasattr(config, "__dict__"):
        return vars(config)
    return config


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    section = config[name]
    if not isinstance(section, dict):
        raise TypeError(f"Config section '{name}' must be a mapping")
    return section
