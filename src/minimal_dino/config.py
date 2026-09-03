from __future__ import annotations

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

    if augmentation["name"] not in {"dropout", "word"}:
        raise ValueError("augmentation.name must be 'dropout' or 'word'")
    if objective["name"] not in {"dino", "infonce"}:
        raise ValueError("objective.name must be 'dino' or 'infonce'")
    if model["pooling"] not in {"cls", "mean"}:
        raise ValueError("model.pooling must be 'cls' or 'mean'")
    if not isinstance(model["use_mlp"], bool):
        raise ValueError("model.use_mlp must be a boolean")
    if (
        isinstance(model["uniformity_step_size"], bool)
        or not isinstance(model["uniformity_step_size"], (int, float))
        or model["uniformity_step_size"] < 0
    ):
        raise ValueError("model.uniformity_step_size must be a non-negative number")

    device = runtime["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dino_precision = runtime.get("dino_precision", "bf16")
    if dino_precision not in {"fp32", "bf16"}:
        raise ValueError("runtime.dino_precision must be 'fp32' or 'bf16'")

    return SimpleNamespace(
        train_file=data["train_file"],
        max_length=data["max_length"],
        num_workers=data["num_workers"],
        model_name=model["name"],
        model_revision=model["revision"],
        random_init=model["random_init"],
        dropout=model["dropout"],
        pooling=model["pooling"],
        use_mlp=model["use_mlp"],
        output_dim=model["output_dim"],
        head_hidden_dim=model["head_hidden_dim"],
        bottleneck_dim=model["bottleneck_dim"],
        uniformity_step_size=model["uniformity_step_size"],
        augmentation=augmentation["name"],
        augmentation_strength=augmentation["strength"],
        objective=objective["name"],
        student_temp=objective.get("student_temp", 0.1),
        center_momentum=objective.get("center_momentum", 0.9),
        dino_reset_interval=objective.get("reset_interval"),
        infonce_temp=objective.get("temperature", 0.05),
        epochs=optimization["epochs"],
        max_steps=optimization["max_steps"],
        batch_size=optimization["batch_size"],
        learning_rate=optimization["learning_rate"],
        weight_decay=optimization["weight_decay"],
        warmup_ratio=optimization["warmup_ratio"],
        max_grad_norm=optimization["max_grad_norm"],
        teacher_temp=teacher["temperature"],
        warmup_teacher_temp=teacher["warmup_temperature"],
        teacher_temp_warmup_steps=teacher["temperature_warmup_steps"],
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
        dino_precision=dino_precision,
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
