from __future__ import annotations

import gc
import math
from pathlib import Path
from typing import Any

import hydra
import optuna
import torch
from omegaconf import DictConfig, OmegaConf

from minimal_dino.config import to_train_args
from minimal_dino.train import train


def sample_config(config: DictConfig, trial: optuna.Trial) -> DictConfig:
    """Return a per-trial training config with the requested parameters sampled."""
    trial_config = OmegaConf.create(OmegaConf.to_container(config, resolve=False))

    trial_config.optimization.encoder_learning_rate = trial.suggest_float(
        "encoder_learning_rate", 2e-7, 5e-6, log=True
    )
    trial_config.optimization.head_learning_rate = trial.suggest_float(
        "head_learning_rate", 3e-5, 3e-4, log=True
    )
    trial_config.model.dropout = trial.suggest_float("dropout", 0.12, 0.35)
    trial_config.objective.center_scale = trial.suggest_categorical(
        "center_scale", [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]
    )
    trial_config.objective.center_momentum = trial.suggest_float(
        "center_momentum", 0.80, 0.99
    )
    teacher_one_minus_momentum = trial.suggest_float(
        "teacher_one_minus_momentum", 1e-4, 1e-2, log=True
    )
    trial_config.teacher.momentum = 1.0 - teacher_one_minus_momentum
    trial_config.optimization.encoder_freeze_steps = trial.suggest_categorical(
        "encoder_freeze_steps", [100, 150, 200, 300, 400]
    )
    trial_config.optimization.warmup_ratio = trial.suggest_float(
        "warmup_ratio", 0.03, 0.15
    )
    return trial_config


def run_trial(config: DictConfig, trial: optuna.Trial) -> float:
    """Train one configuration and return its best validation metric."""
    trial_config = sample_config(config, trial)
    base_output_dir = Path(config.runtime.output_dir)
    trial_output_dir = base_output_dir / f"trial-{trial.number:04d}"
    trial_config.runtime.output_dir = str(trial_output_dir)
    trial_config.checkpoint.resume_from = None
    trial_config.checkpoint.save_steps = 0
    trial_config.logging.tensorboard = False
    trial_config.logging.quiet = True

    metric_name = str(config.tuning.metric)
    best_value = -math.inf
    evaluation_count = 0

    def report_evaluation(step: int, metrics: dict[str, float]) -> None:
        nonlocal best_value, evaluation_count
        # Step zero is independent of sampled hyperparameters and is not useful for pruning.
        if step == 0:
            return
        if metric_name not in metrics:
            raise KeyError(f"Evaluation did not produce tuning metric {metric_name!r}")
        value = float(metrics[metric_name])
        best_value = max(best_value, value)
        evaluation_count += 1
        trial.report(value, step)
        if trial.should_prune():
            raise optuna.TrialPruned(f"pruned at training step {step}")

    trial.set_user_attr("output_dir", str(trial_output_dir))
    try:
        train(
            to_train_args(trial_config),
            run_config=trial_config,
            evaluation_callback=report_evaluation,
            save_final_checkpoint=False,
        )
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if evaluation_count == 0:
        raise ValueError(
            "No post-training evaluation ran; set evaluation.steps <= optimization.max_steps"
        )
    return best_value


def create_study(config: DictConfig) -> optuna.Study:
    tuning: dict[str, Any] = OmegaConf.to_container(
        config.tuning, resolve=True, throw_on_missing=True
    )
    sampler = optuna.samplers.TPESampler(seed=int(tuning["sampler_seed"]))
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=int(tuning["pruner_startup_trials"]),
        n_warmup_steps=int(tuning["pruner_warmup_steps"]),
    )
    return optuna.create_study(
        study_name=str(tuning["study_name"]),
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=tuning["storage"],
        load_if_exists=True,
    )


@hydra.main(version_base="1.3", config_path="conf", config_name="tune")
def main(config: DictConfig) -> None:
    output_dir = Path(config.runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    study = create_study(config)
    study.optimize(
        lambda trial: run_trial(config, trial),
        n_trials=int(config.tuning.n_trials),
        timeout=config.tuning.timeout,
    )
    print(f"best value: {study.best_value:.6f}")
    print(f"best params: {study.best_params}")
    print(f"best trial output: {study.best_trial.user_attrs['output_dir']}")


if __name__ == "__main__":
    main()
