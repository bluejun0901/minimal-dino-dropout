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
    """Return a per-trial training config with the five requested parameters sampled."""
    trial_config = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    search = config.tuning.search

    dropout_low, dropout_high = search.model_dropout
    # ``to_train_args`` requires target dropout to be strictly lower than model dropout.
    dropout_low = max(float(dropout_low), float(config.teacher.dropout) + 1e-6)
    if dropout_low > dropout_high:
        raise ValueError("model_dropout search range must include values above teacher.dropout")

    trial_config.objective.center_scale = trial.suggest_float(
        "center_scale", *search.center_scale
    )
    trial_config.model.dropout = trial.suggest_float(
        "model_dropout", dropout_low, float(dropout_high)
    )
    trial_config.objective.center_momentum = trial.suggest_float(
        "center_momentum", *search.center_momentum
    )
    trial_config.optimization.encoder_learning_rate = trial.suggest_float(
        "encoder_learning_rate", *search.encoder_learning_rate, log=True
    )
    trial_config.optimization.head_learning_rate = trial.suggest_float(
        "head_learning_rate", *search.head_learning_rate, log=True
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
