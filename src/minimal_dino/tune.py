from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import optuna
import torch
from hydra import compose, initialize_config_module

from minimal_dino.config import to_train_args
from minimal_dino.train import train

MANAGED_OVERRIDE_KEYS = {
    "optimization.learning_rate",
    "optimization.batch_size",
    "optimization.weight_decay",
    "optimization.warmup_ratio",
    "model.dropout",
    "model.use_mlp",
    "model.uniformity_step_size",
    "objective.student_temp",
    "objective.reset_interval",
    "teacher.temperature",
    "objective.center_momentum",
    "teacher.momentum",
    "runtime.output_dir",
    "logging.tensorboard",
    "checkpoint.save_steps",
}
OPTUNA_TO_HYDRA = {
    "learning_rate": "optimization.learning_rate",
    "batch_size": "optimization.batch_size",
    "weight_decay": "optimization.weight_decay",
    "warmup_ratio": "optimization.warmup_ratio",
    "dropout": "model.dropout",
    "use_mlp": "model.use_mlp",
    "uniformity_step_size": "model.uniformity_step_size",
    "student_temp": "objective.student_temp",
    "reset_interval": "objective.reset_interval",
    "teacher_temperature": "teacher.temperature",
    "center_momentum": "objective.center_momentum",
    "teacher_momentum": "teacher.momentum",
}


def suggest_hyperparameters(trial: optuna.Trial) -> dict[str, Any]:
    """Return the focused DINO + dropout search space."""
    return {
        "optimization.learning_rate": trial.suggest_float(
            "learning_rate", 1.0e-5, 5.0e-5, log=True
        ),
        "optimization.batch_size": trial.suggest_categorical("batch_size", [32, 64]),
        "optimization.weight_decay": trial.suggest_categorical(
            "weight_decay", [0.0, 0.01, 0.05, 0.1]
        ),
        "optimization.warmup_ratio": trial.suggest_categorical(
            "warmup_ratio", [0.0, 0.05, 0.1, 0.2]
        ),
        "model.dropout": trial.suggest_float("dropout", 0.05, 0.3, step=0.05),
        "model.use_mlp": trial.suggest_categorical("use_mlp", [True, False]),
        "model.uniformity_step_size": trial.suggest_categorical(
            "uniformity_step_size", [0.0, 0.001, 0.01, 0.05, 0.1, 0.5]
        ),
        "objective.student_temp": trial.suggest_float(
            "student_temp", 0.05, 0.2, step=0.05
        ),
        "objective.reset_interval": trial.suggest_categorical(
            "reset_interval", [None, 150, 300, 600]
        ),
        "teacher.temperature": trial.suggest_float(
            "teacher_temperature", 0.01, 0.1
        ),
        "objective.center_momentum": trial.suggest_categorical(
            "center_momentum", [0.5, 0.9, 0.99, 0.999]
        ),
        "teacher.momentum": trial.suggest_categorical(
            "teacher_momentum", [0.99, 0.996, 0.999, 0.9999]
        ),
    }


def validate_base_overrides(overrides: list[str]) -> None:
    """Prevent callers from changing the experiment family or trial-owned values."""
    for override in overrides:
        key = override.lstrip("+").split("=", 1)[0]
        if key in {"objective", "objective.name", "augmentation", "augmentation.name"}:
            raise ValueError("objective and augmentation are fixed to DINO + dropout")
        if key in MANAGED_OVERRIDE_KEYS:
            raise ValueError(f"Override is managed by the tuner: {key}")


def parameter_overrides(parameters: dict[str, Any]) -> list[str]:
    def hydra_value(value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return str(value).lower()
        return str(value)

    return [f"{key}={hydra_value(value)}" for key, value in parameters.items()]


def run_trial(
    trial: optuna.Trial,
    *,
    output_root: Path,
    base_overrides: list[str],
) -> float:
    parameters = suggest_hyperparameters(trial)
    trial_dir = output_root / f"trial-{trial.number:04d}"
    if trial_dir.exists():
        raise FileExistsError(f"Trial output already exists: {trial_dir}")

    overrides = [
        "objective=dino",
        "augmentation=dropout",
        *base_overrides,
        *parameter_overrides(parameters),
        f"runtime.output_dir={trial_dir}",
        "logging.tensorboard=false",
        "checkpoint.save_steps=0",
    ]
    with initialize_config_module(version_base="1.3", config_module="minimal_dino.conf"):
        config = compose(config_name="config", overrides=overrides)
    args = to_train_args(config)

    observed_scores: list[tuple[int, float]] = []

    def report_evaluation(step: int, metrics: dict[str, float]) -> None:
        score = float(metrics["max_sts_spearman"])
        observed_scores.append((step, score))
        if step == 0:
            return
        trial.report(score, step=step)
        if trial.should_prune():
            raise optuna.TrialPruned(
                f"Pruned at step {step} with max STS-B Spearman {score:.6f}"
            )

    trial.set_user_attr("output_dir", str(trial_dir))
    try:
        train(
            args,
            run_config=config,
            evaluation_callback=report_evaluation,
            save_final_checkpoint=False,
        )
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    positive_step_scores = [score for step, score in observed_scores if step > 0]
    if not positive_step_scores:
        raise RuntimeError(
            "No post-training evaluation was produced. Set evaluation.steps no larger than "
            "optimization.max_steps (or the number of steps in one epoch)."
        )
    return positive_step_scores[-1]


def default_storage(output_root: Path, study_name: str) -> str:
    database = (output_root / f"{study_name}.db").resolve()
    return f"sqlite:///{database}"


def save_best_trial(study: optuna.Study, output_root: Path) -> Path:
    hydra_parameters = {
        OPTUNA_TO_HYDRA[name]: value for name, value in study.best_params.items()
    }
    result = {
        "study_name": study.study_name,
        "trial_number": study.best_trial.number,
        "value": study.best_value,
        "parameters": study.best_params,
        "hydra_overrides": parameter_overrides(hydra_parameters),
        "output_dir": study.best_trial.user_attrs.get("output_dir"),
    }
    path = output_root / "best_trial.json"
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(path)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune the DINO + dropout configuration")
    parser.add_argument("overrides", nargs="*", help="Fixed Hydra overrides for every trial")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--timeout", type=int, help="Overall timeout in seconds")
    parser.add_argument("--study-name", default="dino-dropout")
    parser.add_argument("--output-root", type=Path, default=Path("runs/optuna/dino-dropout"))
    parser.add_argument("--storage", help="Optuna storage URL; defaults to SQLite in output-root")
    parser.add_argument("--sampler-seed", type=int, default=42)
    parser.add_argument("--pruner-startup-trials", type=int, default=5)
    parser.add_argument("--pruner-warmup-steps", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    if cli.n_trials < 1:
        raise ValueError("--n-trials must be positive")
    validate_base_overrides(cli.overrides)
    cli.output_root.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=cli.study_name,
        storage=cli.storage or default_storage(cli.output_root, cli.study_name),
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=cli.sampler_seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=cli.pruner_startup_trials,
            n_warmup_steps=cli.pruner_warmup_steps,
        ),
    )
    study.optimize(
        lambda trial: run_trial(
            trial,
            output_root=cli.output_root,
            base_overrides=cli.overrides,
        ),
        n_trials=cli.n_trials,
        timeout=cli.timeout,
        gc_after_trial=True,
        catch=(torch.cuda.OutOfMemoryError,),
    )
    result_path = save_best_trial(study, cli.output_root)
    print(f"Best STS-B Spearman: {study.best_value:.6f}")
    print(f"Best parameters: {json.dumps(study.best_params, sort_keys=True)}")
    print(f"Saved summary to {result_path}")


if __name__ == "__main__":
    main()
