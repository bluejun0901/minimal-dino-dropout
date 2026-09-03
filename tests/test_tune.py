import json

import optuna
import pytest

from minimal_dino.tune import (
    default_storage,
    parameter_overrides,
    run_trial,
    save_best_trial,
    suggest_hyperparameters,
    validate_base_overrides,
)


def fixed_trial():
    return optuna.trial.FixedTrial(
        {
            "learning_rate": 3.0e-5,
            "batch_size": 64,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "dropout": 0.1,
            "use_mlp": True,
            "uniformity_step_size": 0.01,
            "student_temp": 0.1,
            "reset_interval": None,
            "teacher_temperature": 0.04,
            "center_momentum": 0.9,
            "teacher_momentum": 0.996,
        }
    )


def test_search_space_maps_optuna_names_to_hydra_keys():
    parameters = suggest_hyperparameters(fixed_trial())

    assert parameters["optimization.learning_rate"] == 3.0e-5
    assert parameters["model.dropout"] == 0.1
    assert parameters["model.use_mlp"] is True
    assert parameters["model.uniformity_step_size"] == 0.01
    assert parameters["objective.reset_interval"] is None
    assert parameters["teacher.temperature"] == 0.04
    assert len(parameters) == 12
    assert "optimization.learning_rate=3e-05" in parameter_overrides(parameters)
    assert "objective.reset_interval=null" in parameter_overrides(parameters)
    assert "model.use_mlp=true" in parameter_overrides(parameters)


@pytest.mark.parametrize(
    "override",
    [
        "objective=infonce",
        "objective.name=infonce",
        "augmentation=word",
        "model.dropout=0.4",
        "model.use_mlp=false",
        "model.uniformity_step_size=0.1",
        "objective.reset_interval=300",
        "runtime.output_dir=elsewhere",
        "checkpoint.save_steps=10",
    ],
)
def test_tuner_rejects_fixed_or_managed_overrides(override):
    with pytest.raises(ValueError):
        validate_base_overrides([override])


def test_tuner_accepts_fixed_training_budget_overrides():
    validate_base_overrides(
        [
            "data.train_file=train.txt",
            "optimization.max_steps=500",
            "evaluation.steps=100",
            "model.pooling=cls",
        ]
    )


def test_run_trial_composes_dino_dropout_and_returns_latest_score(tmp_path, monkeypatch):
    captured = {}

    def fake_train(
        args, *, run_config, evaluation_callback, save_final_checkpoint
    ):
        captured["args"] = args
        captured["config"] = run_config
        captured["save_final_checkpoint"] = save_final_checkpoint
        evaluation_callback(0, {"sts_spearman": 0.2, "max_sts_spearman": 0.2})
        evaluation_callback(100, {"sts_spearman": 0.6, "max_sts_spearman": 0.6})
        evaluation_callback(200, {"sts_spearman": 0.5, "max_sts_spearman": 0.6})

    monkeypatch.setattr("minimal_dino.tune.train", fake_train)

    score = run_trial(
        fixed_trial(),
        output_root=tmp_path,
        base_overrides=["data.train_file=train.txt", "runtime.device=cpu"],
    )

    assert score == 0.6
    assert captured["args"].objective == "dino"
    assert captured["args"].augmentation == "dropout"
    assert captured["args"].pooling == "mean"
    assert captured["args"].use_mlp is True
    assert captured["args"].uniformity_step_size == 0.01
    assert captured["args"].dino_reset_interval is None
    assert captured["save_final_checkpoint"] is False


def test_default_storage_is_an_absolute_sqlite_url(tmp_path):
    assert default_storage(tmp_path, "study") == f"sqlite:///{tmp_path.resolve()}/study.db"


def test_save_best_trial_writes_reproduction_summary(tmp_path):
    study = optuna.create_study(direction="maximize")

    def objective(trial):
        value = trial.suggest_float("learning_rate", 1e-5, 5e-5)
        trial.set_user_attr("output_dir", "trial-0000")
        return value

    study.optimize(objective, n_trials=1)
    result_path = save_best_trial(study, tmp_path)
    result = json.loads(result_path.read_text())

    assert result["trial_number"] == 0
    assert result["parameters"] == study.best_params
    assert result["hydra_overrides"] == [
        f"optimization.learning_rate={study.best_params['learning_rate']}"
    ]
    assert result["output_dir"] == "trial-0000"
