from hydra import compose, initialize_config_module
from optuna.trial import FixedTrial

from minimal_dino.tune import sample_config


def test_sample_config_updates_only_tuned_training_values():
    with initialize_config_module(config_module="minimal_dino.conf", version_base="1.3"):
        config = compose(config_name="tune")
    original_center_scale = config.objective.center_scale
    original_dropout = config.model.dropout
    trial = FixedTrial(
        {
            "encoder_learning_rate": 2e-6,
            "head_learning_rate": 2e-4,
            "dropout": 0.15,
            "center_scale": 0.2,
            "center_momentum": 0.97,
            "teacher_one_minus_momentum": 1e-3,
            "encoder_freeze_steps": 300,
            "warmup_ratio": 0.08,
        }
    )

    sampled = sample_config(config, trial)

    assert sampled.objective.center_scale == 0.2
    assert sampled.model.dropout == 0.15
    assert sampled.objective.center_momentum == 0.97
    assert sampled.optimization.encoder_learning_rate == 2e-6
    assert sampled.optimization.head_learning_rate == 2e-4
    assert sampled.teacher.momentum == 0.999
    assert sampled.optimization.encoder_freeze_steps == 300
    assert sampled.optimization.warmup_ratio == 0.08
    assert sampled.optimization.batch_size == config.optimization.batch_size
    assert sampled.runtime.output_dir == config.runtime.output_dir
    assert config.objective.center_scale == original_center_scale
    assert config.model.dropout == original_dropout
