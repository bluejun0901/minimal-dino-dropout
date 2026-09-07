import copy
from functools import partial

import pytest
import torch
from hydra import compose, initialize_config_module
from test_training import TinyEncoder, TinyTokenizer

from minimal_dino.config import to_train_args
from minimal_dino.evaluation import embedding_diagnostics
from minimal_dino.geometry import GeometryConfig, PairedTargetGeometry
from minimal_dino.model import SentenceBYOL
from minimal_dino.objective import BYOLLoss
from minimal_dino.train import restore_checkpoint, save_checkpoint


def test_common_translation_cannot_change_pairwise_differences_or_centered_covariance():
    torch.manual_seed(1)
    embeddings = torch.randn(64, 5)
    centered = embeddings - 0.73 * torch.randn(1, 5)
    torch.testing.assert_close(torch.pdist(centered), torch.pdist(embeddings))
    torch.testing.assert_close(torch.cov(centered.T), torch.cov(embeddings.T))


def test_ema_covariance_is_covariance_of_mixture_including_mean_drift():
    geometry = PairedTargetGeometry(2, GeometryConfig(momentum=0.75))
    first = torch.tensor([[1.0, 3.0], [3.0, 1.0]])
    second = first + 4
    geometry.update((first, first))
    geometry.update((second, second))
    values = torch.cat((first, second))
    weights = torch.tensor([0.375, 0.375, 0.125, 0.125])
    expected_mean = (values * weights[:, None]).sum(0)
    delta = values - expected_mean
    expected_covariance = (delta * weights[:, None]).T @ delta
    torch.testing.assert_close(geometry.mean.squeeze(0), expected_mean)
    torch.testing.assert_close(geometry.covariance, expected_covariance)
    assert geometry.noise_covariance.count_nonzero() == 0


def test_reliability_distinguishes_equal_variance_signal_and_noise():
    geometry = PairedTargetGeometry(3, GeometryConfig(ridge=1e-5))
    geometry.covariance.copy_(torch.diag(torch.tensor([0.1, 0.2, 10.0])))
    geometry.noise_covariance.copy_(torch.diag(torch.tensor([0.1, 0.0, 0.0])))
    geometry.refresh()
    axis_gains = 1 + geometry.correction.diag()
    assert axis_gains[0] < 1.001  # low variance, entirely augmentation noise
    assert axis_gains[1] > 1.99  # low variance, stable across views
    assert axis_gains[2] < 0.6
    assert geometry.gains.min() >= 0.5 - 1e-6
    assert geometry.gains.max() <= 2 + 1e-6


def test_wrong_pairing_changes_noise_but_preserves_mean_and_total_covariance():
    torch.manual_seed(8)
    views = (torch.randn(128, 4),)
    views = (views[0], views[0] + 0.01 * torch.randn(128, 4))
    paired = PairedTargetGeometry(4, GeometryConfig(warmup_steps=1))
    shuffled = PairedTargetGeometry(4, GeometryConfig(warmup_steps=1, reliability="shuffled"))
    rng = torch.get_rng_state().clone()
    paired.update(views)
    shuffled.update(views)
    assert torch.equal(rng, torch.get_rng_state())
    torch.testing.assert_close(paired.mean, shuffled.mean)
    torch.testing.assert_close(paired.covariance, shuffled.covariance)
    assert paired.reliability.mean() > shuffled.reliability.mean() + 0.8


def test_rotation_equivariance_for_nondegenerate_covariance():
    torch.manual_seed(13)
    rotation, _ = torch.linalg.qr(torch.randn(5, 5))
    views = (torch.randn(256, 5) * torch.arange(1, 6), torch.randn(256, 5))
    left = PairedTargetGeometry(5, GeometryConfig(warmup_steps=1))
    right = PairedTargetGeometry(5, GeometryConfig(warmup_steps=1))
    left.update(views)
    right.update(tuple(x @ rotation for x in views))
    torch.testing.assert_close(left(views[0]) @ rotation, right(views[0] @ rotation))


def test_matrix_gate_handles_degenerate_covariance_and_remains_rotation_equivariant():
    covariance = torch.diag(torch.tensor([0.1, 0.1, 10.0]))
    noise = torch.tensor([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
    geometry = PairedTargetGeometry(3, GeometryConfig(ridge=1e-5))
    geometry.covariance.copy_(covariance)
    geometry.noise_covariance.copy_(noise)
    geometry.refresh()
    noisy = torch.tensor([[1.0, 1.0, 0.0]])
    stable = torch.tensor([[1.0, -1.0, 0.0]])
    torch.testing.assert_close(geometry(noisy), torch.zeros_like(noisy), atol=1e-5, rtol=0)
    assert geometry(stable).norm() > 0.99 * stable.norm()
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    rotated = PairedTargetGeometry(3, geometry.config)
    rotated.covariance.copy_(rotation.T @ covariance @ rotation)
    rotated.noise_covariance.copy_(rotation.T @ noise @ rotation)
    rotated.refresh()
    torch.testing.assert_close(
        geometry.correction @ rotation, rotation @ rotated.correction, atol=1e-4, rtol=1e-4
    )


def test_geometry_statistics_and_transform_remain_fp32_under_bfloat16_autocast():
    torch.manual_seed(25)
    geometry = PairedTargetGeometry(16, GeometryConfig(warmup_steps=1))
    views = (torch.randn(64, 16), torch.randn(64, 16))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        geometry.update(views)
        correction = geometry(views[0].bfloat16())
    assert geometry.covariance.dtype == torch.float32
    assert geometry.correction.dtype == torch.float32
    assert correction.dtype == torch.float32
    assert torch.isfinite(correction).all()


def test_noncommuting_matrix_sandwich_respects_gain_bounds():
    torch.manual_seed(26)
    signal = torch.randn(64, 12) @ torch.randn(12, 12)
    views = (signal + torch.randn(64, 12), signal + torch.randn(64, 12))
    geometry = PairedTargetGeometry(12, GeometryConfig(warmup_steps=1))
    geometry.update(views)
    assert geometry.gains.min() >= geometry.config.min_gain - 1e-5
    assert geometry.gains.max() <= geometry.config.max_gain + 1e-5
    before = geometry.correction.clone()
    geometry(views[0])
    assert torch.equal(before, geometry.correction)


@pytest.mark.parametrize("mode", ["paired", "none", "shuffled", "diagonal"])
def test_degenerate_and_singleton_batches_stay_finite(mode):
    geometry = PairedTargetGeometry(4, GeometryConfig(warmup_steps=1, reliability=mode))
    for count in (1, 3):
        view = torch.ones(count, 4)
        geometry.update((view, view))
        assert torch.isfinite(geometry(view)).all()
        assert geometry.gains.min() >= 0.5
        assert geometry.gains.max() <= 2


def test_zero_strength_exactly_matches_baseline_and_statistics_are_causal():
    torch.manual_seed(19)
    baseline = BYOLLoss(8, center_momentum=0.96)
    zero = BYOLLoss(8, center_momentum=0.96, target_geometry={"strength": 0, "warmup_steps": 1})
    active = BYOLLoss(8, target_geometry={"warmup_steps": 2, "update_interval": 1})
    for step in range(4):
        views = (torch.randn(16, 8), torch.randn(16, 8))
        assert torch.equal(
            baseline.transform_target(views[0], 0.05), zero.transform_target(views[0], 0.05)
        )
        if step < 2:
            assert active.geometry.correction.count_nonzero() == 0
        for objective in (baseline, zero, active):
            objective.update_center(views)
    assert active.geometry.correction.count_nonzero() > 0


def test_transform_is_target_only_preserves_raw_embeddings_and_stops_teacher_gradient():
    torch.manual_seed(23)
    student = SentenceBYOL(
        TinyEncoder(), projection_dim=4, projector_hidden_dim=16, predictor_hidden_dim=12
    )
    teacher = copy.deepcopy(student).eval().requires_grad_(False)
    objective = BYOLLoss(8, target_geometry={"warmup_steps": 1})
    objective.update_center((torch.randn(32, 8), torch.randn(32, 8)))
    transform = partial(objective.transform_target, center_scale=0.05)
    batch = {
        "input_ids": torch.tensor([[1, 2], [3, 4]]),
        "attention_mask": torch.ones(2, 2, dtype=torch.long),
    }
    online = student(**batch, use_dropout=True)
    with torch.no_grad():
        target = teacher(**batch, target=True, use_dropout=False, target_transform=transform)
        raw = teacher(**batch, target=True, use_dropout=False)
    assert torch.equal(target.embedding, raw.embedding)
    assert not torch.equal(target.projection, raw.projection)
    loss, _ = objective(
        (online.prediction, online.prediction), (target.projection, target.projection)
    )
    loss.backward()
    assert any(p.grad is not None for p in student.encoder.parameters())
    assert all(p.grad is None for p in teacher.parameters())
    assert all(not b.requires_grad for b in objective.buffers())
    with pytest.raises(ValueError, match="target_transform"):
        student(**batch, use_dropout=False, target_transform=transform)


def test_scalar_target_callable_is_bitwise_identical_to_original_center_path():
    model = SentenceBYOL(
        TinyEncoder(), projection_dim=4, projector_hidden_dim=16, predictor_hidden_dim=12
    ).eval()
    objective = BYOLLoss(8)
    objective.center.copy_(torch.randn(1, 8))
    batch = {
        "input_ids": torch.tensor([[1, 2], [3, 4]]),
        "attention_mask": torch.ones(2, 2, dtype=torch.long),
    }
    torch.manual_seed(31)
    legacy = model(
        **batch,
        use_dropout=True,
        target=True,
        center=objective.center,
        center_scale=0.05,
        dropout_probability=0.02,
    )
    torch.manual_seed(31)
    refactored = model(
        **batch,
        use_dropout=True,
        target=True,
        target_transform=partial(objective.transform_target, center_scale=0.05),
        dropout_probability=0.02,
    )
    assert torch.equal(legacy.embedding, refactored.embedding)
    assert torch.equal(legacy.projection, refactored.projection)


def test_checkpoint_restores_geometry_and_rejects_method_switch(tmp_path):
    from types import SimpleNamespace

    model = SentenceBYOL(
        TinyEncoder(), projection_dim=4, projector_hidden_dim=16, predictor_hidden_dim=12
    )
    teacher = copy.deepcopy(model)
    config = {"warmup_steps": 1}
    objective = BYOLLoss(8, target_geometry=config)
    objective.update_center((torch.randn(16, 8), torch.randn(16, 8)))
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    path = save_checkpoint(
        tmp_path,
        model,
        teacher,
        objective,
        optimizer,
        scheduler,
        TinyTokenizer(),
        SimpleNamespace(objective="byol", target_geometry=config),
        step=1,
    )
    restored = BYOLLoss(8, target_geometry=config)
    restore_checkpoint(path, model, teacher, restored, optimizer, scheduler, torch.device("cpu"))
    for name, value in objective.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])
    wrong = BYOLLoss(8, target_geometry={**config, "reliability": "none"})
    with pytest.raises(ValueError, match="target geometry"):
        restore_checkpoint(path, model, teacher, wrong, optimizer, scheduler, torch.device("cpu"))


def test_hydra_geometry_is_opt_in_and_validates_config():
    with initialize_config_module(version_base="1.3", config_module="minimal_dino.conf"):
        baseline = to_train_args(compose(config_name="config"))
        paired = to_train_args(compose(config_name="config", overrides=["objective=paired_byol"]))
    assert baseline.target_geometry is None
    assert paired.target_geometry["reliability"] == "paired"
    for invalid in (
        {"ridge": 0},
        {"strength": float("nan")},
        {"update_interval": 0},
        {"reliability": "unknown"},
    ):
        with pytest.raises(ValueError):
            GeometryConfig(**invalid)


def test_collapse_diagnostics_distinguish_rank_zero_and_rank_one():
    zero = embedding_diagnostics(torch.ones(8, 4))
    one = embedding_diagnostics(torch.arange(8).float()[:, None].repeat(1, 4))
    assert zero["effective_rank"] == 0
    assert zero["participation_ratio"] == 0
    assert one["participation_ratio"] == pytest.approx(1)
    assert one["covariance_top1_mass"] == pytest.approx(1)
