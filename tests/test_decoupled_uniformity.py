import pytest
import torch
from hydra import compose, initialize_config_module
from torch.nn import functional as F

from minimal_dino.config import to_train_args
from minimal_dino.objective import DecoupledUniformityLoss, UniformityLoss


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_decoupled_matches_explicit_centroid_kernel_and_scale_invariance(dtype):
    torch.manual_seed(7)
    first = torch.randn(5, 8, dtype=dtype, requires_grad=True)
    second = torch.randn(5, 8, dtype=dtype, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = DecoupledUniformityLoss(1.5)((first, second))
    mean = (F.normalize(first.float(), dim=-1) + F.normalize(second.float(), dim=-1)) / 2
    expected = (-1.5 * torch.pdist(mean).square()).exp().mean().log()
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert actual.dtype == torch.float32
    assert torch.isfinite(first.grad).all() and first.grad.norm() > 0
    assert torch.isfinite(second.grad).all() and second.grad.norm() > 0
    torch.testing.assert_close(
        actual, DecoupledUniformityLoss(1.5)((second.float() * 2, first.float() * 4))
    )


def test_centroid_norm_carries_alignment_gradient_removed_by_normalized_mean():
    # Independent orthogonal centers and opposite view noise: normalized means
    # are fixed while their lengths reflect augmentation agreement.
    theta = torch.tensor(0.4, requires_grad=True)
    centers, noise = torch.eye(8)[:4], torch.eye(8)[4:]
    first = theta.cos() * centers + theta.sin() * noise
    second = theta.cos() * centers - theta.sin() * noise
    old = UniformityLoss(2)((first + second) / 2)
    new = DecoupledUniformityLoss(2)((first, second))
    (old_gradient,) = torch.autograd.grad(old, theta, retain_graph=True)
    (new_gradient,) = torch.autograd.grad(new, theta)
    torch.testing.assert_close(old, torch.tensor(-4.0))
    torch.testing.assert_close(new, -4 * theta.cos().square())
    assert abs(old_gradient.item()) < 1e-6
    torch.testing.assert_close(new_gradient, 8 * theta.sin() * theta.cos())
    assert new_gradient > 0  # Gradient descent reduces view separation.


def test_identical_views_recover_uniformity_and_simplex_lower_bound():
    # Four simplex vertices on S^3: theoretical optimum -2*t*n/(n-1).
    vertices = F.normalize(torch.eye(4) - torch.ones(4, 4) / 4, dim=-1)
    loss = DecoupledUniformityLoss(2)((vertices, vertices))
    torch.testing.assert_close(loss, UniformityLoss(2)(vertices))
    torch.testing.assert_close(loss, torch.tensor(-16 / 3))


@pytest.mark.parametrize("size", [0, 1])
def test_no_pairs_have_zero_loss_and_zero_gradients(size):
    first = torch.randn(size, 8, requires_grad=True)
    second = torch.randn(size, 8, requires_grad=True)
    loss = DecoupledUniformityLoss()((first, second))
    loss.backward()
    assert loss.item() == 0
    torch.testing.assert_close(first.grad, torch.zeros_like(first))
    torch.testing.assert_close(second.grad, torch.zeros_like(second))


@pytest.mark.parametrize("t", [0, -1, float("nan"), float("inf"), True])
def test_rejects_invalid_kernel_scale(t):
    with pytest.raises(ValueError):
        DecoupledUniformityLoss(t)


@pytest.mark.parametrize("objective", ["byol", "infonce"])
def test_mode_config_is_explicit_and_rejects_typos(objective):
    with initialize_config_module(version_base=None, config_module="minimal_dino.conf"):
        config = compose(
            config_name="config",
            overrides=[f"objective={objective}", "objective.uniformity_mode=decoupled"],
        )
    assert to_train_args(config).uniformity_mode == "decoupled"
    config.objective.uniformity_mode = "normalized_mean"
    assert to_train_args(config).uniformity_mode == "normalized_mean"
    config.objective.uniformity_mode = "typo"
    with pytest.raises(ValueError, match="uniformity_mode"):
        to_train_args(config)
