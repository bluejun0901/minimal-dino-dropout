import pytest
import torch
from torch.nn import functional as F

from minimal_dino.objective import BYOLLoss, InfoNCELoss, UniformityLoss, build_objective


def test_uniformity_matches_wang_formula_and_gradients():
    embeddings = torch.tensor(
        [[2.0, 0.5, -1.0], [-1.0, 3.0, 0.2], [0.1, -0.5, 4.0]], requires_grad=True
    )
    t = 1.5
    loss = UniformityLoss(t)(embeddings)
    normalized = F.normalize(embeddings, dim=-1)
    expected = torch.pdist(normalized).square().mul(-t).exp().mean().log()
    expected_grad, = torch.autograd.grad(expected, embeddings, retain_graph=True)
    loss.backward()

    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(embeddings.grad, expected_grad)
    assert torch.count_nonzero(embeddings.grad) > 0
    torch.testing.assert_close(
        UniformityLoss(t)(embeddings.detach() * torch.tensor([[2.0], [3.0], [4.0]])),
        loss,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_uniformity_is_stable_for_large_t_under_autocast(dtype):
    embeddings = torch.eye(3, dtype=dtype, requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = UniformityLoss(t=1000.0)(embeddings)
    loss.backward()

    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss, torch.tensor(-2000.0))
    assert torch.isfinite(embeddings.grad).all()


@pytest.mark.parametrize("batch_size", [0, 1])
def test_uniformity_without_pairs_is_differentiable_zero(batch_size):
    embeddings = torch.randn(batch_size, 3, requires_grad=True)
    loss = UniformityLoss()(embeddings)
    loss.backward()

    assert loss.item() == 0.0
    torch.testing.assert_close(embeddings.grad, torch.zeros_like(embeddings))


def test_uniformity_prefers_spread_embeddings_and_handles_collapse():
    collapsed = torch.ones(3, 3, requires_grad=True)
    collapsed_loss = UniformityLoss()(collapsed)
    collapsed_loss.backward()

    torch.testing.assert_close(collapsed_loss, torch.tensor(0.0))
    assert UniformityLoss()(torch.eye(3)) < collapsed_loss
    assert torch.isfinite(collapsed.grad).all()


@pytest.mark.parametrize("t", [0.0, -1.0, float("nan"), float("inf"), True, "bad"])
def test_uniformity_rejects_invalid_t(t):
    with pytest.raises(ValueError, match="uniformity t"):
        UniformityLoss(t)


def test_byol_loss_matches_opposite_views_and_stops_teacher_gradient():
    objective = BYOLLoss(embedding_dim=3)
    student1 = torch.tensor([[0.2, -0.1, 0.4]], requires_grad=True)
    student2 = torch.tensor([[0.5, 0.0, -0.2]], requires_grad=True)
    teacher1 = torch.tensor([[0.1, 0.3, -0.2]], requires_grad=True)
    teacher2 = torch.tensor([[0.4, -0.1, 0.2]], requires_grad=True)

    loss, metrics = objective((student1, student2), (teacher1, teacher2))
    expected = (
        2 - 2 * F.cosine_similarity(student1, teacher2).mean()
        + 2
        - 2 * F.cosine_similarity(student2, teacher1).mean()
    ) / 2
    loss.backward()

    assert torch.allclose(loss, expected)
    assert student1.grad is not None
    assert student2.grad is not None
    assert teacher1.grad is None
    assert teacher2.grad is None
    assert torch.allclose(metrics["byol_cosine"], 1 - loss.detach() / 2)


def test_byol_loss_is_scale_invariant():
    objective = BYOLLoss(embedding_dim=3)
    student = (torch.randn(2, 3), torch.randn(2, 3))
    teacher = (torch.randn(2, 3), torch.randn(2, 3))

    expected, _ = objective(student, teacher)
    actual, _ = objective(
        (student[0] * 3, student[1] * 5), (teacher[0] * 7, teacher[1] * 11)
    )

    assert torch.allclose(actual, expected)


def test_byol_can_skip_diagnostics_without_changing_loss():
    objective = BYOLLoss(embedding_dim=3)
    student = (torch.randn(2, 3), torch.randn(2, 3))
    teacher = (torch.randn(2, 3), torch.randn(2, 3))

    expected, metrics = objective(student, teacher)
    actual, skipped_metrics = objective(student, teacher, compute_metrics=False)

    assert torch.equal(actual, expected)
    assert metrics
    assert skipped_metrics == {}


def test_byol_center_updates_from_raw_teacher_embeddings_after_loss():
    objective = BYOLLoss(embedding_dim=2, center_momentum=0.5)
    teacher = (torch.tensor([[2.0, 0.0]]), torch.tensor([[0.0, 2.0]]))
    student = (torch.ones(1, 2), torch.ones(1, 2))

    _, metrics_before = objective(student, teacher)
    objective.update_center(teacher)

    assert metrics_before["center_norm"] == 0
    assert torch.equal(objective.center, torch.tensor([[0.5, 0.5]]))


def test_byol_loss_does_not_center_target_after_projection():
    objective = BYOLLoss(embedding_dim=2)
    objective.center.copy_(torch.tensor([[1.0, 1.0]]))
    student = (torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]]))
    teacher = (torch.tensor([[0.0, 1.0]]), torch.tensor([[1.0, 0.0]]))

    loss, _ = objective(student, teacher)

    assert torch.equal(loss, torch.tensor(0.0))


def test_infonce_uses_diagonal_pairs_in_both_directions():
    objective = InfoNCELoss(temperature=0.2)
    view1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    view2 = torch.tensor([[0.8, 0.2], [0.1, 0.9]], requires_grad=True)

    loss, metrics = objective((view1, view2))
    logits = F.normalize(view1, dim=-1) @ F.normalize(view2, dim=-1).T / 0.2
    labels = torch.arange(2)
    expected = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    loss.backward()

    assert torch.allclose(loss, expected)
    assert view1.grad is not None
    assert view2.grad is not None
    assert metrics["positive_cosine"] > metrics["negative_cosine"]
    assert metrics["contrastive_accuracy"] == 1


def test_objective_factory_defaults_to_distinct_representation_paths():
    byol = build_objective(
        "byol",
        projection_dim=3,
        embedding_dim=5,
        center_momentum=0.8,
        infonce_temp=0.05,
    )
    infonce = build_objective(
        "infonce", projection_dim=3, center_momentum=0.8, infonce_temp=0.2
    )

    assert isinstance(byol, BYOLLoss)
    assert byol.representation == "prediction"
    assert byol.center_momentum == 0.8
    assert byol.center.shape == (1, 5)
    assert isinstance(infonce, InfoNCELoss)
    assert infonce.representation == "embedding"
    assert infonce.temperature == 0.2
