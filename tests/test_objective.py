import torch
from torch.nn import functional as F

from minimal_dino.objective import DINOLoss, InfoNCELoss, build_objective


def test_dino_loss_uses_only_opposite_views_and_stops_teacher_gradient():
    objective = DINOLoss(output_dim=3, student_temp=0.2, center_momentum=0.5)
    student1 = torch.tensor([[0.2, -0.1, 0.4]], requires_grad=True)
    student2 = torch.tensor([[0.5, 0.0, -0.2]], requires_grad=True)
    teacher1 = torch.tensor([[0.1, 0.3, -0.2]], requires_grad=True)
    teacher2 = torch.tensor([[0.4, -0.1, 0.2]], requires_grad=True)

    loss, _ = objective((student1, student2), (teacher1, teacher2), teacher_temp=0.1)
    probability1 = F.softmax(teacher1.detach() / 0.1, dim=-1)
    probability2 = F.softmax(teacher2.detach() / 0.1, dim=-1)
    expected = (
        -(probability1 * F.log_softmax(student2 / 0.2, dim=-1)).sum()
        - (probability2 * F.log_softmax(student1 / 0.2, dim=-1)).sum()
    ) / 2
    loss.backward()

    assert torch.allclose(loss, expected)
    assert student1.grad is not None
    assert student2.grad is not None
    assert teacher1.grad is None
    assert teacher2.grad is None


def test_center_uses_raw_teacher_logits_after_loss():
    objective = DINOLoss(output_dim=2, center_momentum=0.5)
    views = (torch.tensor([[2.0, 0.0]]), torch.tensor([[0.0, 2.0]]))
    student = (torch.zeros(1, 2), torch.zeros(1, 2))

    _, metrics_before = objective(student, views, teacher_temp=0.1)
    objective.update_center(views)

    assert metrics_before["center_norm"].item() == 0.0
    assert torch.equal(objective.center, torch.tensor([[0.5, 0.5]]))


def test_dino_can_skip_diagnostics_without_changing_loss():
    objective = DINOLoss(output_dim=3, student_temp=0.2)
    student = (torch.randn(2, 3), torch.randn(2, 3))
    teacher = (torch.randn(2, 3), torch.randn(2, 3))

    expected, metrics = objective(student, teacher, teacher_temp=0.1)
    actual, skipped_metrics = objective(
        student, teacher, teacher_temp=0.1, compute_metrics=False
    )

    assert torch.equal(actual, expected)
    assert metrics
    assert skipped_metrics == {}


def test_dino_reuses_shared_teacher_view_for_loss_and_center():
    objective = DINOLoss(output_dim=3, student_temp=0.2, center_momentum=0.5)
    student = (torch.randn(2, 3), torch.randn(2, 3))
    teacher = torch.randn(2, 3)

    shared_loss, _ = objective(student, (teacher, teacher), teacher_temp=0.1)
    copied_loss, _ = objective(student, (teacher, teacher.clone()), teacher_temp=0.1)
    objective.update_center((teacher, teacher))

    assert torch.equal(shared_loss, copied_loss)
    assert torch.allclose(objective.center, teacher.mean(dim=0, keepdim=True) * 0.5)


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
    dino = build_objective(
        "dino", output_dim=3, student_temp=0.1, center_momentum=0.9, infonce_temp=0.05
    )
    infonce = build_objective(
        "infonce", output_dim=3, student_temp=0.1, center_momentum=0.9, infonce_temp=0.2
    )

    assert isinstance(dino, DINOLoss)
    assert dino.representation == "logits"
    assert isinstance(infonce, InfoNCELoss)
    assert infonce.representation == "embedding"
    assert infonce.temperature == 0.2
