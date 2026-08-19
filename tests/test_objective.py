import torch
from torch.nn import functional as F

from minimal_dino.objective import DINOLoss


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
