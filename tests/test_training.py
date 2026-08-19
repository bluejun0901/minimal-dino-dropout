import copy
from types import SimpleNamespace

import torch
from torch import nn

from minimal_dino.model import SentenceDINO
from minimal_dino.objective import DINOLoss
from minimal_dino.train import cosine_teacher_momentum, update_teacher


class TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embedding = nn.Embedding(24, 8)
        self.dropout = nn.Dropout(0.2)
        self.layer = nn.Linear(8, 8)

    def forward(self, input_ids, attention_mask, return_dict=True):
        hidden = self.layer(self.dropout(self.embedding(input_ids)))
        return SimpleNamespace(last_hidden_state=hidden)


def test_one_step_smoke_has_student_gradients_no_teacher_gradients_and_ema():
    torch.manual_seed(2)
    student = SentenceDINO(TinyEncoder(), output_dim=12, head_hidden_dim=16, bottleneck_dim=4)
    teacher = copy.deepcopy(student).eval().requires_grad_(False)
    objective = DINOLoss(12)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        "attention_mask": torch.ones(3, 3, dtype=torch.long),
    }
    teacher_before = next(teacher.parameters()).detach().clone()

    student1 = student(**batch, use_dropout=True)
    student2 = student(**batch, use_dropout=True)
    with torch.no_grad():
        teacher1 = teacher(**batch, use_dropout=True)
        teacher2 = teacher(**batch, use_dropout=True)
    loss, _ = objective(
        (student1.logits, student2.logits), (teacher1.logits, teacher2.logits), 0.04
    )
    loss.backward()
    assert any(parameter.grad is not None for parameter in student.parameters())
    assert all(parameter.grad is None for parameter in teacher.parameters())
    optimizer.step()
    update_teacher(student, teacher, momentum=0.9)
    objective.update_center((teacher1.logits, teacher2.logits))

    student_after = next(student.parameters()).detach()
    teacher_after = next(teacher.parameters()).detach()
    expected = teacher_before * 0.9 + student_after * 0.1
    assert torch.allclose(teacher_after, expected)
    assert objective.center.norm() > 0
    assert torch.isfinite(loss)


def test_teacher_momentum_cosine_schedule_reaches_one():
    values = [cosine_teacher_momentum(step, 5, 0.996) for step in range(5)]
    assert values[0] == 0.996
    assert values[-1] == 1.0
    assert values == sorted(values)
