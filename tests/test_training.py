import copy
import json
import subprocess
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_module
from torch import nn

from minimal_dino.model import SentenceDINO
from minimal_dino.objective import DINOLoss
from minimal_dino.train import (
    DEFAULT_MODEL_REVISION,
    _print_progress,
    cosine_teacher_momentum,
    is_dino_reset_step,
    log_metrics,
    remove_old_periodic_checkpoints,
    reset_dino_state,
    resolve_model_revision,
    restore_checkpoint,
    save_checkpoint,
    save_run_artifacts,
    update_teacher,
)


class TinyConfig(SimpleNamespace):
    def to_dict(self):
        return vars(self)


class TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = TinyConfig(hidden_size=8)
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


def test_reset_dino_state_copies_student_and_zeros_center():
    student = SentenceDINO(TinyEncoder(), output_dim=12, head_hidden_dim=16, bottleneck_dim=4)
    teacher = copy.deepcopy(student).eval().requires_grad_(False)
    objective = DINOLoss(12)

    with torch.no_grad():
        next(student.parameters()).add_(1)
        objective.center.fill_(2)
    reset_dino_state(student, teacher, objective)

    assert all(
        torch.equal(student_value, teacher.state_dict()[name])
        for name, student_value in student.state_dict().items()
    )
    assert torch.equal(objective.center, torch.zeros_like(objective.center))
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_dino_reset_interval_uses_completed_steps_and_none_disables_it():
    assert [step for step in range(1, 8) if is_dino_reset_step(step, 3)] == [3, 6]
    assert not any(is_dino_reset_step(step, None) for step in range(1, 8))


class TinyTokenizer:
    def save_pretrained(self, path):
        path.mkdir(parents=True, exist_ok=True)


def test_checkpoint_round_trip_restores_models_optimizer_center_and_rng(tmp_path):
    torch.manual_seed(7)
    student = SentenceDINO(TinyEncoder(), output_dim=12, head_hidden_dim=16, bottleneck_dim=4)
    teacher = copy.deepcopy(student).eval().requires_grad_(False)
    objective = DINOLoss(12)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    original_student = copy.deepcopy(student.state_dict())
    checkpoint = save_checkpoint(
        tmp_path,
        student,
        teacher,
        objective,
        optimizer,
        scheduler,
        TinyTokenizer(),
        SimpleNamespace(seed=7),
        step=3,
        checkpoint_name="checkpoint-step-3.pt",
    )
    expected_random = torch.rand(4)

    with torch.no_grad():
        next(student.parameters()).add_(10)
        objective.center.add_(1)
    step = restore_checkpoint(
        checkpoint, student, teacher, objective, optimizer, scheduler, torch.device("cpu")
    )
    actual_random = torch.rand(4)

    assert step == 3
    assert all(
        torch.equal(value, original_student[name]) for name, value in student.state_dict().items()
    )
    assert torch.equal(objective.center, torch.zeros_like(objective.center))
    assert torch.equal(actual_random, expected_random)
    assert not (tmp_path / "checkpoint-step-3.pt.tmp").exists()


def test_periodic_checkpoint_retention_keeps_newest_steps(tmp_path):
    for step in (100, 300, 200):
        (tmp_path / f"checkpoint-step-{step}.pt").touch()

    remove_old_periodic_checkpoints(tmp_path, keep_last=2)

    assert not (tmp_path / "checkpoint-step-100.pt").exists()
    assert (tmp_path / "checkpoint-step-200.pt").exists()
    assert (tmp_path / "checkpoint-step-300.pt").exists()


def test_default_model_resolves_to_immutable_revision():
    assert resolve_model_revision("bert-base-uncased", None) == DEFAULT_MODEL_REVISION


def test_custom_hub_model_requires_revision():
    with pytest.raises(ValueError, match="model.revision"):
        resolve_model_revision("organization/model", None)


def test_model_revision_must_be_full_commit_hash():
    with pytest.raises(ValueError, match="40-character commit hash"):
        resolve_model_revision("organization/model", "main")


def test_local_model_does_not_require_revision(tmp_path):
    assert resolve_model_revision(str(tmp_path), None) is None


def test_log_metrics_matches_stdout_and_appends_jsonl(tmp_path, capsys):
    first = {"step": 1, "loss": 2.0}
    second = {"step": 2, "loss": 1.0}

    log_metrics(tmp_path, first)
    log_metrics(tmp_path, second)

    stdout_lines = capsys.readouterr().out.splitlines()
    file_lines = (tmp_path / "metrics.jsonl").read_text().splitlines()
    assert file_lines == stdout_lines
    assert [json.loads(line) for line in file_lines] == [first, second]


def test_quiet_metrics_only_write_jsonl(tmp_path, capsys):
    metrics = {"step": 1, "loss": 2.0}

    log_metrics(tmp_path, metrics, quiet=True)

    assert capsys.readouterr().out == ""
    assert json.loads((tmp_path / "metrics.jsonl").read_text()) == metrics


def test_log_metrics_writes_namespaced_tensorboard_scalars(tmp_path):
    class RecordingWriter:
        def __init__(self):
            self.scalars = []
            self.flush_count = 0

        def add_scalar(self, tag, value, step):
            self.scalars.append((tag, value, step))

        def flush(self):
            self.flush_count += 1

    writer = RecordingWriter()
    log_metrics(
        tmp_path,
        {"step": 4, "loss": 1.25, "lr": 3e-5},
        quiet=True,
        tensorboard_writer=writer,
        namespace="train",
    )

    assert writer.scalars == [("train/loss", 1.25, 4), ("train/lr", 3e-5, 4)]
    assert writer.flush_count == 1


def test_progress_bar_finishes_with_newline(capsys):
    _print_progress(2, 2, width=4)

    assert capsys.readouterr().out == "\rTraining [####] 2/2\n"


def test_save_run_artifacts_records_config_and_dirty_git_state(tmp_path, monkeypatch):
    outputs = iter(
        [
            str(tmp_path) + "\n",
            "0123456789abcdef0123456789abcdef01234567\n",
            " M src/minimal_dino/train.py\n?? notes.txt\n",
            "diff --git a/file b/file\n",
        ]
    )

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout=next(outputs), stderr="")

    monkeypatch.setattr("minimal_dino.train.subprocess.run", fake_run)
    args = SimpleNamespace(output_dir=str(tmp_path), seed=42, model_revision="abc")

    save_run_artifacts(tmp_path, args)

    assert json.loads((tmp_path / "config.json").read_text()) == vars(args)
    git_state = json.loads((tmp_path / "git_state.json").read_text())
    assert git_state == {
        "available": True,
        "commit_hash": "0123456789abcdef0123456789abcdef01234567",
        "dirty": True,
        "repository_root": str(tmp_path),
        "status": " M src/minimal_dino/train.py\n?? notes.txt",
        "diff_file": "git.diff",
    }
    assert (tmp_path / "git.diff").read_text() == "diff --git a/file b/file\n"


def test_save_run_artifacts_survives_missing_git(tmp_path, monkeypatch):
    def missing_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr("minimal_dino.train.subprocess.run", missing_git)

    save_run_artifacts(tmp_path, SimpleNamespace(seed=7))

    git_state = json.loads((tmp_path / "git_state.json").read_text())
    assert git_state["available"] is False
    assert "git" in git_state["error"]
    assert (tmp_path / "git.diff").read_text() == ""


def test_hydra_config_groups_compose_and_translate_to_training_args():
    from minimal_dino.config import to_train_args

    with initialize_config_module(version_base="1.3", config_module="minimal_dino.conf"):
        default_config = compose(
            config_name="config", overrides=["data.train_file=train.txt"]
        )
        reset_config = compose(
            config_name="config",
            overrides=["data.train_file=train.txt", "objective.reset_interval=25"],
        )
        alternate_config = compose(
            config_name="config",
            overrides=[
                "data.train_file=train.txt",
                "objective=infonce",
                "objective.temperature=0.2",
                "augmentation=word",
                "model.random_init=true",
                "logging.quiet=true",
                "logging.tensorboard=false",
                "runtime.device=cpu",
            ],
        )

    default_args = to_train_args(default_config)
    reset_args = to_train_args(reset_config)
    alternate_args = to_train_args(alternate_config)

    assert default_args.objective == "dino"
    assert default_args.dino_reset_interval is None
    assert reset_args.dino_reset_interval == 25
    assert default_args.augmentation == "dropout"
    assert default_args.quiet is True
    assert default_args.tensorboard is True
    assert default_args.random_init is False
    assert alternate_args.objective == "infonce"
    assert alternate_args.infonce_temp == 0.2
    assert alternate_args.augmentation == "word"
    assert alternate_args.random_init is True
    assert alternate_args.quiet is True
    assert alternate_args.tensorboard is False
