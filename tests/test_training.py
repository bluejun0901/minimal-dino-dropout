import copy
import json
import subprocess
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_module
from torch import nn

from minimal_dino.model import SentenceBYOL
from minimal_dino.objective import BYOLLoss
from minimal_dino.train import (
    DEFAULT_MODEL_REVISION,
    _print_progress,
    cosine_center_scale,
    cosine_teacher_momentum,
    load_max_logged_metric,
    log_metrics,
    remove_old_periodic_checkpoints,
    resolve_model_revision,
    restore_checkpoint,
    save_checkpoint,
    save_run_artifacts,
    set_encoder_trainable,
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


@pytest.mark.parametrize("objective", ["byol", "infonce"])
@pytest.mark.parametrize(
    "uniformity_weight,uniformity_mode",
    [(0.0, "normalized_mean"), (0.1, "normalized_mean"), (0.1, "decoupled")],
)
@pytest.mark.parametrize("resume_step", [0, 1])
@pytest.mark.parametrize(
    "augmentations", [["dropout"], ["word"], ["word", "dropout"], ["dropout", "word"]]
)
def test_training_applies_selected_augmentations(
    tmp_path,
    monkeypatch,
    objective,
    augmentations,
    resume_step,
    uniformity_weight,
    uniformity_mode,
):
    from minimal_dino.config import to_train_args
    from minimal_dino.train import train

    train_file = tmp_path / "sentences.txt"
    train_file.write_text("one two three\nfour five six\nseven eight nine\n")
    with initialize_config_module(version_base="1.3", config_module="minimal_dino.conf"):
        config = compose(
            config_name="config",
            overrides=[
                f"objective={objective}",
                f"objective.uniformity_weight={uniformity_weight}",
                f"objective.uniformity_mode={uniformity_mode}",
                "objective.uniformity_t=1.5",
                "augmentation.names=[" + ",".join(augmentations) + "]",
                f"data.train_file={train_file}",
                "data.num_workers=0",
                "runtime.device=cpu",
                f"runtime.output_dir={tmp_path / 'run'}",
                "++optimization.max_steps=3",
                "optimization.epochs=3",
                "optimization.encoder_freeze_steps=0",
                "evaluation.steps=0",
                "checkpoint.save_steps=0",
                "logging.tensorboard=false",
                "logging.steps=1",
                "model.projection_dim=4",
                "model.projector_hidden_dim=16",
                "model.predictor_hidden_dim=12",
            ],
        )
    args = to_train_args(config)
    args.center_scale_start = 0.1
    args.center_scale_end = 0.5
    if resume_step:
        args.resume_from_checkpoint = "mock-checkpoint.pt"
        monkeypatch.setattr("minimal_dino.train.restore_checkpoint", lambda *a: resume_step)
    forwards = []
    augmented_sentences = []
    target_scales = []

    class RecordingModel(SentenceBYOL):
        def forward(self, *a, **kwargs):
            if kwargs.get("center") is not None:
                target_scales.append(kwargs["center_scale"])
            return super().forward(*a, **kwargs)

    class RecordingEncoder(TinyEncoder):
        def forward(self, input_ids, attention_mask, return_dict=True):
            forwards.append((input_ids.clone(), self.dropout.training, self.dropout.p))
            return super().forward(input_ids, attention_mask, return_dict)

    class Tokenizer:
        def __call__(self, sentences, **kwargs):
            lengths = [len(sentence.split()) for sentence in sentences]
            mask = torch.arange(max(lengths))[None, :] < torch.tensor(lengths)[:, None]
            return {"input_ids": mask.long(), "attention_mask": mask.long()}

    def build_model(*unused, **kwargs):
        encoder = RecordingEncoder()
        encoder.dropout.p = kwargs["dropout"]
        return RecordingModel(
            encoder, projection_dim=4, projector_hidden_dim=16, predictor_hidden_dim=12
        )

    def augment(text, strength):
        # Different lengths also verify that combined views may have different padding.
        augmented_sentences.append(text)
        return text + " extra" * len(augmented_sentences)

    monkeypatch.setattr(
        "minimal_dino.train.AutoTokenizer.from_pretrained", lambda *a, **k: Tokenizer()
    )
    monkeypatch.setattr(SentenceBYOL, "from_pretrained", build_model)
    monkeypatch.setattr("minimal_dino.data.augment_words", augment)
    monkeypatch.setattr("minimal_dino.train.save_run_artifacts", lambda *a, **k: None)
    if uniformity_weight == 0:
        def unexpected_uniformity(*a, **k):
            pytest.fail("Disabled uniformity must not be computed")

        monkeypatch.setattr("minimal_dino.train.UniformityLoss.forward", unexpected_uniformity)

    train(args, save_final_checkpoint=False)

    has_word = "word" in augmentations
    has_dropout = "dropout" in augmentations
    steps = 3 - resume_step
    forwards_per_step = 3 if objective == "byol" and not has_word else 4
    assert len(forwards) == forwards_per_step * steps
    assert len(augmented_sentences) == (6 * steps if has_word else 0)
    assert all(enabled == has_dropout for _, enabled, _ in forwards)
    for offset in range(0, len(forwards), forwards_per_step):
        batch_forwards = forwards[offset : offset + forwards_per_step]
        student_forwards, teacher_forwards = batch_forwards[:-2], batch_forwards[-2:]
        assert all(rate == args.dropout for _, _, rate in student_forwards)
        if has_dropout:
            assert all(rate == args.target_dropout for _, _, rate in teacher_forwards)
        if has_word:
            assert len(student_forwards) == 2
            assert student_forwards[0][0].shape[1] != student_forwards[1][0].shape[1]
            for student_forward, teacher_forward in zip(student_forwards, teacher_forwards):
                assert torch.equal(student_forward[0], teacher_forward[0])
        else:
            assert all(ids.shape[1] == 3 for ids, _, _ in batch_forwards)
    metrics = [
        json.loads(line)
        for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    ]
    assert [record["step"] for record in metrics] == list(range(resume_step + 1, 4))
    assert all(torch.isfinite(torch.tensor(record["loss"])) for record in metrics)
    for record in metrics:
        if uniformity_weight:
            assert record["weighted_uniformity_loss"] == pytest.approx(
                uniformity_weight * record["uniformity_loss"], abs=1e-7
            )
            assert record["loss"] == pytest.approx(
                record["base_loss"] + record["weighted_uniformity_loss"], abs=1e-6
            )
        else:
            assert "uniformity_loss" not in record
    if objective == "byol":
        expected_scales = [0.1, 0.3, 0.5][resume_step:]
        assert target_scales == pytest.approx([s for s in expected_scales for _ in range(2)])
        assert [record["center_scale"] for record in metrics] == pytest.approx(expected_scales)
    else:
        assert target_scales == []
        assert all("center_scale" not in record for record in metrics)


def test_one_step_smoke_has_student_gradients_no_teacher_gradients_and_ema():
    torch.manual_seed(2)
    student = SentenceBYOL(
        TinyEncoder(), projection_dim=4, projector_hidden_dim=16, predictor_hidden_dim=12
    )
    teacher = copy.deepcopy(student).eval().requires_grad_(False)
    objective = BYOLLoss(embedding_dim=8)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        "attention_mask": torch.ones(3, 3, dtype=torch.long),
    }
    teacher_before = next(teacher.parameters()).detach().clone()

    student1 = student(**batch, use_dropout=True)
    student2 = student(**batch, use_dropout=True)
    with torch.no_grad():
        teacher1 = teacher(**batch, use_dropout=True, target=True, center=objective.center)
        teacher2 = teacher(**batch, use_dropout=True, target=True, center=objective.center)
    loss, _ = objective(
        (student1.prediction, student2.prediction),
        (teacher1.projection, teacher2.projection),
    )
    loss.backward()
    assert any(parameter.grad is not None for parameter in student.parameters())
    assert all(parameter.grad is None for parameter in teacher.parameters())
    optimizer.step()
    update_teacher(student, teacher, momentum=0.9)

    student_after = next(student.parameters()).detach()
    teacher_after = next(teacher.parameters()).detach()
    expected = teacher_before * 0.9 + student_after * 0.1
    assert torch.allclose(teacher_after, expected)
    assert torch.isfinite(loss)


def test_teacher_momentum_cosine_schedule_reaches_one():
    values = [cosine_teacher_momentum(step, 5, 0.996) for step in range(5)]
    assert values[0] == 0.996
    assert values[-1] == 1.0
    assert values == sorted(values)


@pytest.mark.parametrize(
    "start,end,expected",
    [(0.0, 1.0, [0.0, 0.1464466094, 0.5, 0.8535533906, 1.0]),
     (1.0, 0.0, [1.0, 0.8535533906, 0.5, 0.1464466094, 0.0]),
     (0.2, 0.2, [0.2, 0.2, 0.2, 0.2, 0.2])],
)
def test_cosine_center_scale(start, end, expected):
    assert [cosine_center_scale(t, 5, start, end) for t in range(5)] == pytest.approx(expected)
    assert cosine_center_scale(0, 1, start, end) == start


@pytest.mark.parametrize(
    "overrides,expected",
    [
        (["objective.center_scale=0.25"], (0.25, 0.25)),
        (["objective.center_scale_start=0.1"], (0.1, 0.1)),
        (["objective.center_scale=0.25", "objective.center_scale_end=0.5"], (0.25, 0.5)),
        (["objective.center_scale_start=0.5", "objective.center_scale_end=0.0"], (0.5, 0.0)),
    ],
)
def test_center_scale_schedule_config(overrides, expected):
    from minimal_dino.config import to_train_args

    with initialize_config_module(version_base="1.3", config_module="minimal_dino.conf"):
        args = to_train_args(compose(
            config_name="config",
            overrides=["objective.center_scale_start=null", "objective.center_scale_end=null"]
            + overrides,
        ))
    assert (args.center_scale_start, args.center_scale_end) == expected


@pytest.mark.parametrize("field", ["center_scale", "center_scale_start", "center_scale_end"])
@pytest.mark.parametrize("value", ["-0.1", "true", "bad", ".nan", ".inf"])
def test_center_scale_config_rejects_invalid_values(field, value):
    from minimal_dino.config import to_train_args

    with initialize_config_module(version_base="1.3", config_module="minimal_dino.conf"):
        config = compose(config_name="config", overrides=[f"objective.{field}={value}"])
    with pytest.raises(ValueError, match=f"objective.{field}"):
        to_train_args(config)


@pytest.mark.parametrize("objective", ["byol", "infonce"])
@pytest.mark.parametrize("field", ["uniformity_weight", "uniformity_t"])
@pytest.mark.parametrize("value", ["-0.1", "true", "bad", ".nan", ".inf"])
def test_uniformity_config_rejects_invalid_values(objective, field, value):
    from minimal_dino.config import to_train_args

    with initialize_config_module(version_base="1.3", config_module="minimal_dino.conf"):
        config = compose(
            config_name="config", overrides=[f"objective={objective}", f"objective.{field}={value}"]
        )
    with pytest.raises(ValueError, match=f"objective.{field}"):
        to_train_args(config)


@pytest.mark.parametrize("objective", ["byol", "infonce"])
def test_uniformity_config_defaults_and_zero_t(objective):
    from minimal_dino.config import to_train_args

    with initialize_config_module(version_base="1.3", config_module="minimal_dino.conf"):
        config = compose(config_name="config", overrides=[f"objective={objective}"])
    args = to_train_args(config)
    assert args.uniformity_weight == config.objective.uniformity_weight
    assert args.uniformity_t == 2.0
    assert args.uniformity_mode == "normalized_mean"
    del config.objective.uniformity_weight
    del config.objective.uniformity_mode
    legacy_args = to_train_args(config)
    assert legacy_args.uniformity_weight == 0.0
    assert legacy_args.uniformity_mode == "normalized_mean"
    config.objective.uniformity_t = 0.0
    with pytest.raises(ValueError, match="objective.uniformity_t"):
        to_train_args(config)


def test_encoder_can_be_frozen_without_freezing_byol_head():
    model = SentenceBYOL(
        TinyEncoder(), projection_dim=4, projector_hidden_dim=16, predictor_hidden_dim=12
    )

    set_encoder_trainable(model, False)

    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.head.parameters())

    set_encoder_trainable(model, True)
    assert all(parameter.requires_grad for parameter in model.encoder.parameters())


class TinyTokenizer:
    def save_pretrained(self, path):
        path.mkdir(parents=True, exist_ok=True)


@pytest.mark.parametrize("pooling", ["mean", "cls"])
def test_checkpoint_round_trip_restores_models_optimizer_objective_and_rng(
    tmp_path, pooling
):
    torch.manual_seed(7)
    student = SentenceBYOL(
        TinyEncoder(),
        projection_dim=4,
        projector_hidden_dim=16,
        predictor_hidden_dim=12,
        pooling=pooling,
    )
    teacher = copy.deepcopy(student).eval().requires_grad_(False)
    objective = BYOLLoss(embedding_dim=8)
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
        SimpleNamespace(seed=7, objective="byol"),
        step=3,
        checkpoint_name="checkpoint-step-3.pt",
    )
    expected_random = torch.rand(4)

    with torch.no_grad():
        next(student.parameters()).add_(10)
        objective.center.fill_(10)
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


def test_load_max_logged_metric_uses_all_valid_previous_evaluations(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        "\n".join(
            [
                json.dumps({"step": 0, "sts_spearman": 0.3}),
                json.dumps({"step": 100, "sts_spearman": 0.7}),
                "not-json",
                json.dumps({"step": 200, "sts_spearman": 0.6}),
            ]
        )
        + "\n"
    )

    assert load_max_logged_metric(tmp_path, "sts_spearman") == 0.7
    assert load_max_logged_metric(tmp_path, "missing") is None


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
    commands = []
    outputs = iter(
        [
            str(tmp_path) + "\n",
            "0123456789abcdef0123456789abcdef01234567\n",
            " M src/minimal_dino/train.py\n?? notes.txt\n",
            "diff --git a/file b/file\n",
            "notes.txt\0",
            (
                "diff --git a/notes.txt b/notes.txt\n"
                "new file mode 100644\n"
                "--- /dev/null\n"
                "+++ b/notes.txt\n"
            ),
        ]
    )

    def fake_run(command, **kwargs):
        commands.append(command)
        returncode = 1 if "--no-index" in command else 0
        return subprocess.CompletedProcess(command, returncode, stdout=next(outputs), stderr="")

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
    assert (tmp_path / "git.diff").read_text() == (
        "diff --git a/file b/file\n"
        "diff --git a/notes.txt b/notes.txt\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/notes.txt\n"
    )
    assert ["git", "ls-files", "--others", "--exclude-standard", "-z"] in commands
    assert [
        "git",
        "diff",
        "--binary",
        "--no-index",
        "--",
        "/dev/null",
        "notes.txt",
    ] in commands


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
        scaled_center_config = compose(
            config_name="config",
            overrides=["data.train_file=train.txt", "objective.center_scale=0.25"],
        )
        alternate_config = compose(
            config_name="config",
            overrides=[
                "data.train_file=train.txt",
                "objective=infonce",
                "objective.temperature=0.2",
                "augmentation=word",
                "model.random_init=true",
                "model.pooling=cls",
                "model.projection_dim=64",
                "logging.quiet=true",
                "logging.tensorboard=false",
                "runtime.device=cpu",
            ],
        )

    default_args = to_train_args(default_config)
    scaled_center_args = to_train_args(scaled_center_config)
    alternate_args = to_train_args(alternate_config)

    assert default_args.objective == "byol"
    assert default_args.augmentation == list(default_config.augmentation.names)
    assert default_args.quiet is True
    assert default_args.tensorboard is True
    assert default_args.byol_precision == "bf16"
    assert default_args.random_init is False
    assert default_args.pooling == default_config.model.pooling
    assert default_args.projection_dim == default_config.model.projection_dim
    assert default_args.projector_hidden_dim == 4096
    assert default_args.predictor_hidden_dim == 4096
    assert default_args.target_dropout == 0.02
    assert default_args.center_momentum == default_config.objective.center_momentum
    assert default_args.center_scale == default_config.objective.center_scale
    assert scaled_center_args.center_scale == 0.25
    assert default_args.teacher_momentum == default_config.teacher.momentum
    assert default_args.batch_size == default_config.optimization.batch_size
    assert default_args.encoder_learning_rate == default_config.optimization.encoder_learning_rate
    assert default_args.head_learning_rate == default_config.optimization.head_learning_rate
    assert default_args.encoder_freeze_steps == default_config.optimization.encoder_freeze_steps
    assert default_args.max_length == 256
    assert default_args.num_workers == default_config.data.num_workers
    assert default_args.max_steps == default_config.optimization.get("max_steps")
    assert default_args.eval_steps == default_config.evaluation.steps
    assert alternate_args.objective == "infonce"
    assert alternate_args.infonce_temp == 0.2
    assert alternate_args.augmentation == ["word"]
    assert alternate_args.random_init is True
    assert alternate_args.pooling == "cls"
    assert alternate_args.projection_dim == 64
    assert alternate_args.center_scale == 0.5  # InfoNCE has no center setting.
    assert alternate_args.quiet is True
    assert alternate_args.tensorboard is False
