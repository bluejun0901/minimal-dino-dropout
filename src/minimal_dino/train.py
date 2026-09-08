from __future__ import annotations

import copy
import json
import logging
import math
import random
import re
import subprocess
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from minimal_dino.config import config_to_container, to_train_args
from minimal_dino.data import TextLineDataset, TokenizeCollator, WordViewCollator
from minimal_dino.evaluation import (
    encode_stsb_dataset,
    load_stsb_split,
    stsb_metrics,
)
from minimal_dino.model import BYOLOutput, SentenceBYOL, checkpoint_model_config, model_config
from minimal_dino.objective import BYOLLoss, InfoNCELoss, build_objective

DEFAULT_MODEL_NAME = "bert-base-uncased"
DEFAULT_MODEL_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"

logging.getLogger("httpx").setLevel(logging.WARNING)


def _write_text_atomically(path: Path, content: str) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)


def _run_git(
    arguments: list[str],
    cwd: Path,
    *,
    allowed_returncodes: tuple[int, ...] = (0,),
) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode not in allowed_returncodes:
        raise subprocess.CalledProcessError(
            result.returncode,
            result.args,
            output=result.stdout,
            stderr=result.stderr,
        )
    return result.stdout


def _git_diff_including_untracked(repository_root: Path) -> str:
    """Return an applicable binary patch for all tracked and untracked changes."""
    diff = _run_git(["diff", "--binary", "HEAD", "--"], repository_root)
    untracked_output = _run_git(
        ["ls-files", "--others", "--exclude-standard", "-z"], repository_root
    )
    for relative_path in untracked_output.split("\0"):
        if not relative_path:
            continue
        # `git diff --no-index` returns 1 when it finds the expected difference.
        diff += _run_git(
            ["diff", "--binary", "--no-index", "--", "/dev/null", relative_path],
            repository_root,
            allowed_returncodes=(0, 1),
        )
    return diff


def save_run_artifacts(
    output_dir: str | Path,
    args: SimpleNamespace,
    git_cwd: str | Path | None = None,
) -> None:
    """Save the resolved arguments and source-tree state needed to reproduce a run."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    config = json.dumps(config_to_container(args), indent=2, sort_keys=True, default=str) + "\n"

    diff = ""
    try:
        repository_root = Path(
            _run_git(["rev-parse", "--show-toplevel"], Path(git_cwd or Path.cwd())).strip()
        )
        commit_hash = _run_git(["rev-parse", "HEAD"], repository_root).strip()
        status = _run_git(
            ["status", "--short", "--untracked-files=all"], repository_root
        ).rstrip()
        # Comparing tracked files against HEAD includes staged and unstaged changes.
        # New files need explicit no-index patches because regular Git diffs omit them.
        diff = _git_diff_including_untracked(repository_root)
        git_state: dict[str, Any] = {
            "available": True,
            "commit_hash": commit_hash,
            "dirty": bool(status),
            "repository_root": str(repository_root),
            "status": status,
            "diff_file": "git.diff",
        }
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        git_state = {
            "available": False,
            "error": str(error),
            "diff_file": "git.diff",
        }

    # Capture Git before writing files so an unignored output directory does not dirty itself.
    _write_text_atomically(output_path / "config.json", config)
    _write_text_atomically(
        output_path / "git_state.json",
        json.dumps(git_state, indent=2, sort_keys=True) + "\n",
    )
    _write_text_atomically(output_path / "git.diff", diff)


def log_metrics(
    output_dir: str | Path,
    metrics: dict[str, float | int],
    *,
    quiet: bool = False,
    tensorboard_writer: SummaryWriter | None = None,
    namespace: str | None = None,
) -> None:
    """Emit one metrics record to stdout, JSONL, and optionally TensorBoard."""
    line = json.dumps(metrics, sort_keys=True)
    if not quiet:
        print(line, flush=True)
    path = Path(output_dir) / "metrics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
        stream.flush()
    if tensorboard_writer is not None:
        step = int(metrics["step"])
        for name, value in metrics.items():
            if name != "step":
                tag = f"{namespace}/{name}" if namespace else name
                tensorboard_writer.add_scalar(tag, value, step)
        tensorboard_writer.flush()


def load_max_logged_metric(output_dir: str | Path, name: str) -> float | None:
    """Return the largest finite metric value already recorded for a run."""
    path = Path(output_dir) / "metrics.jsonl"
    if not path.is_file():
        return None
    maximum = None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                value = json.loads(line).get(name)
            except (json.JSONDecodeError, AttributeError):
                continue
            if isinstance(value, (int, float)) and math.isfinite(value):
                maximum = value if maximum is None else max(maximum, value)
    return maximum


def _print_progress(step: int, total_steps: int, width: int = 30) -> None:
    filled = int(width * step / total_steps)
    bar = "#" * filled + "-" * (width - filled)
    print(
        f"\rTraining [{bar}] {step}/{total_steps}",
        end="\n" if step >= total_steps else "",
        flush=True,
    )


def resolve_model_revision(model_name: str, revision: str | None) -> str | None:
    """Require immutable revisions for Hub models while allowing local directories."""
    if revision:
        if re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None:
            raise ValueError("model.revision must be a full 40-character commit hash")
        return revision
    if Path(model_name).exists():
        return None
    if model_name in {DEFAULT_MODEL_NAME, "google-bert/bert-base-uncased"}:
        return DEFAULT_MODEL_REVISION
    raise ValueError("model.revision is required when model.name refers to a Hub model")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_teacher_momentum(step: int, total_steps: int, base_momentum: float) -> float:
    if total_steps <= 1:
        return base_momentum
    progress = step / (total_steps - 1)
    return 1.0 - (1.0 - base_momentum) * (math.cos(math.pi * progress) + 1.0) / 2.0


def set_encoder_trainable(model: SentenceBYOL, trainable: bool) -> None:
    """Freeze or unfreeze only the online encoder, leaving the BYOL head trainable."""
    model.encoder.requires_grad_(trainable)


@torch.no_grad()
def update_teacher(
    student: SentenceBYOL,
    teacher: SentenceBYOL,
    momentum: float,
    *,
    use_foreach: bool = False,
) -> None:
    student_parameters = tuple(student.named_parameters())
    teacher_parameters = tuple(teacher.named_parameters())
    if tuple(name for name, _ in student_parameters) != tuple(
        name for name, _ in teacher_parameters
    ):
        raise RuntimeError("Student and teacher parameters do not match")
    teacher_tensors = [parameter for _, parameter in teacher_parameters]
    student_tensors = [parameter.detach() for _, parameter in student_parameters]
    if use_foreach:
        torch._foreach_lerp_(teacher_tensors, student_tensors, 1.0 - momentum)
    else:
        for teacher_parameter, student_parameter in zip(teacher_tensors, student_tensors):
            teacher_parameter.mul_(momentum).add_(
                student_parameter, alpha=1.0 - momentum
            )

    # BERT's non-trainable buffers (for example position ids) are copied exactly.
    student_buffers = dict(student.named_buffers())
    teacher_buffers = dict(teacher.named_buffers())
    for name, teacher_buffer in teacher_buffers.items():
        if name not in student_buffers:
            raise RuntimeError(f"Teacher buffer has no student counterpart: {name}")
        teacher_buffer.copy_(student_buffers[name])


def off_diagonal_cosine(embeddings: torch.Tensor) -> torch.Tensor:
    if embeddings.shape[0] < 2:
        return embeddings.new_tensor(float("nan"))
    normalized = F.normalize(embeddings, dim=-1)
    similarities = normalized @ normalized.T
    indices = torch.triu_indices(embeddings.shape[0], embeddings.shape[0], offset=1)
    return similarities[indices[0], indices[1]].mean()


def save_checkpoint(
    output_dir: Path,
    student: SentenceBYOL,
    teacher: SentenceBYOL,
    objective: BYOLLoss | InfoNCELoss,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    tokenizer: Any,
    args: SimpleNamespace,
    step: int,
    checkpoint_name: str = "checkpoint.pt",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir / "tokenizer")
    head_config = model_config(student)
    checkpoint_path = output_dir / checkpoint_name
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(
        {
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "objective_name": args.objective,
            "objective": objective.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "encoder_config": student.encoder.config.to_dict(),
            "head_config": head_config,
            "pooling": student.pooling,
            "args": vars(args),
            "step": step,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_state": (
                torch.cuda.get_rng_state(next(student.parameters()).device)
                if next(student.parameters()).device.type == "cuda"
                else None
            ),
        },
        temporary_path,
    )
    temporary_path.replace(checkpoint_path)
    return checkpoint_path


def restore_checkpoint(
    checkpoint_path: str | Path,
    student: SentenceBYOL,
    teacher: SentenceBYOL,
    objective: BYOLLoss | InfoNCELoss,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
) -> int:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_objective = checkpoint["objective_name"]
    if checkpoint_objective != objective_name(objective):
        raise ValueError(
            f"Checkpoint objective is {checkpoint_objective}, but this run uses "
            f"{objective_name(objective)}"
        )
    checkpoint_head_config = checkpoint_model_config(checkpoint)
    expected_head_config = model_config(student)
    if checkpoint_head_config != expected_head_config:
        raise ValueError("Checkpoint projection-head configuration does not match this run")
    student.load_state_dict(checkpoint["student"])
    teacher.load_state_dict(checkpoint["teacher"])
    objective.load_state_dict(checkpoint["objective"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    random.setstate(checkpoint["python_random_state"])
    np.random.set_state(checkpoint["numpy_random_state"])
    torch.set_rng_state(checkpoint["torch_random_state"].cpu())
    if device.type == "cuda" and checkpoint["cuda_random_state"] is not None:
        torch.cuda.set_rng_state(checkpoint["cuda_random_state"].cpu(), device)
    return int(checkpoint["step"])


def objective_name(objective: BYOLLoss | InfoNCELoss) -> str:
    if isinstance(objective, BYOLLoss):
        return "byol"
    if isinstance(objective, InfoNCELoss):
        return "infonce"
    raise TypeError(f"Unsupported objective type: {type(objective).__name__}")


def remove_old_periodic_checkpoints(output_dir: Path, keep_last: int) -> None:
    checkpoints = sorted(
        output_dir.glob("checkpoint-step-*.pt"),
        key=lambda path: int(path.stem.rsplit("-", maxsplit=1)[1]),
    )
    for checkpoint in checkpoints[:-keep_last]:
        checkpoint.unlink()


def train(
    args: SimpleNamespace,
    run_config: Any | None = None,
    evaluation_callback: Callable[[int, dict[str, float]], None] | None = None,
    save_final_checkpoint: bool = True,
) -> Path | None:
    torch.set_float32_matmul_precision("high")
    set_seed(args.seed)

    if not 0.0 <= args.augmentation_strength <= 1.0:
        raise ValueError("augmentation_strength must be in [0, 1]")
    use_word_augmentation = "word" in args.augmentation
    use_dropout = "dropout" in args.augmentation
    device = torch.device(args.device)
    args.model_revision = resolve_model_revision(args.model_name, args.model_revision)
    save_run_artifacts(args.output_dir, run_config if run_config is not None else args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, revision=args.model_revision)
    model_factory = (
        SentenceBYOL.from_random_init if args.random_init else SentenceBYOL.from_pretrained
    )
    student = model_factory(
        args.model_name,
        revision=args.model_revision,
        dropout=args.dropout,
        pooling=args.pooling,
        projection_dim=args.projection_dim,
        projector_hidden_dim=args.projector_hidden_dim,
        predictor_hidden_dim=args.predictor_hidden_dim,
    ).to(device)
    teacher = copy.deepcopy(student).to(device).eval()
    teacher.requires_grad_(False)
    objective = build_objective(
        args.objective,
        projection_dim=args.projection_dim,
        embedding_dim=student.head.input_dim,
        center_momentum=args.center_momentum,
        infonce_temp=args.infonce_temp,
    ).to(device)
    byol_precision = args.byol_precision
    if byol_precision not in {"fp32", "bf16"}:
        raise ValueError("runtime.byol_precision must be 'fp32' or 'bf16'")
    optimize_byol_cuda = isinstance(objective, BYOLLoss) and device.type == "cuda"
    use_byol_bf16 = optimize_byol_cuda and byol_precision == "bf16"
    if use_byol_bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("runtime.byol_precision=bf16 requires a BF16-capable CUDA device")

    if not all(torch.equal(a, b) for a, b in zip(student.parameters(), teacher.parameters())):
        raise RuntimeError("Teacher was not initialized exactly from the student")
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise RuntimeError("Teacher parameters must not require gradients")

    train_dataset = TextLineDataset(args.train_file)
    evaluation_dataset = (
        load_stsb_split(args.stsb_dir, "validation") if args.eval_steps else None
    )
    generator = torch.Generator().manual_seed(args.seed)
    collator = (
        TokenizeCollator(tokenizer, args.max_length)
        if not use_word_augmentation
        else WordViewCollator(tokenizer, args.max_length, args.augmentation_strength)
    )
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        generator=generator,
        drop_last=False,
        pin_memory=optimize_byol_cuda,
        persistent_workers=(
            isinstance(objective, BYOLLoss)
            and not use_word_augmentation
            and args.num_workers > 0
        ),
    )
    steps_per_epoch = len(loader)
    requested_steps = args.epochs * steps_per_epoch
    total_steps = min(requested_steps, args.max_steps) if args.max_steps else requested_steps
    if total_steps < 1:
        raise ValueError("Training requires at least one optimizer step")

    optimizer_kwargs = {"weight_decay": args.weight_decay}
    if optimize_byol_cuda:
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(
        [
            {
                "params": student.encoder.parameters(),
                "lr": args.encoder_learning_rate,
                "name": "encoder",
            },
            {
                "params": student.head.parameters(),
                "lr": args.head_learning_rate,
                "name": "head",
            },
        ],
        **optimizer_kwargs,
    )
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if args.save_steps < 0:
        raise ValueError("save_steps must be non-negative")
    if args.keep_last_checkpoints < 1:
        raise ValueError("keep_last_checkpoints must be at least 1")

    tensorboard_writer = (
        SummaryWriter(log_dir=Path(args.output_dir) / "tensorboard")
        if getattr(args, "tensorboard", True)
        else None
    )

    step_zero_eval_metrics = None
    max_sts_spearman = load_max_logged_metric(args.output_dir, "sts_spearman")
    if evaluation_dataset is not None:
        embedding1, embedding2, evaluation_scores = encode_stsb_dataset(
            student,
            tokenizer,
            evaluation_dataset,
            device=device,
            batch_size=args.eval_batch_size,
            limit=args.eval_limit,
        )
        step_zero_eval_metrics = stsb_metrics(
            embedding1,
            embedding2,
            evaluation_scores,
        )

    global_step = 0
    if args.resume_from_checkpoint:
        global_step = restore_checkpoint(
            args.resume_from_checkpoint, student, teacher, objective, optimizer, scheduler, device
        )
        if global_step >= total_steps:
            raise ValueError("Checkpoint has already reached the requested total training steps")
        if args.quiet:
            _print_progress(global_step, total_steps)
        else:
            print(f"Resumed from {args.resume_from_checkpoint} at step {global_step}", flush=True)
    elif step_zero_eval_metrics is not None:
        max_sts_spearman = max(
            max_sts_spearman if max_sts_spearman is not None else -math.inf,
            step_zero_eval_metrics["sts_spearman"],
        )
        step_zero_eval_metrics = {
            **step_zero_eval_metrics,
            "max_sts_spearman": max_sts_spearman,
        }
        log_metrics(
            args.output_dir,
            {"step": 0, **step_zero_eval_metrics},
            quiet=args.quiet,
            tensorboard_writer=tensorboard_writer,
            namespace="eval",
        )
        if evaluation_callback is not None:
            evaluation_callback(0, step_zero_eval_metrics)
    dropout_views_checked = False
    collapse_warning_emitted = False
    set_encoder_trainable(student, global_step >= args.encoder_freeze_steps)
    student.train()
    optimizer.zero_grad(set_to_none=True)
    start_epoch = global_step // steps_per_epoch
    resume_batch = global_step % steps_per_epoch
    for epoch in range(start_epoch, args.epochs):
        if use_word_augmentation:
            # Replaying skipped batches after resume reproduces the same augmented views.
            random.seed(args.seed + epoch)
        generator.manual_seed(args.seed + epoch)
        for batch_index, batch in enumerate(loader):
            if epoch == start_epoch and batch_index < resume_batch:
                continue
            if global_step == args.encoder_freeze_steps:
                set_encoder_trainable(student, True)
            completed_step = global_step + 1
            should_log = completed_step == 1 or completed_step % args.log_steps == 0
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_byol_bf16,
            ):
                if not use_word_augmentation:
                    batch = {
                        name: value.to(device, non_blocking=optimize_byol_cuda)
                        for name, value in batch.items()
                    }
                    if isinstance(objective, BYOLLoss):
                        batch_size = next(iter(batch.values())).shape[0]
                        combined_batch = {
                            name: torch.cat((value, value), dim=0) for name, value in batch.items()
                        }
                        combined_student = student(**combined_batch, use_dropout=True)
                        student_view1 = BYOLOutput(
                            embedding=combined_student.embedding[:batch_size],
                            projection=combined_student.projection[:batch_size],
                            prediction=combined_student.prediction[:batch_size],
                        )
                        student_view2 = BYOLOutput(
                            embedding=combined_student.embedding[batch_size:],
                            projection=combined_student.projection[batch_size:],
                            prediction=combined_student.prediction[batch_size:],
                        )
                        # Target views use independent, lower-rate dropout masks.
                        with torch.no_grad():
                            teacher_view1 = teacher(
                                **batch,
                                use_dropout=True,
                                dropout_probability=args.target_dropout,
                                target=True,
                                center=objective.center,
                                center_scale=args.center_scale,
                            )
                            teacher_view2 = teacher(
                                **batch,
                                use_dropout=True,
                                dropout_probability=args.target_dropout,
                                target=True,
                                center=objective.center,
                                center_scale=args.center_scale,
                            )
                    else:
                        # Keep the existing InfoNCE execution path unchanged.
                        student_view1 = student(**batch, use_dropout=True)
                        student_view2 = student(**batch, use_dropout=True)
                        with torch.no_grad():
                            teacher_view1 = teacher(
                                **batch,
                                use_dropout=True,
                                dropout_probability=args.target_dropout,
                                target=True,
                            )
                            teacher_view2 = teacher(
                                **batch,
                                use_dropout=True,
                                dropout_probability=args.target_dropout,
                                target=True,
                            )
                else:
                    view1, view2 = (
                        {
                            name: value.to(device, non_blocking=optimize_byol_cuda)
                            for name, value in view.items()
                        }
                        for view in batch
                    )
                    student_view1 = student(**view1, use_dropout=use_dropout)
                    student_view2 = student(**view2, use_dropout=use_dropout)
                    with torch.no_grad():
                        teacher_view1 = teacher(
                            **view1,
                            use_dropout=use_dropout,
                            dropout_probability=args.target_dropout,
                            target=True,
                            center=objective.center if isinstance(objective, BYOLLoss) else None,
                            center_scale=args.center_scale,
                        )
                        teacher_view2 = teacher(
                            **view2,
                            use_dropout=use_dropout,
                            dropout_probability=args.target_dropout,
                            target=True,
                            center=objective.center if isinstance(objective, BYOLLoss) else None,
                            center_scale=args.center_scale,
                        )

                if isinstance(objective, BYOLLoss):
                    if student_view1.prediction is None or student_view2.prediction is None:
                        raise RuntimeError("Online BYOL views must contain predictions")
                    loss, loss_metrics = objective(
                        (student_view1.prediction, student_view2.prediction),
                        (teacher_view1.projection, teacher_view2.projection),
                        compute_metrics=should_log,
                    )
                else:
                    loss, loss_metrics = objective(
                        (student_view1.embedding, student_view2.embedding)
                    )
            if optimize_byol_cuda:
                torch._assert_async(
                    torch.isfinite(loss), f"Non-finite loss at step {global_step}"
                )
            elif not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {global_step}: {loss.item()}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            momentum = cosine_teacher_momentum(
                global_step, total_steps, args.teacher_momentum
            )
            update_teacher(
                student,
                teacher,
                momentum,
                use_foreach=isinstance(objective, BYOLLoss),
            )
            if isinstance(objective, BYOLLoss):
                objective.update_center(
                    (teacher_view1.embedding, teacher_view2.embedding)
                )
            optimizer.zero_grad(set_to_none=True)

            if use_dropout and not use_word_augmentation and not dropout_views_checked:
                if torch.equal(student_view1.embedding, student_view2.embedding):
                    warnings.warn(
                        "Dropout views are identical; check that encoder dropout is nonzero."
                    )
                dropout_views_checked = True

            if should_log:
                student_view_cosine = F.cosine_similarity(
                    student_view1.embedding.detach(), student_view2.embedding.detach()
                ).mean()
                teacher_view_cosine = F.cosine_similarity(
                    teacher_view1.embedding, teacher_view2.embedding
                ).mean()
                pairwise_cosine = off_diagonal_cosine(student_view1.embedding.detach())
                embedding_std = (
                    student_view1.embedding.detach().float().std(dim=0, unbiased=False).mean()
                )
                if (
                    not collapse_warning_emitted
                    and global_step >= args.collapse_warning_after
                    and (
                        embedding_std < 1e-3
                        or (torch.isfinite(pairwise_cosine) and pairwise_cosine > 0.99)
                    )
                ):
                    warnings.warn(
                        "Sentence embeddings show a possible collapse; inspect logged diagnostics."
                    )
                    collapse_warning_emitted = True

            global_step += 1
            if args.quiet and (should_log or global_step == total_steps):
                _print_progress(global_step, total_steps)
            if should_log:
                log = {
                    "step": global_step,
                    "loss": loss.item(),
                    "encoder_lr": scheduler.get_last_lr()[0],
                    "head_lr": scheduler.get_last_lr()[1],
                    "encoder_frozen": int(global_step <= args.encoder_freeze_steps),
                    "teacher_momentum": momentum,
                    "grad_norm": float(grad_norm),
                    "student_view_cosine": student_view_cosine.item(),
                    "teacher_view_cosine": teacher_view_cosine.item(),
                    "embedding_std": embedding_std.item(),
                    "pairwise_cosine": pairwise_cosine.item(),
                    **{name: value.item() for name, value in loss_metrics.items()},
                }
                log_metrics(
                    args.output_dir,
                    log,
                    quiet=args.quiet,
                    tensorboard_writer=tensorboard_writer,
                    namespace="train",
                )

            if args.eval_steps and global_step % args.eval_steps == 0:
                if evaluation_dataset is None:
                    raise RuntimeError("Evaluation data was not initialized")
                embedding1, embedding2, evaluation_scores = encode_stsb_dataset(
                    student,
                    tokenizer,
                    evaluation_dataset,
                    device=device,
                    batch_size=args.eval_batch_size,
                    limit=args.eval_limit,
                )
                metrics = stsb_metrics(
                    embedding1,
                    embedding2,
                    evaluation_scores,
                )
                max_sts_spearman = max(
                    max_sts_spearman if max_sts_spearman is not None else -math.inf,
                    metrics["sts_spearman"],
                )
                metrics = {**metrics, "max_sts_spearman": max_sts_spearman}
                log_metrics(
                    args.output_dir,
                    {"step": global_step, **metrics},
                    quiet=args.quiet,
                    tensorboard_writer=tensorboard_writer,
                    namespace="eval",
                )
                if evaluation_callback is not None:
                    evaluation_callback(global_step, metrics)
                student.train()

            if args.save_steps and global_step % args.save_steps == 0:
                periodic_path = save_checkpoint(
                    Path(args.output_dir),
                    student,
                    teacher,
                    objective,
                    optimizer,
                    scheduler,
                    tokenizer,
                    args,
                    global_step,
                    checkpoint_name=f"checkpoint-step-{global_step}.pt",
                )
                remove_old_periodic_checkpoints(Path(args.output_dir), args.keep_last_checkpoints)
                if not args.quiet:
                    print(f"Saved periodic checkpoint to {periodic_path}", flush=True)
            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break

    if any(parameter.grad is not None for parameter in teacher.parameters()):
        raise RuntimeError("A gradient unexpectedly reached the teacher")
    checkpoint = None
    if save_final_checkpoint:
        checkpoint = save_checkpoint(
            Path(args.output_dir),
            student,
            teacher,
            objective,
            optimizer,
            scheduler,
            tokenizer,
            args,
            global_step,
        )
    if tensorboard_writer is not None:
        tensorboard_writer.close()
    return checkpoint


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    args = to_train_args(config)
    checkpoint = train(args, run_config=config)
    if checkpoint is not None and not args.quiet:
        print(f"Saved checkpoint to {checkpoint}")


if __name__ == "__main__":
    main()
