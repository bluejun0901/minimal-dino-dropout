from __future__ import annotations

import argparse
import copy
import json
import math
import random
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from minimal_dino.data import TextLineDataset, TokenizeCollator
from minimal_dino.evaluation import evaluate_stsb
from minimal_dino.model import SentenceDINO
from minimal_dino.objective import DINOLoss


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


def teacher_temperature(
    step: int, warmup_steps: int, warmup_temperature: float, temperature: float
) -> float:
    if warmup_steps <= 0 or step >= warmup_steps:
        return temperature
    alpha = step / max(1, warmup_steps - 1)
    return warmup_temperature + alpha * (temperature - warmup_temperature)


@torch.no_grad()
def update_teacher(student: SentenceDINO, teacher: SentenceDINO, momentum: float) -> None:
    student_parameters = dict(student.named_parameters())
    teacher_parameters = dict(teacher.named_parameters())
    if student_parameters.keys() != teacher_parameters.keys():
        raise RuntimeError("Student and teacher parameters do not match")
    for name, teacher_parameter in teacher_parameters.items():
        teacher_parameter.mul_(momentum).add_(
            student_parameters[name].detach(), alpha=1.0 - momentum
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
    student: SentenceDINO,
    teacher: SentenceDINO,
    objective: DINOLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    step: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir / "tokenizer")
    head_config = {
        "output_dim": student.head.last_weight.shape[0],
        "head_hidden_dim": student.head.mlp[0].out_features,
        "bottleneck_dim": student.head.last_weight.shape[1],
    }
    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save(
        {
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "objective": objective.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "encoder_config": student.encoder.config.to_dict(),
            "head_config": head_config,
            "args": vars(args),
            "step": step,
        },
        checkpoint_path,
    )
    return checkpoint_path


def train(args: argparse.Namespace) -> Path:
    set_seed(args.seed)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    student = SentenceDINO.from_pretrained(
        args.model_name,
        output_dim=args.output_dim,
        head_hidden_dim=args.head_hidden_dim,
        bottleneck_dim=args.bottleneck_dim,
    ).to(device)
    teacher = copy.deepcopy(student).to(device).eval()
    teacher.requires_grad_(False)
    objective = DINOLoss(args.output_dim, args.student_temp, args.center_momentum).to(device)

    if not all(torch.equal(a, b) for a, b in zip(student.parameters(), teacher.parameters())):
        raise RuntimeError("Teacher was not initialized exactly from the student")
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise RuntimeError("Teacher parameters must not require gradients")

    dataset = TextLineDataset(args.train_file)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=TokenizeCollator(tokenizer, args.max_length),
        num_workers=args.num_workers,
        generator=generator,
        drop_last=False,
    )
    steps_per_epoch = len(loader)
    requested_steps = args.epochs * steps_per_epoch
    total_steps = min(requested_steps, args.max_steps) if args.max_steps else requested_steps
    if total_steps < 1:
        raise ValueError("Training requires at least one optimizer step")

    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    global_step = 0
    dropout_warning_emitted = False
    collapse_warning_emitted = False
    student.train()
    optimizer.zero_grad(set_to_none=True)
    for _epoch in range(args.epochs):
        for batch in loader:
            batch = {name: value.to(device) for name, value in batch.items()}

            student_view1 = student(**batch, use_dropout=True)
            student_view2 = student(**batch, use_dropout=True)
            with torch.no_grad():
                teacher_view1 = teacher(**batch, use_dropout=True)
                teacher_view2 = teacher(**batch, use_dropout=True)

            temperature = teacher_temperature(
                global_step,
                args.teacher_temp_warmup_steps,
                args.warmup_teacher_temp,
                args.teacher_temp,
            )
            loss, loss_metrics = objective(
                (student_view1.logits, student_view2.logits),
                (teacher_view1.logits, teacher_view2.logits),
                temperature,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {global_step}: {loss.item()}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            momentum = cosine_teacher_momentum(global_step, total_steps, args.teacher_momentum)
            update_teacher(student, teacher, momentum)
            objective.update_center((teacher_view1.logits, teacher_view2.logits))
            optimizer.zero_grad(set_to_none=True)

            student_view_cosine = F.cosine_similarity(
                student_view1.embedding.detach(), student_view2.embedding.detach()
            ).mean()
            teacher_view_cosine = F.cosine_similarity(
                teacher_view1.embedding, teacher_view2.embedding
            ).mean()
            pairwise_cosine = off_diagonal_cosine(student_view1.embedding.detach())
            embedding_std = student_view1.embedding.detach().std(dim=0, unbiased=False).mean()

            if not dropout_warning_emitted and (
                torch.equal(student_view1.embedding, student_view2.embedding)
                or torch.equal(teacher_view1.embedding, teacher_view2.embedding)
            ):
                warnings.warn("Dropout views are identical; check that encoder dropout is nonzero.")
                dropout_warning_emitted = True
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
            if global_step == 1 or global_step % args.log_steps == 0:
                log = {
                    "step": global_step,
                    "loss": loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "teacher_momentum": momentum,
                    "teacher_temp": temperature,
                    "grad_norm": float(grad_norm),
                    "student_view_cosine": student_view_cosine.item(),
                    "teacher_view_cosine": teacher_view_cosine.item(),
                    "embedding_std": embedding_std.item(),
                    "pairwise_cosine": pairwise_cosine.item(),
                    **{name: value.item() for name, value in loss_metrics.items()},
                }
                print(json.dumps(log, sort_keys=True), flush=True)

            if args.eval_steps and global_step % args.eval_steps == 0:
                metrics = evaluate_stsb(
                    teacher,
                    tokenizer,
                    device=device,
                    split="validation",
                    batch_size=args.eval_batch_size,
                    max_length=args.max_length,
                    limit=args.eval_limit,
                )
                print(json.dumps({"step": global_step, **metrics}, sort_keys=True), flush=True)
                teacher.eval()
                student.train()

            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break

    if any(parameter.grad is not None for parameter in teacher.parameters()):
        raise RuntimeError("A gradient unexpectedly reached the teacher")
    return save_checkpoint(
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train BERT-base with DINO and dropout views")
    parser.add_argument("--train-file", required=True, help="UTF-8 text, one sentence per line")
    parser.add_argument("--output-dir", default="runs/minimal-dino")
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0, help="0 uses all epoch steps")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--output-dim", type=int, default=65_536)
    parser.add_argument("--head-hidden-dim", type=int, default=2_048)
    parser.add_argument("--bottleneck-dim", type=int, default=256)
    parser.add_argument("--student-temp", type=float, default=0.1)
    parser.add_argument("--teacher-temp", type=float, default=0.04)
    parser.add_argument("--warmup-teacher-temp", type=float, default=0.04)
    parser.add_argument("--teacher-temp-warmup-steps", type=int, default=0)
    parser.add_argument("--teacher-momentum", type=float, default=0.996)
    parser.add_argument("--center-momentum", type=float, default=0.9)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=250, help="0 disables STS evaluation")
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--eval-limit", type=int)
    parser.add_argument("--collapse-warning-after", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    checkpoint = train(args)
    print(f"Saved checkpoint to {checkpoint}")


if __name__ == "__main__":
    main()
