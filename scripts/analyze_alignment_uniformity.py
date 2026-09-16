"""Audit historical logs and probe global geometry without fitting to STS labels.

Run from the repository root with the project venv activated. Outputs are isolated
from historical runs. The centering sweep is a diagnostic, not SSL training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import torch
from scipy.stats import spearmanr
from torch.nn import functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from minimal_dino.data import TextLineDataset, TokenizeCollator
from minimal_dino.evaluation import load_checkpoint, load_stsb_split, stsb_metrics
from minimal_dino.objective import DecoupledUniformityLoss, UniformityLoss


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(output: Path) -> None:
    paths = [
        Path("runs/re"),
        *sorted(Path("runs").glob("some_uniformity*")),
        Path("archive/check_trajectory_infonce_fin"),
    ]
    records = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for path in paths:
        raw = (path / "metrics.jsonl").read_bytes()
        segments: list[list[dict]] = [[]]
        for line in raw.splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # A concurrently written trailing line may be incomplete.
            if "sts_spearman" not in row:
                continue
            if segments[-1] and row["step"] <= segments[-1][-1]["step"]:
                segments.append([])
            row["relative_alignment"] = row["alignment"] / (2 * (1 - row["pairwise_cosine_mean"]))
            segments[-1].append(row)
        records.append(
            {
                "path": str(path),
                "metrics_sha256": hashlib.sha256(raw).hexdigest(),
                "config": json.loads((path / "config.json").read_text()),
                "git_state": json.loads((path / "git_state.json").read_text()),
                "segments": segments,
            }
        )
        for index, segment in enumerate(segments):
            if not segment:
                continue
            label = path.name + (f" #{index + 1}" if len(segments) > 1 else "")
            for ax, metric in zip(axes, ["alignment", "relative_alignment", "sts_spearman"]):
                ax.plot(
                    [r["uniformity"] for r in segment],
                    [r[metric] for r in segment],
                    marker=".",
                    label=label,
                )
                ax.set(xlabel="Uniformity (lower is more uniform)", ylabel=metric)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "trajectories.png", dpi=180)
    plt.close(fig)
    (output / "audit.json").write_text(
        json.dumps(
            {"snapshot_utc": datetime.now(timezone.utc).isoformat(), "runs": records}, indent=2
        )
    )


@torch.inference_mode()
def encode(model, tokenizer, sentences, *, max_length, batch_size=32):
    collate = TokenizeCollator(tokenizer, max_length=max_length)
    chunks = []
    for start in range(0, len(sentences), batch_size):
        chunks.append(model.encode(**collate(sentences[start : start + batch_size])).float())
        if start % (batch_size * 8) == 0:
            print(f"encoded {min(start + batch_size, len(sentences))}/{len(sentences)}", flush=True)
    return torch.cat(chunks)


def probe(output: Path, checkpoint: Path, wiki_count: int) -> None:
    started = time.monotonic()
    cache_path = output / "features.pt"
    manifest_path = output / "probe_manifest.json"
    identity = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "wiki_count": wiki_count,
        "seed": 42,
        "wiki_sha256": sha256(Path("data/wiki1m_for_simcse.txt")),
        "validation_sha256": sha256(Path("data/stsb/validation.parquet")),
    }
    if cache_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if any(manifest[k] != v for k, v in identity.items()):
            raise ValueError("Cache provenance differs; use a new output directory")
        features = torch.load(cache_path, map_location="cpu", weights_only=True)
    else:
        model, tokenizer = load_checkpoint(checkpoint, torch.device("cpu"))
        wiki = TextLineDataset("data/wiki1m_for_simcse.txt")
        indices = random.Random(42).sample(range(len(wiki)), wiki_count)
        print("Encoding unlabeled Wiki calibration sentences", flush=True)
        train = encode(model, tokenizer, [wiki[i] for i in indices], max_length=256)
        dataset = load_stsb_split("data/stsb", "validation")
        sentences = list(dataset["sentence1"]) + list(dataset["sentence2"])
        unique = list(dict.fromkeys(sentences))
        lookup = {sentence: i for i, sentence in enumerate(unique)}
        print("Encoding full STS-B validation, no truncation", flush=True)
        encoded = encode(model, tokenizer, unique, max_length=None)
        embeddings = encoded[[lookup[s] for s in sentences]]
        features = {
            "wiki": train,
            "first": embeddings[: len(dataset)],
            "second": embeddings[len(dataset) :],
            "scores": torch.tensor(dataset["score"]),
        }
        torch.save(features, cache_path)
        manifest = {
            **identity,
            "wiki_indices": indices,
            "wiki_max_length": 256,
            "eval_max_length": None,
            "torch": torch.__version__,
            "device": "cpu",
            "threads": torch.get_num_threads(),
            "kind": "frozen_encoder_centering_diagnostic",
            "calibration_uses_sts_labels": False,
            "elapsed_encoding_seconds": time.monotonic() - started,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
    first, second = features["first"], features["second"]
    scores = features["scores"].numpy()
    center = features["wiki"].mean(0)
    results = []
    predictions = {}
    for alpha in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
        a, b = first - alpha * center, second - alpha * center
        metrics = stsb_metrics(a, b, scores)
        metrics["relative_alignment"] = metrics["alignment"] / (
            2 * (1 - metrics["pairwise_cosine_mean"])
        )
        row = {"alpha": alpha, **metrics}
        results.append(row)
        predictions[str(alpha)] = F.cosine_similarity(a, b).numpy()
        print(json.dumps(row), flush=True)
    np.savez(output / "predictions.npz", scores=scores, **predictions)
    (output / "centering_probe.json").write_text(json.dumps(results, indent=2))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric in zip(axes, ["alignment", "uniformity", "sts_spearman"]):
        ax.plot([r["alpha"] for r in results], [r[metric] for r in results], marker="o")
        ax.set(xlabel="Subtracted Wiki mean coefficient", ylabel=metric)
    fig.tight_layout()
    fig.savefig(output / "centering_probe.png", dpi=180)
    plt.close(fig)


def synthetic_probe(output: Path) -> None:
    centers, noise = torch.eye(8)[:4], torch.eye(8)[4:]
    rows = []
    for mode in ["normalized_mean", "decoupled"]:
        theta = torch.tensor(0.4, requires_grad=True)
        for step in range(41):
            first = theta.cos() * centers + theta.sin() * noise
            second = theta.cos() * centers - theta.sin() * noise
            loss = (
                UniformityLoss(2)((first + second) / 2)
                if mode == "normalized_mean"
                else DecoupledUniformityLoss(2)((first, second))
            )
            (gradient,) = torch.autograd.grad(loss, theta)
            if step in [0, 1, 10, 40]:
                rows.append(
                    {
                        "mode": mode,
                        "step": step,
                        "theta": theta.item(),
                        "view_alignment": (first - second).square().sum(-1).mean().item(),
                        "loss": loss.item(),
                        "d_loss_d_theta": gradient.item(),
                    }
                )
            with torch.no_grad():
                theta.sub_(0.03 * gradient)
    (output / "synthetic_gradient_probe.json").write_text(json.dumps(rows, indent=2))


def bootstrap_probe(output: Path) -> None:
    predictions = np.load(output / "predictions.npz")
    rng = np.random.default_rng(42)
    scores = predictions["scores"]
    before, after = predictions["0.0"], predictions["1.0"]
    differences = []
    for _ in range(2000):
        indices = rng.integers(0, len(scores), len(scores))
        differences.append(
            spearmanr(after[indices], scores[indices]).statistic
            - spearmanr(before[indices], scores[indices]).statistic
        )
    summary = {
        "comparison": "alpha=1 minus alpha=0",
        "paired_bootstrap_draws": 2000,
        "seed": 42,
        "unit": "STS-B validation pair",
        "delta_spearman": float(
            spearmanr(after, scores).statistic - spearmanr(before, scores).statistic
        ),
        "percentile_95_ci": np.quantile(differences, [0.025, 0.975]).tolist(),
        "limitation": "Descriptive pair bootstrap; not independent training seeds; "
        "repeated sentences can induce dependence.",
    }
    (output / "bootstrap.json").write_text(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/alignment-uniformity-20260911"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/re/checkpoint.pt"))
    parser.add_argument("--wiki-count", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    audit(args.output)
    synthetic_probe(args.output)
    if not args.audit_only:
        probe(args.output, args.checkpoint, args.wiki_count)
        bootstrap_probe(args.output)


if __name__ == "__main__":
    main()
