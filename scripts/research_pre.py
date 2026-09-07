"""Reproducible PRE-BYOL audits, frozen representation probes, and controlled pilots.

Run from the repository root after activating .venv. Every command creates a new
output directory and refuses to overwrite an existing experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
from scipy.stats import spearmanr
from torch.nn import functional as F
from transformers import AutoTokenizer

from minimal_dino.augmentation import augment_words
from minimal_dino.config import to_train_args
from minimal_dino.evaluation import encode_stsb_dataset, load_stsb_split, stsb_metrics
from minimal_dino.geometry import GeometryConfig, PairedTargetGeometry
from minimal_dino.model import SentenceBYOL
from minimal_dino.objective import BYOLLoss
from minimal_dino.train import save_run_artifacts, set_seed, train

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def create_output(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    save_run_artifacts(output, args, git_cwd=ROOT)
    return output


def audit(args):
    output = create_output(args)
    with initialize_config_module(version_base="1.3", config_module="minimal_dino.conf"):
        config = compose(config_name="config")
    OmegaConf.save(config, output / "current_config.yaml", resolve=True)
    rows = []
    for root in (ROOT / "runs", ROOT / "archive"):
        for metric_path in sorted(root.rglob("metrics.jsonl")):
            if output in metric_path.parents:
                continue
            records, invalid = [], 0
            for line in metric_path.read_text().splitlines():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    invalid += 1
            evaluation = [r for r in records if "sts_spearman" in r]
            if not evaluation:
                continue
            folder = metric_path.parent
            config_path = folder / ".hydra/config.yaml"
            if config_path.exists():
                config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
                config_source = str(config_path.relative_to(ROOT))
            elif (folder / "config.json").exists():
                config = json.loads((folder / "config.json").read_text())
                config_source = str((folder / "config.json").relative_to(ROOT))
            else:
                config, config_source = {}, None
            state_path = folder / "git_state.json"
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            commit = state.get("commit_hash")
            diff_path = folder / "git.diff"
            diff = diff_path.read_text() if diff_path.exists() else ""
            # Recover committed source plus inspect saved patch. Older artifact writers
            # omitted staged changes; missing source cannot be inferred from a run name.
            source = {}
            if commit:
                for filename in ("train.py", "augmentation.py"):
                    result = subprocess.run(
                        ["git", "show", f"{commit}:src/minimal_dino/{filename}"],
                        cwd=ROOT,
                        capture_output=True,
                        text=True,
                    )
                    source[filename] = result.stdout if result.returncode == 0 else None
            rows.append(
                {
                    "run": str(folder.relative_to(ROOT)),
                    "config": config,
                    "config_source": config_source,
                    "git_state": state,
                    "diff_sha256": digest(diff_path) if diff_path.exists() else None,
                    "source_at_commit": source,
                    "dropout_patch_lines": [
                        line
                        for line in diff.splitlines()
                        if "use_dropout" in line or "dropout_probability" in line
                    ],
                    "invalid_json_lines": invalid,
                    "initial": evaluation[0],
                    "final": evaluation[-1],
                    "best": max(evaluation, key=lambda x: x["sts_spearman"]),
                    "evaluation_trajectory": evaluation,
                }
            )
    write_json(output / "historical_runs.json", rows)
    print(json.dumps({"historical_runs": len(rows), "output": str(output)}), flush=True)


@torch.inference_mode()
def cache(args):
    output = create_output(args)
    set_seed(args.seed)
    with initialize_config_module(version_base="1.3", config_module="minimal_dino.conf"):
        config = compose(config_name="config")
    model = SentenceBYOL.from_pretrained(
        config.model.name,
        revision=config.model.revision,
        dropout=config.model.dropout,
        pooling=config.model.pooling,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, revision=config.model.revision)
    lines = Path(config.data.train_file).read_text().splitlines()
    selected = random.Random(args.seed).sample(range(len(lines)), args.samples)
    sentences = [lines[i] for i in selected]
    rng = random.Random(args.seed + 1)
    views = [
        [augment_words(s, config.augmentation.strength, rng=rng) for s in sentences]
        for _ in range(2)
    ]
    embeddings = []
    start = time.monotonic()
    for index, texts in enumerate(views):
        batches = []
        for offset in range(0, len(texts), args.batch_size):
            encoded = tokenizer(
                texts[offset : offset + args.batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            )
            batches.append(
                model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    use_dropout=True,
                    dropout_probability=config.teacher.dropout,
                    target=True,
                ).embedding.cpu()
            )
            if offset % (args.batch_size * 16) == 0:
                print(
                    json.dumps(
                        {
                            "view": index,
                            "sentences": offset,
                            "elapsed_seconds": time.monotonic() - start,
                        }
                    ),
                    flush=True,
                )
        embeddings.append(torch.cat(batches))
    dataset = load_stsb_split(config.evaluation.stsb_dir, "validation")
    first, second, scores = encode_stsb_dataset(
        model,
        tokenizer,
        dataset,
        device=torch.device("cpu"),
        batch_size=args.batch_size,
    )
    torch.save(
        {
            "train_views": embeddings,
            "validation_first": first,
            "validation_second": second,
            "scores": torch.tensor(scores),
        },
        output / "features.pt",
    )
    write_json(
        output / "manifest.json",
        {
            "kind": "frozen_pretrained_teacher_probe_not_BYOL_training",
            "model": OmegaConf.to_container(config.model),
            "source_train_sha256": digest(config.data.train_file),
            "source_validation_sha256": digest(
                Path(config.evaluation.stsb_dir) / "validation.parquet"
            ),
            "train_indices": selected,
            "augmentation": OmegaConf.to_container(config.augmentation),
            "teacher_dropout": config.teacher.dropout,
            "max_train_length": args.max_length,
            "eval_truncation": False,
            "elapsed_seconds": time.monotonic() - start,
        },
    )
    print(f"Saved frozen features to {output}", flush=True)


def paired_bootstrap(similarities, baseline, scores, seed, samples=1000):
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(samples):
        indices = rng.integers(0, len(scores), len(scores))
        deltas.append(
            spearmanr(similarities[indices], scores[indices]).statistic
            - spearmanr(baseline[indices], scores[indices]).statistic
        )
    return np.quantile(deltas, [0.025, 0.975]).tolist()


def probe(args):
    output = create_output(args)
    features = torch.load(Path(args.cache) / "features.pt", weights_only=True)
    first, second = features["train_views"]
    left, right = features["validation_first"], features["validation_second"]
    scores = features["scores"].numpy()
    # A population probe: fit once on unlabeled Wiki, never on STS sentences or labels.
    # This deliberately tests the operator, not the eventual online training dynamics.
    rows, predictions = [], {}
    for method in ("raw", "scalar", "paired", "none", "shuffled", "diagonal", "zero"):
        config = (
            None
            if method in {"raw", "scalar"}
            else {
                "momentum": 0,
                "warmup_steps": 1,
                "reliability": method if method in {"none", "shuffled", "diagonal"} else "paired",
                "strength": 0 if method == "zero" else 1,
            }
        )
        objective = BYOLLoss(first.shape[1], center_momentum=0, target_geometry=config)
        objective.update_center((first, second))
        scale = 0 if method == "raw" else 0.05
        transformed = [objective.transform_target(h, scale) for h in (left, right)]
        metrics = stsb_metrics(*transformed, scores)
        similarities = F.cosine_similarity(*transformed).numpy()
        predictions[method] = similarities
        row = {"method": method, **metrics}
        if objective.geometry is not None:
            row.update({k: v.item() for k, v in objective.geometry.metrics().items()})
            torch.save(objective.state_dict(), output / f"{method}_operator.pt")
        rows.append(row)
    for row in rows:
        method = row["method"]
        row["delta_vs_scalar"] = row["sts_spearman"] - rows[1]["sts_spearman"]
        row["paired_95pct_ci_vs_scalar"] = paired_bootstrap(
            predictions[method],
            predictions["scalar"],
            scores,
            args.seed,
        )
    np.savez(output / "predictions.npz", scores=scores, **predictions)
    write_json(
        output / "results.json",
        {
            "kind": "frozen_feature_operator_probe_not_trained_encoder_performance",
            "cache": str(Path(args.cache).resolve()),
            "cache_sha256": digest(Path(args.cache) / "features.pt"),
            "fit_sentences": len(first),
            "validation_pairs": len(scores),
            "results": rows,
            "pairing_vs_whitening_95pct_ci": paired_bootstrap(
                predictions["paired"],
                predictions["none"],
                scores,
                args.seed,
            ),
        },
    )
    print(json.dumps(rows, indent=2), flush=True)


def synthetic(args):
    output = create_output(args)
    results = []
    for seed in (11, 22, 33):
        set_seed(seed)
        dim, count = 12, 8192
        # Two equally small-variance groups: reproducible signal versus independent noise.
        signal_variance = torch.tensor([9.0] * 4 + [0.09] * 4 + [0.0001] * 4)
        noise_variance = torch.tensor([0.01] * 4 + [0.001] * 4 + [0.09] * 4)
        signal = torch.randn(count, dim) * signal_variance.sqrt()
        mean = torch.arange(dim).float()[None, :] / 3
        views = tuple(
            signal + torch.randn(count, dim) * noise_variance.sqrt() + mean for _ in range(2)
        )
        for mode in ("paired", "none", "shuffled", "diagonal"):
            geometry = PairedTargetGeometry(
                dim,
                GeometryConfig(
                    warmup_steps=1,
                    momentum=0,
                    reliability=mode,
                ),
            )
            geometry.update(views)
            operator = torch.eye(dim) + geometry.correction
            noise_energy = (noise_variance[:, None] * operator.square()).sum()
            stable_small_energy = (signal_variance[4:8, None] * operator[4:8].square()).sum()
            results.append(
                {
                    "seed": seed,
                    "method": mode,
                    "noise_energy_ratio": float(noise_energy / noise_variance.sum()),
                    "stable_small_signal_energy_ratio": float(
                        stable_small_energy / signal_variance[4:8].sum()
                    ),
                    **{k: v.item() for k, v in geometry.metrics().items()},
                }
            )
    # Counterexample: low-variance semantic information that augmentation destroys.
    # Reliability cannot tell useful-but-unstable semantics from nuisance variation.
    geometry = PairedTargetGeometry(3, GeometryConfig(ridge=1e-5))
    geometry.covariance.copy_(torch.diag(torch.tensor([0.1, 0.2, 10.0])))
    geometry.noise_covariance.copy_(geometry.covariance)
    geometry.refresh()
    write_json(
        output / "results.json",
        {
            "kind": "synthetic_mechanism_probe_not_STS",
            "results": results,
            "destroyed_semantics_counterexample": {
                "gains": geometry.gains.tolist(),
                "meaning": "If all between-view variation carries semantics, reliability blocks "
                "equalization of useful modes too; statistics cannot identify semantics.",
            },
        },
    )
    print(json.dumps(results, indent=2), flush=True)


def pilot(args):
    output = create_output(args)
    manifest = []
    for seed in args.seeds:
        for method in args.methods:
            directory = output / f"{method}-seed{seed}"
            overrides = [
                f"runtime.output_dir={directory}",
                f"runtime.seed={seed}",
                f"runtime.device={args.device}",
                "logging.quiet=false",
                "logging.tensorboard=false",
                "checkpoint.save_steps=0",
                "data.num_workers=0",
            ]
            if args.device == "cpu":
                overrides += ["runtime.byol_precision=fp32"]
            if not args.full:
                overrides += [
                    f"optimization.max_steps={args.steps}",
                    f"optimization.batch_size={args.batch_size}",
                    f"data.max_length={args.max_length}",
                    f"evaluation.steps={args.steps}",
                ]
            if method != "scalar":
                overrides += ["objective=paired_byol"]
                if method in {"none", "shuffled", "diagonal"}:
                    overrides += [f"objective.target_geometry.reliability={method}"]
                elif method == "zero":
                    overrides += ["objective.target_geometry.strength=0"]
            with initialize_config_module(version_base="1.3", config_module="minimal_dino.conf"):
                config = compose(config_name="config", overrides=overrides)
            directory.mkdir(parents=True, exist_ok=False)
            hydra_dir = directory / ".hydra"
            hydra_dir.mkdir()
            OmegaConf.save(config, hydra_dir / "config.yaml", resolve=True)
            OmegaConf.save(OmegaConf.create(overrides), hydra_dir / "overrides.yaml")
            start = time.monotonic()
            print(f"Starting {directory}", flush=True)
            checkpoint = train(to_train_args(config), run_config=config)
            # Export predictions for paired uncertainty estimates, using the saved encoder.
            from minimal_dino.evaluation import load_checkpoint

            model, tokenizer = load_checkpoint(checkpoint, torch.device(args.device))
            dataset = load_stsb_split(config.evaluation.stsb_dir, "validation")
            first, second, scores = encode_stsb_dataset(
                model,
                tokenizer,
                dataset,
                device=torch.device(args.device),
                batch_size=config.evaluation.batch_size,
            )
            np.savez(
                directory / "validation_predictions.npz",
                scores=scores,
                similarities=F.cosine_similarity(first, second).numpy(),
            )
            del model
            manifest.append(
                {
                    "method": method,
                    "seed": seed,
                    "output": str(directory),
                    "elapsed_seconds": time.monotonic() - start,
                }
            )
            write_json(
                output / "manifest.json",
                {
                    "kind": "full_3000_steps" if args.full else "limited_CPU_pilot",
                    "completed": manifest,
                },
            )


def summarize(args):
    output = create_output(args)
    manifest = json.loads((Path(args.pilot) / "manifest.json").read_text())
    rows = []
    for run in manifest["completed"]:
        directory = Path(run["output"])
        records = [json.loads(x) for x in (directory / "metrics.jsonl").read_text().splitlines()]
        evaluation = [x for x in records if "sts_spearman" in x]
        last_train = [x for x in records if "loss" in x][-1]
        pred = np.load(directory / "validation_predictions.npz")
        baseline = np.load(
            Path(args.pilot) / f"scalar-seed{run['seed']}" / "validation_predictions.npz"
        )
        rows.append(
            {
                **run,
                "initial": evaluation[0],
                "final": evaluation[-1],
                "last_training_metrics": last_train,
                "paired_95pct_ci_vs_scalar": paired_bootstrap(
                    pred["similarities"],
                    baseline["similarities"],
                    pred["scores"],
                    args.seed,
                ),
            }
        )
    write_json(output / "results.json", {"kind": manifest["kind"], "results": rows})
    print(json.dumps(rows, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("audit", "cache", "probe", "synthetic", "pilot", "summarize")
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache")
    parser.add_argument("--pilot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("scalar", "paired", "none", "shuffled", "diagonal", "zero"),
        default=["scalar", "paired", "none"],
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--full", action="store_true", help="Use current 3000-step defaults unchanged"
    )
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    globals()[args.command](args)


if __name__ == "__main__":
    main()
