"""Aggregate completed matched PRE pilots with seed and example uncertainty separated."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


def common_config(config):
    config = copy.deepcopy(config)
    config["runtime"].pop("output_dir")
    config["runtime"].pop("seed")
    config["objective"].pop("target_geometry", None)
    return config


def source_sections(patch):
    # Source files loaded into the training process must agree across arms.
    sections = {}
    for section in patch.split("diff --git ")[1:]:
        header = section.splitlines()[0]
        if " b/src/minimal_dino/" in header:
            sections[header] = section
    return sections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilots", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--methods", nargs="+", default=["scalar", "paired", "none"])
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    runs = []
    training_logs = {}
    reference_config = reference_source = None
    for pilot in args.pilots:
        manifest = json.loads((Path(pilot) / "manifest.json").read_text())
        for row in manifest["completed"]:
            directory = Path(row["output"])
            config = common_config(json.loads((directory / "config.json").read_text()))
            source = source_sections((directory / "git.diff").read_text())
            if reference_config is not None:
                assert config == reference_config, "Unmatched common experiment settings"
                assert source == reference_source, "Training source differs across arms"
            reference_config, reference_source = config, source
            logs = [json.loads(x) for x in (directory / "metrics.jsonl").read_text().splitlines()]
            initial = [x for x in logs if "sts_spearman" in x][0]
            final = [x for x in logs if "sts_spearman" in x][-1]
            assert final["step"] == config["optimization"]["max_steps"], "Incomplete run"
            last_train = [x for x in logs if "loss" in x][-1]
            training_logs[(row["seed"], row["method"])] = {
                x["step"]: x for x in logs if "loss" in x
            }
            runs.append({**row, "initial": initial, "final": final, "last_train": last_train})
    assert len({(x["seed"], x["method"]) for x in runs}) == len(runs), "Duplicate arms"
    methods = sorted({r["method"] for r in runs})
    assert set(methods) == set(args.methods), "Requested comparison arms are missing or unexpected"
    seeds = sorted({r["seed"] for r in runs})
    lookup = {(r["seed"], r["method"]): r for r in runs}
    assert all((s, m) in lookup for s in seeds for m in methods), "Unbalanced seed/method design"
    trajectory_checks = []
    for seed in seeds:
        baseline_logs = training_logs[(seed, "scalar")]
        for method in methods:
            candidate_logs = training_logs[(seed, method)]
            warmup_error = max(
                abs(record["loss"] - candidate_logs[step]["loss"])
                for step, record in baseline_logs.items()
                if step <= 50
            )
            frozen_error = max(
                abs(record[field] - candidate_logs[step][field])
                for step, record in baseline_logs.items()
                if step <= 150
                for field in ("embedding_std", "student_view_cosine", "teacher_view_cosine")
            )
            if reference_config["runtime"]["device"] == "cpu":
                assert warmup_error == 0, "The common 50-step path differs"
                assert frozen_error == 0, "Frozen encoder views differ across matched methods"
            trajectory_checks.append(
                {
                    "seed": seed,
                    "method": method,
                    "max_loss_error_first50": warmup_error,
                    "max_raw_geometry_error_first150": frozen_error,
                }
            )
    scores = None
    predictions = {}
    prediction_roundoff = []
    for row in runs:
        pred = np.load(Path(row["output"]) / "validation_predictions.npz")
        if scores is not None:
            np.testing.assert_array_equal(scores, pred["scores"])
        scores = pred["scores"]
        predictions[(row["seed"], row["method"])] = pred["similarities"]
        exported_rho = float(spearmanr(pred["similarities"], scores).statistic)
        difference = exported_rho - row["final"]["sts_spearman"]
        # Export uses one cosine normalization; the legacy evaluator normalizes twice.
        # Re-encoding and near ties can introduce tiny FP32 differences, but not a large shift.
        assert abs(difference) < 1e-5, "Saved predictions disagree with the logged evaluation"
        prediction_roundoff.append(
            {"method": row["method"], "seed": row["seed"], "exported_minus_logged_rho": difference}
        )
    rng = np.random.default_rng(20260907)
    bootstrap_indices = [rng.integers(0, len(scores), len(scores)) for _ in range(1000)]
    metrics = (
        "sts_spearman",
        "sts_pearson",
        "alignment",
        "uniformity",
        "effective_rank",
        "participation_ratio",
        "covariance_top1_mass",
        "covariance_top10_mass",
        "embedding_std",
        "pairwise_cosine_mean",
        "pairwise_cosine_std",
    )
    summary = {}
    for method in methods:
        values = {}
        for metric in metrics:
            observations = [lookup[(s, method)]["final"][metric] for s in seeds]
            values[metric] = {
                "mean": float(np.mean(observations)),
                "sample_std": float(np.std(observations, ddof=1)) if len(seeds) > 1 else None,
            }
        deltas = [
            lookup[(s, method)]["final"]["sts_spearman"]
            - lookup[(s, "scalar")]["final"]["sts_spearman"]
            for s in seeds
        ]
        distribution = []
        for indices in bootstrap_indices:
            distribution.append(
                np.mean(
                    [
                        spearmanr(predictions[(s, method)][indices], scores[indices]).statistic
                        - spearmanr(predictions[(s, "scalar")][indices], scores[indices]).statistic
                        for s in seeds
                    ]
                )
            )
        summary[method] = {
            "metrics": values,
            "paired_seed_deltas": deltas,
            "mean_delta_vs_scalar": float(np.mean(deltas)),
            "example_bootstrap_95pct_ci_fixed_seeds": np.quantile(
                distribution, [0.025, 0.975]
            ).tolist(),
        }
    result = {
        "seeds": seeds,
        "runs": runs,
        "summary": summary,
        "common_config": reference_config,
        "prediction_roundoff": prediction_roundoff,
        "trajectory_checks": trajectory_checks,
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fairness_checks": "common config, source patch, validation scores, balanced arms",
        "uncertainty": "Sample SD is across seeds; bootstrap resamples examples with seeds "
        "held fixed. Neither is proof of generalization outside this pilot.",
    }
    (output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    figure, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    names = [m for m in ("scalar", "paired", "none") if m in methods]
    for seed in seeds:
        axes[0].plot(
            names,
            [lookup[(seed, m)]["final"]["sts_spearman"] for m in names],
            marker="o",
            label=f"seed {seed}",
        )
        axes[1].plot(
            names, [lookup[(seed, m)]["final"]["effective_rank"] for m in names], marker="o"
        )
    axes[0].set(title="200-step CPU pilot (150 frozen steps)", ylabel="STS-B Spearman")
    axes[1].set(title="Raw online encoder geometry", ylabel="Effective rank")
    axes[0].legend()
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output / "pilot_comparison.png", dpi=180)
    figure.savefig(output / "pilot_comparison.pdf")
    plt.close(figure)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
