"""Predeclared mechanism checks after a negative frozen-feature PRE result.

Uses only an existing feature cache, does not train or change any pilot runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from scipy.stats import spearmanr

from minimal_dino.evaluation import stsb_metrics
from minimal_dino.objective import BYOLLoss
from minimal_dino.train import save_run_artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-curve", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    save_run_artifacts(output, args)
    data = torch.load(Path(args.cache) / "features.pt", weights_only=True)
    first, second = data["train_views"]
    if args.sample_curve:
        rows = []
        held_difference = (first[-512:] - second[-512:]) / 2**0.5
        for count in (256, 512, 1024, 1536):
            if count > len(first) - 512:
                continue
            for method in ("paired", "none", "diagonal"):
                objective = BYOLLoss(
                    first.shape[1],
                    center_momentum=0,
                    target_geometry={
                        "momentum": 0,
                        "warmup_steps": 1,
                        "reliability": method,
                    },
                )
                objective.update_center((first[:count], second[:count]))
                operator = torch.eye(first.shape[1]) + objective.geometry.correction
                fit_difference = (first[:count] - second[:count]) / 2**0.5
                row = {"method": method, "fit_sentences": count, "heldout_sentences": 512}
                for name, difference in (("fit", fit_difference), ("heldout", held_difference)):
                    row[f"{name}_noise_energy_ratio"] = (
                        (difference @ operator).square().sum() / difference.square().sum()
                    ).item()
                rows.append(row)
        (output / "results.json").write_text(
            json.dumps(
                {
                    "kind": "exploratory_sample_size_ablation_fixed_Wiki_holdout_no_STS_labels",
                    "results": rows,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(json.dumps(rows, indent=2), flush=True)
        return
    left, right, scores = (
        data["validation_first"],
        data["validation_second"],
        data["scores"].numpy(),
    )
    origin_rows, heldout_rows = [], []
    for method in ("scalar", "paired", "none", "diagonal", "shuffled"):
        config = (
            None
            if method == "scalar"
            else {
                "momentum": 0,
                "warmup_steps": 1,
                "reliability": method,
            }
        )
        objective = BYOLLoss(first.shape[1], center_momentum=0, target_geometry=config)
        objective.update_center((first, second))
        # Exactly two fixed origins; this is a mechanism ablation, not an alpha search.
        for scale in (0.05, 1.0):
            metrics = stsb_metrics(
                objective.transform_target(left, scale),
                objective.transform_target(right, scale),
                scores,
            )
            origin_rows.append({"method": method, "center_scale": scale, **metrics})
        if method == "scalar":
            continue
        midpoint = len(first) // 2
        for fold in (0, 1):
            fit = slice(0, midpoint) if fold == 0 else slice(midpoint, None)
            hold = slice(midpoint, None) if fold == 0 else slice(0, midpoint)
            estimator = BYOLLoss(first.shape[1], center_momentum=0, target_geometry=config)
            estimator.update_center((first[fit], second[fit]))
            geometry = estimator.geometry
            transform = torch.eye(first.shape[1]) + geometry.correction
            for partition, selection in (("fit", fit), ("heldout", hold)):
                difference = (first[selection] - second[selection]) / 2**0.5
                original_energy = difference.square().sum()
                filtered_energy = (difference @ transform).square().sum()
                row = {
                    "method": method,
                    "fold": fold,
                    "partition": partition,
                    "noise_energy_ratio": (filtered_energy / original_energy).item(),
                }
                if method == "paired":
                    values, vectors = torch.linalg.eigh(geometry.covariance)
                    denominator = values.clamp_min(0) + geometry.config.ridge * values.mean()
                    inverse_root = (vectors * denominator.rsqrt()) @ vectors.T
                    reliability, basis = torch.linalg.eigh(geometry.reliability_operator)
                    directions = inverse_root @ basis
                    observations = torch.cat((first[selection], second[selection])) @ directions
                    total_variance = observations.var(0, unbiased=False)
                    noise_variance = (difference @ directions).square().mean(0)
                    empirical = 1 - noise_variance / total_variance.clamp_min(1e-12)
                    row["reliability_order_spearman"] = spearmanr(
                        reliability.numpy(),
                        empirical.numpy(),
                    ).statistic
                    row["bottom_quarter_empirical_reliability"] = empirical[:192].mean().item()
                    row["top_quarter_empirical_reliability"] = empirical[-192:].mean().item()
                heldout_rows.append(row)
    result = {
        "kind": "post_probe_exploratory_diagnostics_not_independent_confirmation",
        "origin_ablation": origin_rows,
        "split_half_noise_generalization": heldout_rows,
    }
    (output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for row in origin_rows:
        print({key: row[key] for key in ("method", "center_scale", "sts_spearman")}, flush=True)
    print(json.dumps(heldout_rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
