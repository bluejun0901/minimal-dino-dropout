"""Generate exportable research figures from saved JSON evidence (no model execution)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, output = Path(args.root), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    synthetic = json.loads((root / "synthetic-matrix/results.json").read_text())["results"]
    frozen = json.loads((root / "probe/results.json").read_text())["results"]
    colors = {
        "scalar": "#52525b",
        "paired": "#b42318",
        "none": "#2563eb",
        "diagonal": "#059669",
        "shuffled": "#a16207",
    }
    labels = {
        "scalar": "Scalar",
        "paired": "PRE matrix",
        "none": "Whitening (R=I)",
        "diagonal": "Diagonal R",
        "shuffled": "Shuffled pairs",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for method, color in colors.items():
        selected = [r for r in synthetic if r["method"] == method]
        noise = [r["noise_energy_ratio"] for r in selected] or [1.0]
        signal = [r["stable_small_signal_energy_ratio"] for r in selected] or [1.0]
        axes[0].errorbar(
            np.mean(noise),
            np.mean(signal),
            xerr=np.std(noise),
            yerr=np.std(signal),
            fmt="o",
            color=color,
            label=labels[method],
            capsize=3,
        )
    axes[0].axvline(1, color="#999999", linestyle=":", linewidth=1)
    axes[0].axhline(1, color="#999999", linestyle=":", linewidth=1)
    axes[0].set(
        xlabel="Noise energy multiplier (lower is better)",
        ylabel="Weak stable signal energy multiplier",
        title="Synthetic mechanism: 3 seeds",
    )
    axes[0].legend(fontsize=8, loc="lower right")
    for row in frozen:
        method = row["method"]
        if method not in colors:
            continue
        axes[1].scatter(row["effective_rank"], row["sts_spearman"], color=colors[method], s=50)
        offsets = {
            "scalar": (5, 5),
            "paired": (-58, -15),
            "none": (-93, -16),
            "diagonal": (-68, 9),
            "shuffled": (5, -15),
        }
        axes[1].annotate(
            labels[method],
            (row["effective_rank"], row["sts_spearman"]),
            xytext=offsets[method],
            textcoords="offset points",
            fontsize=8,
        )
    axes[1].set(
        xlabel="Effective rank (singular-value entropy)",
        ylabel="STS-B Spearman",
        title="Frozen BERT probe: rank does not imply semantics",
    )
    axes[1].margins(x=0.2, y=0.25)
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "mechanism_vs_semantics.png", dpi=180)
    fig.savefig(output / "mechanism_vs_semantics.pdf")
    plt.close(fig)
    print(output / "mechanism_vs_semantics.png")


if __name__ == "__main__":
    main()
