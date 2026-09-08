"""Recreate trajectory figures from the recorded evaluation metrics."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUNS = {
    "InfoNCE": ROOT / "archive/check_trajectory_infonce_fin",
    "Word augmentation": ROOT / "runs/cls_polling",
    "DINO": ROOT / "archive/check_trajectory_dino_fin",
}
COLORS = ["#2563eb", "#e05469", "#0d9488"]
records = {}
for label, folder in RUNS.items():
    rows = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines() if line.strip()]
    records[label] = sorted((r for r in rows if "sts_spearman" in r), key=lambda r: r["step"])

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.facecolor": "white"})

def finish(fig, ax, name):
    ax.grid(alpha=0.22, linestyle="--")
    ax.set_axisbelow(True)
    ax.margins(0.16)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)

for xkey, ykey, filename, title in [
    ("alignment", "uniformity", "trajectory.png", "Alignment–Uniformity trajectory"),
    ("alignment", "sts_spearman", "alignment_spearman.png", "Alignment–Spearman trajectory"),
    ("uniformity", "sts_spearman", "uniformity_spearman.png", "Uniformity–Spearman trajectory"),
]:
    fig, ax = plt.subplots(figsize=(9, 6.5))
    missing = []
    for run_index, ((label, rows), color) in enumerate(zip(records.items(), COLORS)):
        points = [r for r in rows if xkey in r and ykey in r]
        if not points:
            missing.append(f"{label}: missing " + ", ".join(k for k in (xkey, ykey) if not any(k in r for r in rows)))
            continue
        x = [r[xkey] * (-1 if xkey != "sts_spearman" else 1) for r in points]
        y = [r[ykey] * (-1 if ykey != "sts_spearman" else 1) for r in points]
        ax.plot(x, y, "o-", color=color, lw=2.3, ms=5, label=label)
        ax.scatter(x[0], y[0], s=90, facecolors="white", edgecolors=color, linewidths=2, zorder=4)
        ax.scatter(x[-1], y[-1], s=230, marker="*", color=color, edgecolors="white", zorder=5)
        for i in range(0, len(points)-1, 3):
            ax.annotate("", xy=(x[i+1], y[i+1]), xytext=(x[i], y[i]),
                        arrowprops={"arrowstyle": "->", "color": color, "lw": 1.7})
        start_offset = [(8, 12), (8, -24), (-12, -24)][run_index]
        for i, offset in [(0, start_offset), (len(points)-1, (-8, 24))]:
            ax.annotate(f"step {points[i]['step']}", (x[i], y[i]), xytext=offset,
                        textcoords="offset points", ha="left" if offset[0] > 0 else "right", color=color)
    labels = {"alignment": "−Alignment (higher is better)", "uniformity": "−Uniformity (higher is better)",
              "sts_spearman": "STS Spearman (higher is better)"}
    ax.set(xlabel=labels[xkey], ylabel=labels[ykey], title=title)
    handles, _ = ax.get_legend_handles_labels()
    handles += [Line2D([], [], marker="o", color="gray", markerfacecolor="white", linestyle="", label="Start"),
                Line2D([], [], marker="*", color="gray", markersize=12, linestyle="", label="Final")]
    ax.legend(handles=handles, loc="best", framealpha=0.9)
    fig.text(0.5, 0.025, "; ".join(missing) if missing else "Arrows indicate training direction; stars mark the final evaluation.",
             ha="center", fontsize=9, color="#64748b")
    finish(fig, ax, filename)

fig, ax = plt.subplots(figsize=(9, 6))
for (label, rows), color in zip(records.items(), COLORS):
    ax.plot([r["step"] for r in rows], [r["sts_spearman"] for r in rows], "o-", color=color, lw=2.3, label=label)
ax.set(xlabel="Training step", ylabel="STS Spearman", title="Spearman over training")
ax.legend()
fig.text(0.5, 0.025, "Recorded evaluations only; all runs start at step 0.",
         ha="center", fontsize=9, color="#64748b")
finish(fig, ax, "spearman_steps.png")
print(f"Saved four plots to {OUT}")
