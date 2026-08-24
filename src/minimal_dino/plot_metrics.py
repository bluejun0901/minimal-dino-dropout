from __future__ import annotations

import argparse
import json
import math
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any


def metric_group(record: dict[str, Any]) -> str:
    if "loss" in record:
        return "train"
    if "sts_spearman" in record:
        return "eval"
    return "other"


def parse_metrics(path: str | Path) -> dict[str, list[tuple[int, float]]]:
    """Parse numeric metric series from a JSONL file that may still be growing."""
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                warnings.warn(f"Skipping incomplete JSON on line {line_number}")
                continue
            step = record.get("step")
            if not isinstance(step, int) or isinstance(step, bool):
                warnings.warn(f"Skipping metrics without an integer step on line {line_number}")
                continue
            group = metric_group(record)
            for name, value in record.items():
                if name == "step" or not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                series[f"{group}.{name}"].append((step, float(value)))
    return dict(series)


def plot_metrics(
    series: dict[str, list[tuple[int, float]]],
    output: str | Path,
    selected: list[str] | None = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    names = selected if selected is not None else sorted(series)
    missing = [name for name in names if name not in series]
    if missing:
        available = ", ".join(sorted(series))
        raise ValueError(f"Unknown metrics: {', '.join(missing)}. Available metrics: {available}")
    if not names:
        raise ValueError("No numeric metrics found")

    columns = min(3, len(names))
    rows = math.ceil(len(names) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3 * rows), squeeze=False)
    for axis, name in zip(axes.flat, names):
        points = series[name]
        axis.plot([step for step, _ in points], [value for _, value in points])
        axis.set_title(name)
        axis.set_xlabel("step")
        axis.grid(alpha=0.25)
    for axis in list(axes.flat)[len(names) :]:
        axis.set_visible(False)
    figure.tight_layout()

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot metrics from a minimal DINO JSONL log")
    parser.add_argument("input", help="Path to metrics.jsonl")
    parser.add_argument("--output", help="Output image path; defaults to metrics.png beside input")
    parser.add_argument(
        "--metrics",
        nargs="+",
        help="Series to plot, such as train.loss or eval.sts_spearman (default: all)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix(".png")
    series = parse_metrics(input_path)
    saved_path = plot_metrics(series, output_path, args.metrics)
    print(f"Saved metrics plot to {saved_path}")


if __name__ == "__main__":
    main()
