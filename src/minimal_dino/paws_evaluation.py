from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from scipy.stats import rankdata
from torch.nn import functional as F

from minimal_dino.evaluation import encode_stsb_dataset
from minimal_dino.model import SentenceBYOL


def load_paws_split(data_dir: str | Path, split: str) -> Dataset:
    """Load a local PAWS-Wiki labeled_final Parquet split without Hub access."""
    if split not in {"validation", "test"}:
        raise ValueError("PAWS split must be 'validation' or 'test'")
    path = Path(data_dir) / f"{split}.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"PAWS split not found at {path}. Download the PAWS-Wiki labeled_final "
            "Parquet files described in README.md and set --paws-dir."
        )
    dataset = Dataset.from_parquet(str(path))
    _validate_dataset(dataset)
    # Discard unrelated columns, including any 'score' column: labels are authoritative.
    return dataset.select_columns(["sentence1", "sentence2", "label"])


def _validate_dataset(dataset: Dataset) -> None:
    required = {"sentence1", "sentence2", "label"}
    if not required.issubset(dataset.column_names):
        raise ValueError("PAWS requires sentence1, sentence2, and label columns")
    if len(dataset) < 2:
        raise ValueError("PAWS requires at least two labeled sentence pairs")
    for column in ("sentence1", "sentence2"):
        if any(not isinstance(value, str) or not value.strip() for value in dataset[column]):
            raise ValueError(f"PAWS {column} must contain non-empty strings")
    if not np.isin(np.asarray(dataset["label"]), [0, 1]).all():
        raise ValueError("PAWS labels must be 0 (non-paraphrase) or 1 (paraphrase)")


def _validate_scores(similarities: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    similarities = np.asarray(similarities, dtype=np.float64)
    labels = np.asarray(labels)
    if similarities.ndim != 1 or labels.shape != similarities.shape or len(labels) < 2:
        raise ValueError("Expected equally sized one-dimensional scores and labels, length >= 2")
    if not np.isfinite(similarities).all():
        raise ValueError("PAWS similarities must be finite")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("PAWS labels must be 0 or 1")
    if len(np.unique(labels)) != 2:
        raise ValueError("PAWS evaluation requires both label classes; increase --limit if set")
    return similarities, labels.astype(bool)


def _threshold_counts(
    similarities: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Counts at each distinct score, grouping ties for the >= threshold rule."""
    order = np.argsort(-similarities, kind="stable")
    sorted_scores = similarities[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(sorted_scores) - 1]
    true_positive = np.cumsum(labels[order])[ends]
    false_positive = ends + 1 - true_positive
    return sorted_scores[ends], true_positive, false_positive


def select_paws_threshold(similarities: np.ndarray, labels: np.ndarray) -> float:
    """Maximize validation accuracy; prefer the highest threshold when tied."""
    similarities, labels = _validate_scores(similarities, labels)
    thresholds, true_positive, false_positive = _threshold_counts(similarities, labels)
    # Also consider predicting every pair as negative, even when max cosine is 1.
    thresholds = np.r_[np.nextafter(thresholds[0], np.inf), thresholds]
    correct = np.r_[0, true_positive - false_positive] + np.count_nonzero(~labels)
    return float(thresholds[np.argmax(correct)])


def paws_metrics(
    similarities: np.ndarray, labels: np.ndarray, *, threshold: float
) -> dict[str, float | int]:
    """Binary paraphrase metrics in [0, 1], with tied scores handled together."""
    similarities, labels = _validate_scores(similarities, labels)
    if not np.isfinite(threshold):
        raise ValueError("PAWS threshold must be finite")
    predictions = similarities >= threshold
    tp = int(np.count_nonzero(predictions & labels))
    fp = int(np.count_nonzero(predictions & ~labels))
    fn = int(np.count_nonzero(~predictions & labels))
    tn = int(np.count_nonzero(~predictions & ~labels))
    positives = int(labels.sum())
    negatives = len(labels) - positives
    ranks = rankdata(similarities, method="average")
    auc = (ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    _, cumulative_tp, cumulative_fp = _threshold_counts(similarities, labels)
    recall = cumulative_tp / positives
    precision = cumulative_tp / (cumulative_tp + cumulative_fp)
    return {
        "num_pairs": len(labels),
        "roc_auc": float(auc),
        "average_precision": float(np.sum(np.diff(recall, prepend=0) * precision)),
        "accuracy": (tp + tn) / len(labels),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / positives,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
    }


def evaluate_paws(
    model: SentenceBYOL,
    tokenizer: Any,
    dataset: Dataset,
    *,
    device: torch.device,
    validation_dataset: Dataset | None = None,
    threshold: float | None = None,
    batch_size: int = 64,
    limit: int | None = None,
) -> dict[str, Any]:
    """Evaluate cosine similarity, optionally calibrating on separate validation data.

    The encoder is frozen. Test labels never participate in threshold selection.
    A fixed threshold skips validation encoding entirely. ``limit`` applies to
    both evaluation and calibration data, and is intended only for smoke tests.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if limit is not None and limit < 2:
        raise ValueError("limit must be at least 2")
    if threshold is not None and not np.isfinite(threshold):
        raise ValueError("PAWS threshold must be finite")
    if threshold is None and validation_dataset is None:
        raise ValueError("Provide validation_dataset for calibration or a fixed threshold")

    def prepare(data: Dataset) -> Dataset:
        _validate_dataset(data)
        if limit is not None:
            data = data.select(range(min(limit, len(data))))
        _validate_scores(np.zeros(len(data)), np.asarray(data["label"]))
        return data.select_columns(["sentence1", "sentence2", "label"])

    def score(data: Dataset) -> tuple[np.ndarray, np.ndarray]:
        first, second, labels = encode_stsb_dataset(
            model, tokenizer, data, device=device, batch_size=batch_size
        )
        return F.cosine_similarity(first.float(), second.float()).numpy(), labels

    dataset = prepare(dataset)
    source = "fixed"
    calibration: dict[str, float | int] | None = None
    if threshold is None:
        validation_dataset = prepare(validation_dataset)
        similarities, labels = score(validation_dataset)
        threshold = select_paws_threshold(similarities, labels)
        source = "validation_accuracy"
        calibration = paws_metrics(similarities, labels, threshold=threshold)
    similarities, labels = score(dataset)
    return {
        "suite": "paws",
        "dataset": "PAWS-Wiki labeled_final",
        "limit": limit,
        "threshold": threshold,
        "threshold_source": source,
        "validation": calibration,
        **paws_metrics(similarities, labels, threshold=threshold),
    }
