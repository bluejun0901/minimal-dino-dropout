from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from scipy.stats import pearsonr, spearmanr
from torch.nn import functional as F

from minimal_dino.evaluation import encode_stsb_dataset
from minimal_dino.model import SentenceBYOL

# The English test subsets used by SimCSE's SentEval evaluation. STS13 excludes SMT.
STS_SUBSETS = {
    "STS12": ("MSRpar", "MSRvid", "SMTeuroparl", "surprise.OnWN", "surprise.SMTnews"),
    "STS13": ("FNWN", "headlines", "OnWN"),
    "STS14": ("deft-forum", "deft-news", "headlines", "images", "OnWN", "tweet-news"),
    "STS15": ("answers-forums", "answers-students", "belief", "headlines", "images"),
    "STS16": ("answer-answer", "headlines", "plagiarism", "postediting", "question-question"),
}
STS_TASKS = (*STS_SUBSETS, "STSBenchmark", "SICKRelatedness")


def _read_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"STS evaluation file not found at {path}. "
            "Download the SentEval data described in README.md and set --senteval-dir "
            "to the directory containing STS/ and SICK/."
        )
    return path.read_text(encoding="utf-8").splitlines()


def _make_dataset(rows: list[tuple[str, str, float]], task: str) -> Dataset:
    if len(rows) < 2:
        raise ValueError(f"{task} requires at least two labeled sentence pairs")
    sentence1, sentence2, scores = zip(*rows)
    if not np.isfinite(scores).all():
        raise ValueError(f"{task} contains non-finite similarity scores")
    # SentEval splits sentences on whitespace, then the SimCSE batcher rejoins them.
    return Dataset.from_dict(
        {
            "sentence1": [" ".join(sentence.split()) for sentence in sentence1],
            "sentence2": [" ".join(sentence.split()) for sentence in sentence2],
            "score": list(scores),
        }
    )


def load_sts_suite(data_dir: str | Path) -> dict[str, Dataset]:
    """Read all seven test tasks from a local SentEval downstream directory.

    STS12--16 subsets are concatenated within each year before scoring. Blank
    gold-score lines are skipped together with their corresponding input pair.
    STSBenchmark and SICKRelatedness use their labeled test files only.
    """
    root = Path(data_dir)
    datasets = {}
    for task, subsets in STS_SUBSETS.items():
        rows = []
        directory = root / "STS" / f"{task}-en-test"
        for subset in subsets:
            input_path = directory / f"STS.input.{subset}.txt"
            inputs = _read_lines(input_path)
            scores = _read_lines(directory / f"STS.gs.{subset}.txt")
            if len(inputs) != len(scores):
                raise ValueError(f"Input and gold-score line counts differ for {input_path}")
            for number, (line, score) in enumerate(zip(inputs, scores), start=1):
                if not score.strip():
                    continue
                fields = line.split("\t")
                if len(fields) != 2:
                    raise ValueError(f"Expected two sentences at {input_path}:{number}")
                rows.append((fields[0], fields[1], float(score)))
        datasets[task] = _make_dataset(rows, task)

    for task, path, columns, skip_header in (
        ("STSBenchmark", root / "STS" / "STSBenchmark" / "sts-test.csv", (5, 6, 4), False),
        ("SICKRelatedness", root / "SICK" / "SICK_test_annotated.txt", (1, 2, 3), True),
    ):
        rows = []
        lines = _read_lines(path)
        first, second, score = columns
        for number, line in enumerate(lines[int(skip_header) :], start=1 + int(skip_header)):
            fields = line.split("\t")
            if len(fields) <= max(columns):
                raise ValueError(f"Missing sentence or score columns at {path}:{number}")
            rows.append((fields[first], fields[second], float(fields[score])))
        datasets[task] = _make_dataset(rows, task)
    return datasets


def evaluate_sts_suite(
    model: SentenceBYOL,
    tokenizer: Any,
    datasets: dict[str, Dataset],
    *,
    device: torch.device,
    batch_size: int = 64,
    limit: int | None = None,
) -> dict[str, Any]:
    """Compute cosine correlations and the unweighted seven-task mean, in [-1, 1].

    Unlike STS-B training diagnostics, this does not construct an all-pairs
    similarity matrix or run an SVD. ``limit`` is per task, for smoke tests only.
    Undefined correlations propagate to the mean instead of dropping a task.
    """
    if set(datasets) != set(STS_TASKS):
        raise ValueError(f"STS7 requires exactly these tasks: {', '.join(STS_TASKS)}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if limit is not None and limit < 2:
        raise ValueError("limit must be at least 2 for correlation evaluation")
    for task in STS_TASKS:
        if len(datasets[task]) < 2:
            raise ValueError(f"{task} requires at least two labeled sentence pairs")

    results = {}
    for task in STS_TASKS:
        embedding1, embedding2, scores = encode_stsb_dataset(
            model, tokenizer, datasets[task], device=device, batch_size=batch_size, limit=limit
        )
        similarities = F.cosine_similarity(embedding1.float(), embedding2.float()).numpy()
        results[task] = {
            "sts_spearman": float(spearmanr(similarities, scores).statistic),
            "sts_pearson": float(pearsonr(similarities, scores).statistic),
            "num_pairs": len(scores),
        }
    return {
        "suite": "sts7",
        "split": "test",
        "limit": limit,
        "datasets": results,
        "sts_spearman_mean": float(np.mean([r["sts_spearman"] for r in results.values()])),
        "sts_pearson_mean": float(np.mean([r["sts_pearson"] for r in results.values()])),
    }
