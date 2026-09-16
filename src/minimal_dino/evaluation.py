from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer

from minimal_dino.data import TokenizeCollator
from minimal_dino.model import SentenceBYOL, checkpoint_model_config


def load_stsb_split(data_dir: str | Path, split: str) -> Any:
    """Load a previously downloaded STS-B Parquet split without accessing the Hub."""
    path = Path(data_dir) / f"{split}.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"STS-B split not found at {path}. Download the pinned files described in README.md."
        )

    from datasets import Dataset

    return Dataset.from_parquet(str(path))


@torch.inference_mode()
def encode_sentences(
    model: SentenceBYOL,
    tokenizer: Any,
    sentences: list[str],
    *,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    model.eval()
    loader = DataLoader(
        sentences,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=TokenizeCollator(tokenizer, max_length=None),
    )
    embeddings = []
    for batch in loader:
        batch = {name: value.to(device) for name, value in batch.items()}
        embeddings.append(model.encode(**batch).cpu())
    return torch.cat(embeddings)


@torch.inference_mode()
def encode_sentence_representations(
    model: SentenceBYOL,
    tokenizer: Any,
    sentences: list[str],
    *,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic encoder embeddings and their projector outputs."""
    model.eval()
    loader = DataLoader(
        sentences,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=TokenizeCollator(tokenizer, max_length=None),
    )
    embeddings = []
    projections = []
    for batch in loader:
        batch = {name: value.to(device) for name, value in batch.items()}
        output = model(**batch, use_dropout=False, target=True)
        embeddings.append(output.embedding.cpu())
        projections.append(output.projection.cpu())
    return torch.cat(embeddings), torch.cat(projections)


def embedding_diagnostics(
    embeddings: torch.Tensor,
) -> dict[str, float]:
    """Cheap collapse indicators on deterministic sentence representations."""
    normalized = F.normalize(embeddings.float(), dim=-1)
    n = normalized.shape[0]
    if n > 1:
        similarities = normalized @ normalized.T
        indices = torch.triu_indices(n, n, offset=1)
        distinct = similarities[indices[0], indices[1]]
        pairwise_mean = distinct.mean().item()
        pairwise_std = distinct.std(unbiased=False).item()
        squared_distances = torch.pdist(normalized, p=2).square()
        uniformity = torch.exp(-2.0 * squared_distances).mean().log().item()
    else:
        pairwise_mean = float("nan")
        pairwise_std = float("nan")
        uniformity = float("nan")

    singular_values = torch.linalg.svdvals(embeddings.float() - embeddings.float().mean(0))
    probabilities = singular_values / singular_values.sum().clamp_min(1e-12)
    effective_rank = torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum()).item()
    return {
        "embedding_std": embeddings.float().std(dim=0, unbiased=False).mean().item(),
        "pairwise_cosine_mean": pairwise_mean,
        "pairwise_cosine_std": pairwise_std,
        "uniformity": uniformity,
        "effective_rank": effective_rank,
    }


def encode_stsb_dataset(
    model: SentenceBYOL,
    tokenizer: Any,
    dataset: Any,
    *,
    device: torch.device,
    batch_size: int = 64,
    limit: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    sentence1 = list(dataset["sentence1"])
    sentence2 = list(dataset["sentence2"])
    score_column = "score" if "score" in dataset.column_names else "label"
    scores = np.asarray(dataset[score_column], dtype=np.float64)

    embedding1 = encode_sentences(
        model,
        tokenizer,
        sentence1,
        device=device,
        batch_size=batch_size,
    )
    embedding2 = encode_sentences(
        model,
        tokenizer,
        sentence2,
        device=device,
        batch_size=batch_size,
    )
    return embedding1, embedding2, scores


def encode_stsb_dataset_with_head(
    model: SentenceBYOL,
    tokenizer: Any,
    dataset: Any,
    *,
    device: torch.device,
    batch_size: int = 64,
    limit: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """Encode STS-B pairs before and after the projector in one pass per sentence side."""
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    sentence1 = list(dataset["sentence1"])
    sentence2 = list(dataset["sentence2"])
    score_column = "score" if "score" in dataset.column_names else "label"
    scores = np.asarray(dataset[score_column], dtype=np.float64)

    embedding1, projection1 = encode_sentence_representations(
        model,
        tokenizer,
        sentence1,
        device=device,
        batch_size=batch_size,
    )
    embedding2, projection2 = encode_sentence_representations(
        model,
        tokenizer,
        sentence2,
        device=device,
        batch_size=batch_size,
    )
    return embedding1, embedding2, projection1, projection2, scores


def stsb_metrics(
    embedding1: torch.Tensor,
    embedding2: torch.Tensor,
    scores: np.ndarray,
) -> dict[str, float]:
    normalized1 = F.normalize(embedding1.float(), dim=-1)
    normalized2 = F.normalize(embedding2.float(), dim=-1)
    similarities = F.cosine_similarity(normalized1, normalized2).numpy()
    positive_mask = torch.as_tensor(scores > 0.8, dtype=torch.bool)
    squared_distances = (normalized1 - normalized2).square().sum(dim=-1)
    alignment = (
        squared_distances[positive_mask].mean().item()
        if positive_mask.any()
        else float("nan")
    )
    metrics = {
        "sts_spearman": float(spearmanr(similarities, scores).statistic),
        "sts_pearson": float(pearsonr(similarities, scores).statistic),
        "alignment": alignment,
    }
    metrics.update(embedding_diagnostics(torch.cat((embedding1, embedding2))))
    return metrics


def stsb_metrics_with_head(
    embedding1: torch.Tensor,
    embedding2: torch.Tensor,
    projection1: torch.Tensor,
    projection2: torch.Tensor,
    scores: np.ndarray,
) -> dict[str, float]:
    """Return encoder metrics plus projector metrics under the ``head/`` prefix."""
    metrics = stsb_metrics(embedding1, embedding2, scores)
    head_metrics = stsb_metrics(projection1, projection2, scores)
    metrics.update({f"head/{name}": value for name, value in head_metrics.items()})
    return metrics


def evaluate_stsb(
    model: SentenceBYOL,
    tokenizer: Any,
    dataset: Any,
    *,
    device: torch.device,
    batch_size: int = 64,
    limit: int | None = None,
) -> dict[str, float]:
    embedding1, embedding2, scores = encode_stsb_dataset(
        model,
        tokenizer,
        dataset,
        device=device,
        batch_size=batch_size,
        limit=limit,
    )
    return stsb_metrics(embedding1, embedding2, scores)


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[SentenceBYOL, Any]:
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config_dict = dict(checkpoint["encoder_config"])
    model_type = config_dict.pop("model_type")
    encoder_config = AutoConfig.for_model(model_type, **config_dict)
    encoder = AutoModel.from_config(encoder_config)
    model = SentenceBYOL(encoder, **checkpoint_model_config(checkpoint))
    model.load_state_dict(checkpoint["student"])
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(path.parent / "tokenizer")
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a sentence checkpoint on STS-B, STS7, or PAWS"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--suite", default="stsb", choices=("stsb", "sts7", "paws"))
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        help="Default: validation for stsb, test for sts7/paws; sts7 is test only",
    )
    parser.add_argument("--stsb-dir", default="data/stsb")
    parser.add_argument(
        "--paws-dir", default="data/paws", help="Local PAWS labeled_final Parquet dir"
    )
    parser.add_argument(
        "--paws-threshold",
        type=float,
        help="Fixed cosine threshold; default: select on PAWS validation by accuracy",
    )
    parser.add_argument(
        "--senteval-dir",
        default="data/senteval",
        help="SentEval downstream directory containing STS/ and SICK/ (sts7 only)",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, help="Limit pairs per task for a smoke test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, help="Also save the JSON results to this file")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.limit is not None and args.limit < 2:
        parser.error("--limit must be at least 2 for correlation evaluation")
    if args.suite == "sts7" and args.split == "validation":
        parser.error("--suite sts7 uses test data; omit --split or pass --split test")
    if args.paws_threshold is not None:
        if args.suite != "paws":
            parser.error("--paws-threshold requires --suite paws")
        if not math.isfinite(args.paws_threshold):
            parser.error("--paws-threshold must be finite")
    if args.suite == "paws" and args.split == "validation" and args.paws_threshold is None:
        parser.error("PAWS validation evaluation requires an explicit --paws-threshold")

    device = torch.device(args.device)
    if args.suite == "paws":
        from minimal_dino.paws_evaluation import evaluate_paws, load_paws_split

        split = args.split or "test"
        dataset = load_paws_split(args.paws_dir, split)
        validation_dataset = (
            load_paws_split(args.paws_dir, "validation") if args.paws_threshold is None else None
        )
        model, tokenizer = load_checkpoint(args.checkpoint, device)
        metrics = evaluate_paws(
            model,
            tokenizer,
            dataset,
            device=device,
            validation_dataset=validation_dataset,
            threshold=args.paws_threshold,
            batch_size=args.batch_size,
            limit=args.limit,
        )
        metrics["split"] = split
    elif args.suite == "sts7":
        from minimal_dino.sts_evaluation import evaluate_sts_suite, load_sts_suite

        datasets = load_sts_suite(args.senteval_dir)
        model, tokenizer = load_checkpoint(args.checkpoint, device)
        metrics = evaluate_sts_suite(
            model,
            tokenizer,
            datasets,
            device=device,
            batch_size=args.batch_size,
            limit=args.limit,
        )
    else:
        dataset = load_stsb_split(args.stsb_dir, args.split or "validation")
        model, tokenizer = load_checkpoint(args.checkpoint, device)
        metrics = evaluate_stsb(
            model,
            tokenizer,
            dataset,
            device=device,
            batch_size=args.batch_size,
            limit=args.limit,
        )
    result = json.dumps(metrics, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
