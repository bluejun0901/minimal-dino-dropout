from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer

from minimal_dino.data import TokenizeCollator
from minimal_dino.model import SentenceDINO


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
    model: SentenceDINO,
    tokenizer: Any,
    sentences: list[str],
    *,
    device: torch.device,
    batch_size: int = 64,
    max_length: int = 32,
) -> torch.Tensor:
    model.eval()
    loader = DataLoader(
        sentences,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=TokenizeCollator(tokenizer, max_length),
    )
    embeddings = []
    for batch in loader:
        batch = {name: value.to(device) for name, value in batch.items()}
        embeddings.append(model.encode(**batch).cpu())
    return torch.cat(embeddings)


def embedding_diagnostics(
    embeddings: torch.Tensor,
    initial_embeddings: torch.Tensor | None = None,
) -> dict[str, float]:
    """Cheap collapse indicators on deterministic, pre-head sentence embeddings."""
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
    metrics = {
        "embedding_std": embeddings.float().std(dim=0, unbiased=False).mean().item(),
        "pairwise_cosine_mean": pairwise_mean,
        "pairwise_cosine_std": pairwise_std,
        "uniformity": uniformity,
        "effective_rank": effective_rank,
    }
    if initial_embeddings is not None:
        if initial_embeddings.shape != embeddings.shape:
            raise ValueError(
                "Initial and current embeddings must have the same shape, got "
                f"{tuple(initial_embeddings.shape)} and {tuple(embeddings.shape)}"
            )
        if n > 1:
            initial_normalized = F.normalize(initial_embeddings.float(), dim=-1)
            initial_similarities = initial_normalized @ initial_normalized.T
            initial_distinct = initial_similarities[indices[0], indices[1]]
            metrics["initial_pairwise_cosine_spearman"] = float(
                spearmanr(
                    distinct.detach().cpu().numpy(),
                    initial_distinct.detach().cpu().numpy(),
                ).statistic
            )
        else:
            metrics["initial_pairwise_cosine_spearman"] = float("nan")
    return metrics


def encode_stsb_dataset(
    model: SentenceDINO,
    tokenizer: Any,
    dataset: Any,
    *,
    device: torch.device,
    batch_size: int = 64,
    max_length: int = 32,
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
        max_length=max_length,
    )
    embedding2 = encode_sentences(
        model,
        tokenizer,
        sentence2,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    return embedding1, embedding2, scores


def stsb_metrics(
    embedding1: torch.Tensor,
    embedding2: torch.Tensor,
    scores: np.ndarray,
    *,
    initial_embeddings: torch.Tensor | None = None,
) -> dict[str, float]:
    similarities = F.cosine_similarity(embedding1, embedding2).numpy()
    metrics = {
        "sts_spearman": float(spearmanr(similarities, scores).statistic),
        "sts_pearson": float(pearsonr(similarities, scores).statistic),
    }
    metrics.update(
        embedding_diagnostics(
            torch.cat((embedding1, embedding2)),
            initial_embeddings=initial_embeddings,
        )
    )
    return metrics


def evaluate_stsb(
    model: SentenceDINO,
    tokenizer: Any,
    dataset: Any,
    *,
    device: torch.device,
    batch_size: int = 64,
    max_length: int = 32,
    limit: int | None = None,
    initial_embeddings: torch.Tensor | None = None,
) -> dict[str, float]:
    embedding1, embedding2, scores = encode_stsb_dataset(
        model,
        tokenizer,
        dataset,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        limit=limit,
    )
    return stsb_metrics(
        embedding1,
        embedding2,
        scores,
        initial_embeddings=initial_embeddings,
    )


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[SentenceDINO, Any]:
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config_dict = dict(checkpoint["encoder_config"])
    model_type = config_dict.pop("model_type")
    encoder_config = AutoConfig.for_model(model_type, **config_dict)
    encoder = AutoModel.from_config(encoder_config)
    model = SentenceDINO(encoder, **checkpoint["head_config"])
    model.load_state_dict(checkpoint["teacher"])
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(path.parent / "tokenizer")
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a minimal DINO checkpoint on STS-B")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="validation", choices=("validation", "test"))
    parser.add_argument("--stsb-dir", default="data/stsb")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = load_stsb_split(args.stsb_dir, args.split)
    model, tokenizer = load_checkpoint(args.checkpoint, device)
    metrics = evaluate_stsb(
        model,
        tokenizer,
        dataset,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        limit=args.limit,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
