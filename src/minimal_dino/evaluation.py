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


def embedding_diagnostics(embeddings: torch.Tensor) -> dict[str, float]:
    """Cheap collapse indicators on deterministic, pre-head sentence embeddings."""
    normalized = F.normalize(embeddings.float(), dim=-1)
    n = normalized.shape[0]
    if n > 1:
        similarities = normalized @ normalized.T
        indices = torch.triu_indices(n, n, offset=1)
        distinct = similarities[indices[0], indices[1]]
        pairwise_mean = distinct.mean().item()
        pairwise_std = distinct.std(unbiased=False).item()
    else:
        pairwise_mean = float("nan")
        pairwise_std = float("nan")

    singular_values = torch.linalg.svdvals(embeddings.float() - embeddings.float().mean(0))
    probabilities = singular_values / singular_values.sum().clamp_min(1e-12)
    effective_rank = torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum()).item()
    return {
        "embedding_std": embeddings.float().std(dim=0, unbiased=False).mean().item(),
        "pairwise_cosine_mean": pairwise_mean,
        "pairwise_cosine_std": pairwise_std,
        "effective_rank": effective_rank,
    }


def evaluate_stsb(
    model: SentenceDINO,
    tokenizer: Any,
    *,
    device: torch.device,
    split: str = "validation",
    batch_size: int = 64,
    max_length: int = 32,
    limit: int | None = None,
) -> dict[str, float]:
    from datasets import load_dataset

    dataset = load_dataset("sentence-transformers/stsb", split=split)
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
    similarities = F.cosine_similarity(embedding1, embedding2).numpy()
    metrics = {
        "sts_spearman": float(spearmanr(similarities, scores).statistic),
        "sts_pearson": float(pearsonr(similarities, scores).statistic),
    }
    metrics.update(embedding_diagnostics(torch.cat((embedding1, embedding2))))
    return metrics


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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, tokenizer = load_checkpoint(args.checkpoint, device)
    metrics = evaluate_stsb(
        model,
        tokenizer,
        device=device,
        split=args.split,
        batch_size=args.batch_size,
        max_length=args.max_length,
        limit=args.limit,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
