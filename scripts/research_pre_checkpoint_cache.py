"""Cache the same unlabeled probe views using an existing checkpoint's online encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch

from minimal_dino.augmentation import augment_words
from minimal_dino.evaluation import encode_stsb_dataset, load_checkpoint, load_stsb_split
from minimal_dino.train import save_run_artifacts, set_seed


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    save_run_artifacts(output, args)
    reference = json.loads((Path(args.reference_cache) / "manifest.json").read_text())
    reference_config = json.loads((Path(args.reference_cache) / "config.json").read_text())
    set_seed(reference_config["seed"])
    start = time.monotonic()
    model, tokenizer = load_checkpoint(args.checkpoint, torch.device("cpu"))
    lines = Path("data/wiki1m_for_simcse.txt").read_text().splitlines()
    sentences = [lines[i] for i in reference["train_indices"]]
    rng = random.Random(reference_config["seed"] + 1)
    views = [
        [augment_words(s, reference["augmentation"]["strength"], rng=rng) for s in sentences]
        for _ in range(2)
    ]
    embeddings = []
    for view_index, texts in enumerate(views):
        batches = []
        for offset in range(0, len(texts), reference_config["batch_size"]):
            batch = tokenizer(
                texts[offset : offset + reference_config["batch_size"]],
                padding=True,
                truncation=True,
                max_length=reference["max_train_length"],
                return_tensors="pt",
            )
            batches.append(
                model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_dropout=True,
                    dropout_probability=reference["teacher_dropout"],
                    target=True,
                ).embedding.cpu()
            )
            if offset % 512 == 0:
                print({"view": view_index, "sentences": offset}, flush=True)
        embeddings.append(torch.cat(batches))
    left, right, scores = encode_stsb_dataset(
        model,
        tokenizer,
        load_stsb_split("data/stsb", "validation"),
        device=torch.device("cpu"),
        batch_size=reference_config["batch_size"],
    )
    torch.save(
        {
            "train_views": embeddings,
            "validation_first": left,
            "validation_second": right,
            "scores": torch.tensor(scores),
        },
        output / "features.pt",
    )
    with Path(args.checkpoint).open("rb") as stream:
        checkpoint_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    (output / "manifest.json").write_text(
        json.dumps(
            {
                **reference,
                "kind": "historical_checkpoint_operator_probe_not_matched_training",
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "checkpoint_sha256": checkpoint_sha,
                "uses": "online encoder, not historical EMA teacher",
                "elapsed_seconds": time.monotonic() - start,
                "note": "Same Wiki indices and word views as the reference cache; dropout masks "
                "can differ across encoders because model construction consumes RNG.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
