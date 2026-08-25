from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from minimal_dino.augmentation import augment_words


class TextLineDataset(Dataset[str]):
    """A plain text dataset with one non-empty sentence per line."""

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Training file not found: {path}")
        with path.open(encoding="utf-8") as handle:
            self.sentences = [line.strip() for line in handle if line.strip()]
        if not self.sentences:
            raise ValueError(f"Training file contains no non-empty sentences: {path}")

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, index: int) -> str:
        return self.sentences[index]


class TokenizeCollator:
    def __init__(self, tokenizer: Any, max_length: int = 32) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, sentences: list[str]) -> dict[str, torch.Tensor]:
        batch = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}


class WordViewCollator:
    """Create two independent word-augmented views, then tokenize them."""

    def __init__(self, tokenizer: Any, max_length: int, strength: float) -> None:
        self.tokenize = TokenizeCollator(tokenizer, max_length)
        self.strength = strength

    def __call__(
        self, sentences: list[str]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        view1 = [augment_words(sentence, self.strength) for sentence in sentences]
        view2 = [augment_words(sentence, self.strength) for sentence in sentences]
        return self.tokenize(view1), self.tokenize(view2)
