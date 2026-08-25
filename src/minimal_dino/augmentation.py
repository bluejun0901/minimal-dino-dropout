from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Literal

WordOperation = Literal["repetition", "deletion", "replacement", "random"]
OPERATIONS = ("repetition", "deletion", "replacement")


def augment_words(
    text: str,
    strength: float,
    *,
    operation: WordOperation = "random",
    vocabulary: Sequence[str] | None = None,
    rng: random.Random | None = None,
) -> str:
    """Apply simple word-level augmentation to whitespace-separated text."""
    if not 0.0 <= strength <= 1.0:
        raise ValueError("augmentation strength must be in [0, 1]")
    if operation not in (*OPERATIONS, "random"):
        raise ValueError(f"unknown word augmentation operation: {operation}")
    if strength == 0.0 or not text:
        return text

    generator = rng or random
    words = text.split()
    if not words:
        return text
    selected = [generator.random() < strength for _ in words]
    if not any(selected):
        return text

    output: list[str] = []
    for index, word in enumerate(words):
        if not selected[index]:
            output.append(word)
            continue
        chosen = generator.choice(OPERATIONS) if operation == "random" else operation
        if chosen == "repetition":
            output.extend((word, word))
        elif chosen == "replacement":
            candidates = [candidate for candidate in (vocabulary or words) if candidate != word]
            output.append(generator.choice(candidates) if candidates else word)
        # Deletion deliberately appends nothing.

    # Do not allow deletion (or an empty replacement vocabulary) to erase the sentence.
    return " ".join(output) if output else generator.choice(words)
