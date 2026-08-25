import random

from minimal_dino.augmentation import augment_words


def test_zero_strength_is_exact_identity():
    text = "  spacing stays\texact  "
    assert augment_words(text, 0.0, rng=random.Random(1)) == text


def test_repetition_repeats_every_word():
    assert augment_words("one two", 1.0, operation="repetition", rng=random.Random(1)) == (
        "one one two two"
    )


def test_deletion_keeps_one_word_instead_of_empty_sentence():
    result = augment_words("one two", 1.0, operation="deletion", rng=random.Random(1))
    assert result in {"one", "two"}


def test_replacement_uses_available_vocabulary():
    result = augment_words(
        "one two", 1.0, operation="replacement", vocabulary=["other"], rng=random.Random(1)
    )
    assert result == "other other"


def test_seed_reproduces_random_operations():
    first = augment_words("one two three four", 0.7, rng=random.Random(42))
    second = augment_words("one two three four", 0.7, rng=random.Random(42))
    assert first == second
