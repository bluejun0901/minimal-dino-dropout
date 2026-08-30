import numpy as np
import pytest
import torch
from datasets import Dataset

from minimal_dino.evaluation import (
    embedding_diagnostics,
    encode_sentences,
    load_stsb_split,
    stsb_metrics,
)


def test_load_stsb_split_reads_local_parquet(tmp_path):
    expected = Dataset.from_dict(
        {"sentence1": ["first"], "sentence2": ["second"], "score": [0.5]}
    )
    expected.to_parquet(tmp_path / "validation.parquet")

    actual = load_stsb_split(tmp_path, "validation")

    assert actual[:] == expected[:]


def test_load_stsb_split_reports_missing_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="Download the pinned files"):
        load_stsb_split(tmp_path, "validation")


def test_embedding_diagnostics_reports_uniformity():
    embeddings = torch.tensor(
        [[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
    )

    metrics = embedding_diagnostics(embeddings)

    normalized = torch.nn.functional.normalize(embeddings, dim=-1)
    expected_uniformity = torch.exp(-2.0 * torch.pdist(normalized).square()).mean().log()
    assert metrics["uniformity"] == pytest.approx(expected_uniformity.item())


def test_stsb_metrics_reports_alignment_for_scores_higher_than_point_eight():
    embedding1 = torch.tensor([[2.0, 0.0], [0.0, 3.0], [1.0, 0.0]])
    embedding2 = torch.tensor([[0.0, 4.0], [0.0, 2.0], [-1.0, 0.0]])

    metrics = stsb_metrics(
        embedding1,
        embedding2,
        np.asarray([0.8, 1.0, 0.79]),
    )

    assert "uniformity" in metrics
    assert metrics["alignment"] == pytest.approx(0.0)


def test_encode_sentences_does_not_truncate():
    class RecordingTokenizer:
        def __init__(self):
            self.options = None

        def __call__(self, sentences, **options):
            self.options = options
            return {
                "input_ids": torch.ones((len(sentences), 2), dtype=torch.long),
                "attention_mask": torch.ones((len(sentences), 2), dtype=torch.long),
            }

    class FakeModel:
        def eval(self):
            return self

        def encode(self, input_ids, attention_mask):
            return input_ids.float()

    tokenizer = RecordingTokenizer()

    encode_sentences(FakeModel(), tokenizer, ["a sentence"], device=torch.device("cpu"))

    assert tokenizer.options["truncation"] is False
    assert "max_length" not in tokenizer.options


def test_stsb_metrics_reports_nan_alignment_without_positive_pairs():
    embedding1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    embedding2 = torch.tensor([[0.0, 1.0], [0.0, 1.0]])

    metrics = stsb_metrics(embedding1, embedding2, np.asarray([0.1, 0.79]))

    assert np.isnan(metrics["alignment"])
