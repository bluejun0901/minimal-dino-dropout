import numpy as np
import pytest
import torch
from datasets import Dataset

import minimal_dino.evaluation as evaluation
from minimal_dino.evaluation import (
    embedding_diagnostics,
    encode_sentences,
    load_checkpoint,
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


def test_load_checkpoint_uses_student_weights(tmp_path, monkeypatch):
    checkpoint = {
        "encoder_config": {"model_type": "fake", "hidden_size": 2},
        "head_config": {
            "projection_dim": 3,
            "projector_hidden_dim": 4,
            "predictor_hidden_dim": 4,
            "pooling": "mean",
        },
        "student": {"weight": torch.tensor([1.0])},
        "teacher": {"weight": torch.tensor([2.0])},
    }

    class FakeModel:
        def __init__(self, encoder, **head_config):
            self.encoder = encoder
            self.head_config = head_config
            self.loaded_state = None

        def load_state_dict(self, state):
            self.loaded_state = state

        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(evaluation.torch, "load", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(
        evaluation.AutoConfig, "for_model", lambda model_type, **config: config
    )
    monkeypatch.setattr(evaluation.AutoModel, "from_config", lambda config: config)
    monkeypatch.setattr(evaluation, "SentenceBYOL", FakeModel)
    monkeypatch.setattr(
        evaluation.AutoTokenizer, "from_pretrained", lambda path: "tokenizer"
    )

    model, tokenizer = load_checkpoint(tmp_path / "checkpoint.pt", torch.device("cpu"))

    assert model.loaded_state is checkpoint["student"]
    assert tokenizer == "tokenizer"


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
