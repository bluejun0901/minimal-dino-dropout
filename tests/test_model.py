from types import SimpleNamespace

import pytest
import torch
from torch import nn

from minimal_dino.model import SentenceDINO


class TinyEncoder(nn.Module):
    def __init__(self, hidden_size: int = 12, dropout: float = 0.5) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(32, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids, attention_mask, return_dict=True):
        hidden = self.projection(self.dropout(self.embedding(input_ids)))
        return SimpleNamespace(last_hidden_state=hidden)


def test_dropout_views_are_independent_and_encode_is_deterministic():
    torch.manual_seed(0)
    model = SentenceDINO(TinyEncoder(), output_dim=16, head_hidden_dim=24, bottleneck_dim=8)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
    }

    model.eval()  # Dropout views must work even for the eval-mode EMA teacher.
    view1 = model(**batch, use_dropout=True).embedding
    view2 = model(**batch, use_dropout=True).embedding
    deterministic1 = model.encode(**batch)
    deterministic2 = model.encode(**batch)

    assert not torch.equal(view1, view2)
    assert torch.equal(deterministic1, deterministic2)
    assert model.training is False
    assert model.encoder.dropout.training is False


def test_sentence_embedding_is_masked_mean_before_projection_head():
    model = SentenceDINO(
        TinyEncoder(dropout=0.0), output_dim=16, head_hidden_dim=24, bottleneck_dim=8
    )
    batch = {
        "input_ids": torch.tensor([[1, 2, 0]]),
        "attention_mask": torch.tensor([[1, 1, 0]]),
    }
    hidden = model.encoder(**batch).last_hidden_state
    expected = hidden[:, :2].mean(dim=1)

    output = model(**batch, use_dropout=False)

    assert torch.equal(output.embedding, expected)
    assert output.logits.shape == (1, 16)


def test_sentence_embedding_can_use_cls_pooling():
    model = SentenceDINO(
        TinyEncoder(dropout=0.0),
        output_dim=16,
        head_hidden_dim=24,
        bottleneck_dim=8,
        pooling="cls",
    )
    batch = {
        "input_ids": torch.tensor([[1, 2, 0]]),
        "attention_mask": torch.tensor([[1, 1, 0]]),
    }
    hidden = model.encoder(**batch).last_hidden_state

    output = model(**batch, use_dropout=False)

    assert torch.equal(output.embedding, hidden[:, 0, :])


@pytest.mark.parametrize("use_mlp", [True, False])
def test_projection_head_can_enable_or_disable_mlp(use_mlp):
    model = SentenceDINO(
        TinyEncoder(dropout=0.0),
        output_dim=16,
        head_hidden_dim=24,
        bottleneck_dim=8,
        use_mlp=use_mlp,
    ).eval()
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }

    output = model(**batch, use_dropout=False, is_teacher=True)

    expected_projection_dim = 8 if use_mlp else 12
    assert model.head.last_weight.shape == (16, expected_projection_dim)
    assert torch.isfinite(output.logits).all()


def test_teacher_head_uses_configured_uniformity_step_size(monkeypatch):
    model = SentenceDINO(
        TinyEncoder(dropout=0.0),
        output_dim=16,
        uniformity_step_size=0.05,
    ).eval()
    captured = {}

    def record_uniformization(embedding, step_size):
        captured["step_size"] = step_size
        return embedding

    monkeypatch.setattr(model.head, "uniformize_embedding", record_uniformization)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
    }

    model(**batch, use_dropout=False, is_teacher=True)

    assert captured["step_size"] == 0.05


def test_zero_uniformity_step_size_disables_teacher_adjustment(monkeypatch):
    model = SentenceDINO(
        TinyEncoder(dropout=0.0),
        output_dim=16,
        uniformity_step_size=0.0,
    ).eval()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("uniformization must be disabled")

    monkeypatch.setattr(model.head, "uniformize_embedding", fail_if_called)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
    }

    output = model(**batch, use_dropout=False, is_teacher=True)

    assert torch.isfinite(output.logits).all()


def test_sentence_dino_rejects_negative_uniformity_step_size():
    with pytest.raises(ValueError, match="uniformity_step_size"):
        SentenceDINO(TinyEncoder(), uniformity_step_size=-0.01)


def test_sentence_dino_rejects_invalid_pooling():
    with pytest.raises(ValueError, match="pooling"):
        SentenceDINO(TinyEncoder(), pooling="max")


def test_from_pretrained_passes_revision_and_dropout(monkeypatch):
    captured = {}

    def fake_from_pretrained(model_name, **kwargs):
        captured.update(model_name=model_name, **kwargs)
        return TinyEncoder()

    monkeypatch.setattr("minimal_dino.model.AutoModel.from_pretrained", fake_from_pretrained)

    SentenceDINO.from_pretrained(
        "example/model", revision="immutable-commit", dropout=0.2, output_dim=16
    )

    assert captured == {
        "model_name": "example/model",
        "revision": "immutable-commit",
        "hidden_dropout_prob": 0.2,
        "attention_probs_dropout_prob": 0.2,
    }


def test_from_pretrained_rejects_invalid_dropout():
    with pytest.raises(ValueError, match="dropout must be"):
        SentenceDINO.from_pretrained("example/model", dropout=1.0)


def test_from_random_init_loads_only_config_and_applies_dropout(monkeypatch):
    captured = {}
    config = SimpleNamespace(hidden_size=12)

    def fake_config_from_pretrained(model_name, **kwargs):
        captured["config"] = {"model_name": model_name, **kwargs}
        return config

    def fake_model_from_config(actual_config):
        captured["model_config"] = actual_config
        return TinyEncoder()

    def fail_if_pretrained_is_loaded(*args, **kwargs):
        raise AssertionError("pretrained weights must not be loaded")

    monkeypatch.setattr(
        "minimal_dino.model.AutoConfig.from_pretrained", fake_config_from_pretrained
    )
    monkeypatch.setattr("minimal_dino.model.AutoModel.from_config", fake_model_from_config)
    monkeypatch.setattr(
        "minimal_dino.model.AutoModel.from_pretrained", fail_if_pretrained_is_loaded
    )

    SentenceDINO.from_random_init(
        "example/model", revision="immutable-commit", dropout=0.2, output_dim=16
    )

    assert captured == {
        "config": {
            "model_name": "example/model",
            "revision": "immutable-commit",
            "hidden_dropout_prob": 0.2,
            "attention_probs_dropout_prob": 0.2,
        },
        "model_config": config,
    }


def test_from_random_init_rejects_invalid_dropout():
    with pytest.raises(ValueError, match="dropout must be"):
        SentenceDINO.from_random_init("example/model", dropout=1.0)
