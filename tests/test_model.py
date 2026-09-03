from types import SimpleNamespace

import pytest
import torch
from torch import nn

from minimal_dino.model import BYOLHead, SentenceBYOL


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


def make_model(**kwargs):
    return SentenceBYOL(
        TinyEncoder(dropout=kwargs.pop("dropout", 0.0)),
        projection_dim=8,
        projector_hidden_dim=24,
        predictor_hidden_dim=16,
        **kwargs,
    )


def test_dropout_views_are_independent_and_encode_is_deterministic():
    torch.manual_seed(0)
    model = SentenceBYOL(
        TinyEncoder(), projection_dim=8, projector_hidden_dim=24, predictor_hidden_dim=16
    )
    batch = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
    }

    model.eval()
    view1 = model(**batch, use_dropout=True).embedding
    view2 = model(**batch, use_dropout=True).embedding
    deterministic1 = model.encode(**batch)
    deterministic2 = model.encode(**batch)

    assert not torch.equal(view1, view2)
    assert torch.equal(deterministic1, deterministic2)
    assert model.training is False
    assert model.encoder.dropout.training is False


def test_sentence_embedding_is_masked_mean_before_byol_head():
    model = make_model()
    batch = {
        "input_ids": torch.tensor([[1, 2, 0]]),
        "attention_mask": torch.tensor([[1, 1, 0]]),
    }
    hidden = model.encoder(**batch).last_hidden_state
    expected = hidden[:, :2].mean(dim=1)

    output = model(**batch, use_dropout=False)

    assert torch.equal(output.embedding, expected)
    assert output.projection.shape == (1, 8)
    assert output.prediction is not None
    assert output.prediction.shape == (1, 8)


def test_target_branch_skips_predictor(monkeypatch):
    model = make_model().eval()
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }

    def fail_if_called(*args, **kwargs):
        raise AssertionError("target branch must not call the predictor")

    monkeypatch.setattr(model.head, "predict", fail_if_called)
    output = model(**batch, use_dropout=False, target=True)

    assert output.prediction is None
    assert torch.isfinite(output.projection).all()


def test_target_center_is_applied_before_projection_with_configured_scale(monkeypatch):
    model = make_model().eval()
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }
    center = torch.arange(12, dtype=torch.float32).unsqueeze(0)
    projector_input = None

    def capture_input(module, args):
        nonlocal projector_input
        projector_input = args[0].detach().clone()

    handle = model.head.projector.register_forward_pre_hook(capture_input)
    output = model(
        **batch, use_dropout=False, target=True, center=center, center_scale=0.25
    )
    handle.remove()

    assert torch.equal(projector_input, output.embedding.float() - center * 0.25)


def test_center_is_rejected_for_online_branch():
    model = make_model()
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }

    with pytest.raises(ValueError, match="target branch"):
        model(**batch, use_dropout=False, center=torch.zeros(1, 12))


def test_target_dropout_uses_override_and_restores_module(monkeypatch):
    model = SentenceBYOL(
        TinyEncoder(dropout=0.5),
        projection_dim=8,
        projector_hidden_dim=24,
        predictor_hidden_dim=16,
    ).eval()
    batch = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
    }
    observed = []
    original_forward = model.encoder.dropout.forward

    def record_probability(value):
        observed.append(model.encoder.dropout.p)
        return original_forward(value)

    monkeypatch.setattr(model.encoder.dropout, "forward", record_probability)
    first = model(
        **batch, use_dropout=True, dropout_probability=0.05, target=True
    ).projection
    second = model(
        **batch, use_dropout=True, dropout_probability=0.05, target=True
    ).projection

    assert observed == [0.05, 0.05]
    assert not torch.equal(first, second)
    assert model.encoder.dropout.p == 0.5
    assert model.encoder.dropout.training is False


def test_sentence_embedding_can_use_cls_pooling():
    model = make_model(pooling="cls")
    batch = {
        "input_ids": torch.tensor([[1, 2, 0]]),
        "attention_mask": torch.tensor([[1, 1, 0]]),
    }
    hidden = model.encoder(**batch).last_hidden_state

    output = model(**batch, use_dropout=False)

    assert torch.equal(output.embedding, hidden[:, 0, :])


def test_byol_head_has_distinct_projector_and_predictor():
    head = BYOLHead(12, projection_dim=8, projector_hidden_dim=24, predictor_hidden_dim=16)
    embedding = torch.randn(1, 12)

    projection = head.project(embedding)
    prediction = head.predict(projection)

    assert projection.shape == prediction.shape == (1, 8)
    assert head.projector[0].in_features == 12
    assert head.predictor[0].in_features == 8


@pytest.mark.parametrize("name", ["projection_dim", "projector_hidden_dim", "predictor_hidden_dim"])
def test_byol_head_rejects_invalid_dimensions(name):
    with pytest.raises(ValueError, match=name):
        BYOLHead(12, **{name: 0})


def test_sentence_byol_rejects_invalid_pooling():
    with pytest.raises(ValueError, match="pooling"):
        SentenceBYOL(TinyEncoder(), pooling="max")


def test_from_pretrained_passes_revision_and_dropout(monkeypatch):
    captured = {}

    def fake_from_pretrained(model_name, **kwargs):
        captured.update(model_name=model_name, **kwargs)
        return TinyEncoder()

    monkeypatch.setattr("minimal_dino.model.AutoModel.from_pretrained", fake_from_pretrained)
    SentenceBYOL.from_pretrained(
        "example/model", revision="immutable-commit", dropout=0.2, projection_dim=8
    )

    assert captured == {
        "model_name": "example/model",
        "revision": "immutable-commit",
        "hidden_dropout_prob": 0.2,
        "attention_probs_dropout_prob": 0.2,
    }


def test_from_pretrained_rejects_invalid_dropout():
    with pytest.raises(ValueError, match="dropout must be"):
        SentenceBYOL.from_pretrained("example/model", dropout=1.0)


def test_from_random_init_loads_only_config_and_applies_dropout(monkeypatch):
    captured = {}
    config = SimpleNamespace(hidden_size=12)

    def fake_config_from_pretrained(model_name, **kwargs):
        captured["config"] = {"model_name": model_name, **kwargs}
        return config

    def fake_model_from_config(actual_config):
        captured["model_config"] = actual_config
        return TinyEncoder()

    monkeypatch.setattr(
        "minimal_dino.model.AutoConfig.from_pretrained", fake_config_from_pretrained
    )
    monkeypatch.setattr("minimal_dino.model.AutoModel.from_config", fake_model_from_config)
    monkeypatch.setattr(
        "minimal_dino.model.AutoModel.from_pretrained",
        lambda *args, **kwargs: pytest.fail("pretrained weights must not be loaded"),
    )

    SentenceBYOL.from_random_init(
        "example/model", revision="immutable-commit", dropout=0.2, projection_dim=8
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
        SentenceBYOL.from_random_init("example/model", dropout=1.0)
