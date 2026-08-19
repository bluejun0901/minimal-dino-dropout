from types import SimpleNamespace

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


def test_sentence_embedding_is_cls_before_projection_head():
    model = SentenceDINO(
        TinyEncoder(dropout=0.0), output_dim=16, head_hidden_dim=24, bottleneck_dim=8
    )
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }
    expected = model.encoder(**batch).last_hidden_state[:, 0]

    output = model(**batch, use_dropout=False)

    assert torch.equal(output.embedding, expected)
    assert output.logits.shape == (1, 16)
