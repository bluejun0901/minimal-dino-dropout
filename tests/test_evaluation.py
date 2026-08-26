import numpy as np
import pytest
import torch
from datasets import Dataset

from minimal_dino.evaluation import embedding_diagnostics, load_stsb_split, stsb_metrics


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


def test_embedding_diagnostics_reports_uniformity_and_initial_geometry_correlation():
    initial = torch.tensor(
        [[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
    )

    metrics = embedding_diagnostics(initial, initial_embeddings=initial.clone())

    normalized = torch.nn.functional.normalize(initial, dim=-1)
    expected_uniformity = torch.exp(-2.0 * torch.pdist(normalized).square()).mean().log()
    assert metrics["uniformity"] == pytest.approx(expected_uniformity.item())
    assert metrics["initial_pairwise_cosine_spearman"] == pytest.approx(1.0)


def test_stsb_metrics_uses_both_sentence_sets_for_geometry():
    embedding1 = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    embedding2 = torch.tensor([[0.8, 0.2], [0.2, 0.8]])
    initial_embeddings = torch.cat((embedding1, embedding2))

    metrics = stsb_metrics(
        embedding1,
        embedding2,
        np.asarray([1.0, 0.0]),
        initial_embeddings=initial_embeddings,
    )

    assert "uniformity" in metrics
    assert metrics["initial_pairwise_cosine_spearman"] == pytest.approx(1.0)
