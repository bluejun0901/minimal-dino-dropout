import json
import sys

import numpy as np
import pytest
import torch
from datasets import Dataset

import minimal_dino.evaluation as evaluation
import minimal_dino.paws_evaluation as paws


def make_dataset(scores=(0.1, 0.4, 0.6, 0.9), labels=(0, 0, 1, 1)):
    return Dataset.from_dict(
        {
            "sentence1": [str(score) for score in scores],
            "sentence2": ["second sentence"] * len(scores),
            "label": list(labels),
        }
    )


def test_load_paws_split_preserves_text_and_binary_labels(tmp_path):
    dataset = Dataset.from_dict(
        {
            "sentence1": ['"Quoted" café', "different order"],
            "sentence2": ["café", "order different"],
            "label": [1, 0],
            "score": [5.0, 5.0],
        }
    )
    dataset.to_parquet(tmp_path / "test.parquet")

    actual = paws.load_paws_split(tmp_path, "test")

    assert actual.column_names == ["sentence1", "sentence2", "label"]
    assert actual["sentence1"] == dataset["sentence1"]
    assert actual["label"] == [1, 0]


def test_load_paws_split_reports_missing_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="PAWS-Wiki labeled_final"):
        paws.load_paws_split(tmp_path, "validation")


@pytest.mark.parametrize(
    "columns, message",
    [
        ({"sentence1": ["a", "b"]}, "columns"),
        ({"sentence1": ["a", "b"], "sentence2": ["c", "d"], "label": [0, -1]}, "labels"),
        ({"sentence1": [None, "b"], "sentence2": ["c", "d"], "label": [0, 1]}, "strings"),
    ],
)
def test_load_paws_split_rejects_invalid_data(tmp_path, columns, message):
    Dataset.from_dict(columns).to_parquet(tmp_path / "test.parquet")
    with pytest.raises(ValueError, match=message):
        paws.load_paws_split(tmp_path, "test")


def test_paws_metrics_known_ranking_and_confusion_matrix():
    metrics = paws.paws_metrics([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1], threshold=0.4)

    assert metrics["roc_auc"] == pytest.approx(0.75)
    assert metrics["average_precision"] == pytest.approx(5 / 6)
    for name in ("accuracy", "precision", "recall", "f1"):
        assert metrics[name] == pytest.approx(0.5)
    for name in ("true_positive", "false_positive", "true_negative", "false_negative"):
        assert metrics[name] == 1


def test_paws_metrics_ties_and_no_predicted_positives():
    metrics = paws.paws_metrics([0.5] * 4, [0, 1, 0, 0], threshold=0.6)
    assert metrics["roc_auc"] == pytest.approx(0.5)
    assert metrics["average_precision"] == pytest.approx(0.25)
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == 0


@pytest.mark.parametrize(
    "scores, labels, message",
    [
        ([0.1, 0.5], [1, 1], "both label classes"),
        ([0.1, np.nan], [0, 1], "finite"),
        ([0.1, 0.5], [0, -1], "labels"),
        ([0.1], [0, 1], "equally sized"),
    ],
)
def test_paws_metrics_rejects_invalid_inputs(scores, labels, message):
    with pytest.raises(ValueError, match=message):
        paws.paws_metrics(scores, labels, threshold=0.5)


def test_threshold_selection_matches_exhaustive_search_with_ties():
    rng = np.random.default_rng(42)
    for _ in range(20):
        scores = rng.integers(-5, 6, size=20) / 5
        labels = np.array([0, 1] * 10)
        rng.shuffle(labels)
        candidates = np.r_[np.unique(scores), np.nextafter(max(scores), np.inf)]
        expected = max(candidates, key=lambda t: (np.mean((scores >= t) == labels), t))
        assert paws.select_paws_threshold(scores, labels) == expected


def test_threshold_selection_can_predict_all_negative_at_cosine_one():
    threshold = paws.select_paws_threshold([1.0, 1.0, 1.0], [0, 0, 1])
    assert threshold > 1.0
    assert paws.paws_metrics([1.0] * 3, [0, 0, 1], threshold=threshold)["accuracy"] == 2 / 3


@pytest.fixture
def fake_encoder(monkeypatch):
    calls = []

    def encode(model, tokenizer, dataset, *, device, batch_size):
        calls.append(dataset)
        scores = torch.tensor([float(value) for value in dataset["sentence1"]])
        first = torch.tensor([[1.0, 0.0]]).repeat(len(scores), 1)
        second = torch.stack((scores, (1 - scores.square()).sqrt()), dim=1)
        return first, second, np.asarray(dataset["label"])

    monkeypatch.setattr(paws, "encode_stsb_dataset", encode)
    return calls


def test_evaluate_paws_calibrates_only_on_validation(fake_encoder):
    result = paws.evaluate_paws(
        None,
        None,
        make_dataset((0.2, 0.5, 0.7, 0.8), (0, 1, 0, 1)),
        device=torch.device("cpu"),
        validation_dataset=make_dataset(),
    )
    assert len(fake_encoder) == 2
    assert result["threshold"] == pytest.approx(0.6)
    assert result["threshold_source"] == "validation_accuracy"
    assert result["validation"]["accuracy"] == 1
    assert result["accuracy"] == result["f1"] == 0.5


def test_evaluate_paws_fixed_threshold_skips_validation_and_honors_limit(fake_encoder):
    result = paws.evaluate_paws(
        None,
        None,
        make_dataset((0.2, 0.8, 0.5, 0.4), (0, 1, 1, 0)),
        device=torch.device("cpu"),
        validation_dataset=make_dataset(),
        threshold=0.5,
        limit=2,
    )
    assert len(fake_encoder) == 1
    assert result["num_pairs"] == 2
    assert result["accuracy"] == 1
    assert result["threshold_source"] == "fixed"
    assert result["validation"] is None


def test_evaluate_paws_requires_calibration_data_or_explicit_threshold():
    with pytest.raises(ValueError, match="validation_dataset"):
        paws.evaluate_paws(None, None, make_dataset(), device=torch.device("cpu"))


@pytest.mark.parametrize("threshold", [None, 0.5])
def test_paws_cli_loads_correct_splits_and_saves_metrics(
    tmp_path, monkeypatch, fake_encoder, capsys, threshold
):
    output = tmp_path / "results" / "paws.json"
    argv = ["eval", "--checkpoint", "fake.pt", "--suite", "paws", "--output", str(output)]
    if threshold is not None:
        argv += ["--paws-threshold", str(threshold)]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(evaluation, "load_checkpoint", lambda *args: (None, None))
    splits = []

    def load(root, split):
        splits.append(split)
        return make_dataset()

    monkeypatch.setattr(paws, "load_paws_split", load)
    evaluation.main()

    assert splits == (["test", "validation"] if threshold is None else ["test"])
    result = json.loads(output.read_text())
    assert result == json.loads(capsys.readouterr().out)
    assert result["split"] == "test"
    assert result["accuracy"] == 1


@pytest.mark.parametrize(
    "options",
    [
        ["--suite", "paws", "--split", "validation"],
        ["--suite", "paws", "--paws-threshold", "nan"],
        ["--suite", "sts7", "--paws-threshold", "0.5"],
    ],
)
def test_paws_cli_rejects_ambiguous_or_invalid_options(monkeypatch, options):
    monkeypatch.setattr(sys, "argv", ["eval", "--checkpoint", "fake.pt", *options])
    with pytest.raises(SystemExit, match="2"):
        evaluation.main()
