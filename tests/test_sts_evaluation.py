import json
import sys

import numpy as np
import pytest
import torch
from datasets import Dataset
from scipy.stats import pearsonr, spearmanr

import minimal_dino.evaluation as evaluation
import minimal_dino.sts_evaluation as sts


@pytest.fixture
def senteval_dir(tmp_path):
    for task, subsets in sts.STS_SUBSETS.items():
        directory = tmp_path / "STS" / f"{task}-en-test"
        directory.mkdir(parents=True)
        for subset in subsets:
            (directory / f"STS.input.{subset}.txt").write_text(
                ' a   sentence \tanother sentence\nskipped\tmissing score\n"quoted"\tlast\n',
                encoding="utf-8",
            )
            (directory / f"STS.gs.{subset}.txt").write_text("1.0\n\n4.5\n")
    benchmark = tmp_path / "STS" / "STSBenchmark"
    benchmark.mkdir()
    (benchmark / "sts-test.csv").write_text(
        "main\tfile\t2017\t1\t2.5\tfirst\tsecond\textra\n"
        'main\tfile\t2017\t2\t5.0\t"quoted"\tother\n',
        encoding="utf-8",
    )
    sick = tmp_path / "SICK"
    sick.mkdir()
    (sick / "SICK_test_annotated.txt").write_text(
        "pair_ID\tsentence_A\tsentence_B\trelatedness_score\tentailment_judgment\n"
        "1\tcat\tdog\t1.5\tNEUTRAL\n2\tman\tperson\t4.8\tENTAILMENT\n",
        encoding="utf-8",
    )
    return tmp_path


def test_load_sts_suite_uses_all_test_subsets_and_skips_missing_labels(senteval_dir):
    # Extra subsets must not silently change the standard evaluation protocol.
    excluded = senteval_dir / "STS" / "STS13-en-test" / "STS.input.SMT.txt"
    excluded.write_text("excluded\tpair\n")
    datasets = sts.load_sts_suite(senteval_dir)

    assert tuple(datasets) == sts.STS_TASKS
    for task, subsets in sts.STS_SUBSETS.items():
        assert len(datasets[task]) == 2 * len(subsets)
        assert datasets[task]["score"] == [1.0, 4.5] * len(subsets)
        assert datasets[task]["sentence1"] == ["a sentence", '"quoted"'] * len(subsets)
    assert datasets["STSBenchmark"]["score"] == [2.5, 5.0]
    assert datasets["STSBenchmark"]["sentence2"] == ["second", "other"]
    assert datasets["SICKRelatedness"]["score"] == [1.5, 4.8]
    assert datasets["SICKRelatedness"]["sentence1"] == ["cat", "man"]


def test_load_sts_suite_rejects_missing_subset(senteval_dir):
    (senteval_dir / "STS" / "STS16-en-test" / "STS.gs.headlines.txt").unlink()
    with pytest.raises(FileNotFoundError, match="STS.gs.headlines.txt"):
        sts.load_sts_suite(senteval_dir)


def test_load_sts_suite_rejects_misaligned_scores(senteval_dir):
    (senteval_dir / "STS" / "STS12-en-test" / "STS.gs.MSRpar.txt").write_text("1\n2\n")
    with pytest.raises(ValueError, match="line counts differ"):
        sts.load_sts_suite(senteval_dir)


@pytest.mark.parametrize("limit", [None, 3])
def test_evaluate_sts_suite_averages_tasks_without_weighting_or_rounding(monkeypatch, limit):
    datasets = {}
    expected_spearman, expected_pearson, counts = [], [], []
    for index, task in enumerate(sts.STS_TASKS):
        size = 4 + index
        similarities = np.linspace(-0.9, 0.9, size)
        scores = np.roll(np.arange(size, dtype=float), index)
        datasets[task] = Dataset.from_dict(
            {
                "sentence1": ["a"] * size,
                "sentence2": ["b"] * size,
                "score": scores,
                "similarity": similarities,
            }
        )
        expected_spearman.append(spearmanr(similarities[:limit], scores[:limit]).statistic)
        expected_pearson.append(pearsonr(similarities[:limit], scores[:limit]).statistic)
        counts.append(min(size, limit) if limit else size)

    def encode(model, tokenizer, dataset, *, device, batch_size, limit):
        assert batch_size == 8
        similarities = torch.tensor(dataset["similarity"][:limit])
        first = torch.tensor([[1.0, 0.0]]).repeat(len(similarities), 1)
        second = torch.stack((similarities, (1 - similarities.square()).sqrt()), dim=1)
        return first, second, np.asarray(dataset["score"][:limit])

    monkeypatch.setattr(sts, "encode_stsb_dataset", encode)
    result = sts.evaluate_sts_suite(
        None, None, datasets, device=torch.device("cpu"), batch_size=8, limit=limit
    )

    assert result["sts_spearman_mean"] == pytest.approx(np.mean(expected_spearman))
    assert result["sts_pearson_mean"] == pytest.approx(np.mean(expected_pearson))
    assert [r["num_pairs"] for r in result["datasets"].values()] == counts
    assert result["limit"] == limit
    if limit is None:
        assert result["sts_spearman_mean"] != pytest.approx(
            np.average(expected_spearman, weights=counts)
        )


def test_evaluate_sts_suite_requires_all_seven_tasks():
    with pytest.raises(ValueError, match="exactly these tasks"):
        sts.evaluate_sts_suite(None, None, {}, device=torch.device("cpu"))


@pytest.mark.parametrize("option", [("limit", 1), ("batch_size", 0)])
def test_evaluate_sts_suite_rejects_invalid_options(senteval_dir, option):
    with pytest.raises(ValueError, match=option[0]):
        sts.evaluate_sts_suite(
            None,
            None,
            sts.load_sts_suite(senteval_dir),
            device=torch.device("cpu"),
            **dict([option]),
        )


def test_sts7_cli_saves_results(senteval_dir, monkeypatch, capsys):
    output = senteval_dir / "results" / "sts7.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval",
            "--checkpoint",
            "fake.pt",
            "--suite",
            "sts7",
            "--senteval-dir",
            str(senteval_dir),
            "--output",
            str(output),
            "--limit",
            "2",
            "--device",
            "cpu",
        ],
    )
    monkeypatch.setattr(evaluation, "load_checkpoint", lambda *args: (None, None))

    def evaluate(model, tokenizer, datasets, **kwargs):
        assert tuple(datasets) == sts.STS_TASKS
        assert kwargs["limit"] == 2
        return {"sts_spearman_mean": 0.75}

    monkeypatch.setattr(sts, "evaluate_sts_suite", evaluate)
    evaluation.main()

    assert json.loads(output.read_text()) == {"sts_spearman_mean": 0.75}
    assert json.loads(capsys.readouterr().out) == json.loads(output.read_text())


def test_stsb_cli_still_defaults_to_validation(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["eval", "--checkpoint", "fake.pt"])
    splits = []
    monkeypatch.setattr(evaluation, "load_stsb_split", lambda root, split: splits.append(split))
    monkeypatch.setattr(evaluation, "load_checkpoint", lambda *args: (None, None))
    monkeypatch.setattr(evaluation, "evaluate_stsb", lambda *args, **kwargs: {"sts_spearman": 1})

    evaluation.main()

    assert splits == ["validation"]
    assert json.loads(capsys.readouterr().out) == {"sts_spearman": 1}


def test_sts7_cli_rejects_validation_split(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval",
            "--checkpoint",
            "fake.pt",
            "--suite",
            "sts7",
            "--split",
            "validation",
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        evaluation.main()
