import warnings

from minimal_dino.plot_metrics import parse_metrics, plot_metrics


def test_parse_metrics_groups_training_and_evaluation_records(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        '{"step": 1, "loss": 2.0, "embedding_std": 0.1}\n'
        '{"step": 10, "sts_spearman": 0.5, "embedding_std": 0.2}\n'
        '{"step":'
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        series = parse_metrics(path)

    assert series["train.loss"] == [(1, 2.0)]
    assert series["train.embedding_std"] == [(1, 0.1)]
    assert series["eval.sts_spearman"] == [(10, 0.5)]
    assert series["eval.embedding_std"] == [(10, 0.2)]


def test_plot_metrics_writes_image(tmp_path):
    output = tmp_path / "metrics.png"

    result = plot_metrics({"train.loss": [(1, 2.0), (2, 1.0)]}, output)

    assert result == output
    assert output.stat().st_size > 0
