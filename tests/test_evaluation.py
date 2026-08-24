import pytest
from datasets import Dataset

from minimal_dino.evaluation import load_stsb_split


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
