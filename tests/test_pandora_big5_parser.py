"""Tests for the pandora_big5 adapter."""

import json

import pytest

datasets = pytest.importorskip("datasets")

from src.data.pandora_big5_parser import PandoraBig5Parser


def _write_parquet(path, rows: dict) -> None:
    datasets.Dataset.from_dict(rows).to_parquet(str(path))


def _read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_pandora_big5_parser_preserves_split_mapping_and_labels(tmp_path):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    raw_dir.mkdir()

    base_rows = {
        "O": [80.0],
        "C": [20.0],
        "E": [75.0],
        "A": [40.0],
        "N": [60.0],
        "ptype": [26],
        "text": [
            "This public mirror example has enough words to pass the cleaning threshold safely."
        ],
        "__index_level_0__": [12345],
    }
    _write_parquet(raw_dir / "train-00000-of-00001.parquet", base_rows)
    _write_parquet(
        raw_dir / "validation-00000-of-00001.parquet",
        {
            **base_rows,
            "text": ["Another validation example contains enough words for preprocessing to keep it."],
            "__index_level_0__": [12346],
        },
    )
    _write_parquet(
        raw_dir / "test-00000-of-00001.parquet",
        {
            **base_rows,
            "text": ["A separate test example also contains plenty of words for the parser."],
            "__index_level_0__": [12347],
        },
    )

    parser = PandoraBig5Parser({"log_every": 0})
    parser.run(str(raw_dir), str(output_dir))

    train_records = _read_jsonl(output_dir / "train.jsonl")
    val_records = _read_jsonl(output_dir / "val.jsonl")
    test_records = _read_jsonl(output_dir / "test.jsonl")

    assert len(train_records) == 1
    assert len(val_records) == 1
    assert len(test_records) == 1

    record = train_records[0]
    assert record["split"] == "train"
    assert record["source"] == "pandora_big5"
    assert record["label_ocean"] == {
        "O": "HIGH",
        "C": "LOW",
        "E": "HIGH",
        "A": "LOW",
        "N": "HIGH",
    }
    assert record["metadata"]["raw_row_id"] == 12345
    assert record["metadata"]["bigfive_raw"]["openness"] == 80.0
