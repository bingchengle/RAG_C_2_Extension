"""convert_erc_round2_to_gold_answers schema mapping."""

import json
from pathlib import Path

import pytest

from scripts.convert_erc_round2_to_gold_answers import convert


def test_convert_maps_names_to_name_schema(tmp_path: Path):
    d = tmp_path / "ds"
    d.mkdir()
    (d / "questions.json").write_text(
        json.dumps([{"text": "Q1?", "kind": "names"}], ensure_ascii=False),
        encoding="utf-8",
    )
    (d / "answers.json").write_text(
        json.dumps(
            {"Q1?": {"kind": "names", "answers": ["CEO"], "reference_pools": []}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    gold = convert(dataset_dir=d, questions_path=None, answers_path=None)
    assert len(gold) == 1 and gold[0]["schema"] == "name" and gold[0]["answer"] == ["CEO"]


def test_convert_round2_dataset_has_100_rows():
    from src.data_paths import ENTERPRISE_RAG_ROUND2_DIR

    gold = convert(
        dataset_dir=ENTERPRISE_RAG_ROUND2_DIR,
        questions_path=None,
        answers_path=None,
    )
    assert len(gold) == 100
