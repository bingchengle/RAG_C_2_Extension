"""Smoke test for scripts/compare_baseline_vs_new_params score logic (via benchmark_compare_round1._score)."""

import json
from pathlib import Path

from src.benchmark import compare_round1 as bc


def _gold() -> list:
    return [
        {
            "question": "Q1?",
            "schema": "number",
            "answer": [42.0],
        },
        {
            "question": "Q2?",
            "schema": "boolean",
            "answer": [True],
        },
    ]


def _pred(q1_val, q2_val) -> dict:
    return {
        "answers": [
            {"question_text": "Q1?", "kind": "number", "value": q1_val},
            {"question_text": "Q2?", "kind": "boolean", "value": q2_val},
        ]
    }


def test_score_baseline_beats_new_when_new_wrong(tmp_path: Path):
    gold_path = tmp_path / "gold.json"
    gold_path.write_text(json.dumps(_gold(), ensure_ascii=False), encoding="utf-8")
    gold_items = json.loads(gold_path.read_text(encoding="utf-8"))

    base = bc._unwrap_answers(_pred(42.0, True))
    new = bc._unwrap_answers(_pred(99.0, True))
    s_base = bc._score(base, gold_items)
    s_new = bc._score(new, gold_items)
    assert s_base["correct"] == 2 and s_base["accuracy"] == 1.0
    assert s_new["correct"] == 1 and s_new["accuracy"] == 0.5
    assert s_new["accuracy"] - s_base["accuracy"] == -0.5


def test_score_strict_includes_gold_all_na_rows():
    """Every gold row is scored; gold-only-N/A matches prediction N/A."""
    gold_items = [
        {"question": "Q1?", "schema": "number", "answer": ["N/A"]},
        {"question": "Q2?", "schema": "number", "answer": [1.0]},
    ]
    pred = [
        {"question_text": "Q1?", "kind": "number", "value": "N/A"},
        {"question_text": "Q2?", "kind": "number", "value": 1.0},
    ]
    s = bc._score(pred, gold_items)
    assert s["protocol"] == "strict"
    assert s["total"] == 2 and s["correct"] == 2 and s["accuracy"] == 1.0


def test_abstention_rollup_on_scored_rows():
    gold_items = [{"question": "Q?", "schema": "number", "answer": [1.0]}]
    pred_wrong = [
        {
            "question_text": "Q?",
            "kind": "number",
            "value": "N/A",
            "abstention": {"applied": True},
        }
    ]
    s = bc._score(pred_wrong, gold_items)
    assert s["abstention_applied_count"] == 1
    assert s["abstention_applied_correct"] == 0
    assert s["abstention_applied_accuracy"] == 0.0

    pred_right = [
        {
            "question_text": "Q?",
            "kind": "number",
            "value": "N/A",
            "abstention": {"applied": True},
        }
    ]
    gold_na_ok = [{"question": "Q?", "schema": "number", "answer": ["N/A", 1.0]}]
    s2 = bc._score(pred_right, gold_na_ok)
    assert s2["abstention_applied_count"] == 1
    assert s2["abstention_applied_correct"] == 1
    assert s2["abstention_applied_accuracy"] == 1.0
