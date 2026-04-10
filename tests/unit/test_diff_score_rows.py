"""benchmark_compare_round1.diff_score_rows"""

from src.benchmark import compare_round1 as bc


def test_diff_score_rows_fixed_regressed_both_wrong():
    # Two _score-like payloads sharing the same three questions
    s_base = {
        "rows": [
            {"question": "Q1", "kind": "number", "correct": False, "pred": 1.0, "gold": [2.0]},
            {"question": "Q2", "kind": "number", "correct": True, "pred": 2.0, "gold": [2.0]},
            {"question": "Q3", "kind": "number", "correct": False, "pred": 3.0, "gold": [9.0]},
        ]
    }
    s_new = {
        "rows": [
            {"question": "Q1", "kind": "number", "correct": True, "pred": 2.0, "gold": [2.0]},
            {"question": "Q2", "kind": "number", "correct": False, "pred": 0.0, "gold": [2.0]},
            {"question": "Q3", "kind": "number", "correct": False, "pred": 8.0, "gold": [9.0]},
        ]
    }
    d = bc.diff_score_rows(s_base, s_new)
    assert len(d["fixed_by_new"]) == 1 and d["fixed_by_new"][0]["question"] == "Q1"
    assert len(d["regressed"]) == 1 and d["regressed"][0]["question"] == "Q2"
    assert len(d["both_wrong"]) == 1 and d["both_wrong"][0]["question"] == "Q3"
    assert d["counts"]["fixed_by_new"] == 1
    assert d["counts"]["regressed"] == 1
    assert d["counts"]["both_wrong"] == 1
