"""Unit tests for retrieval strength and abstention helpers (no external APIs)."""

import pytest

from src.questions_processing import QuestionsProcessor


def _minimal_processor(**kwargs):
    defaults = dict(
        questions_file_path=None,
        vector_db_dir=".",
        bm25_db_dir=".",
        documents_dir=".",
    )
    defaults.update(kwargs)
    return QuestionsProcessor(**defaults)


@pytest.mark.parametrize(
    "docs, expected",
    [
        ([], 0.0),
        ([{"combined_score": 0.55}, {"combined_score": 0.2}], 0.55),
        ([{"relevance_score": 0.8}], 0.8),
        ([{"distance": 0.25}], 0.75),
        ([{"distance": 2.0}], 0.0),
    ],
)
def test_retrieval_strength(docs, expected):
    assert QuestionsProcessor._retrieval_strength(docs) == pytest.approx(expected)


def test_retrieval_strength_prefers_combined_over_relevance_on_same_doc():
    d = {"combined_score": 0.1, "relevance_score": 0.9}
    assert QuestionsProcessor._retrieval_strength([d]) == pytest.approx(0.1)


def test_abstention_value_for_schema():
    assert QuestionsProcessor._abstention_value_for_schema("boolean") is False
    assert QuestionsProcessor._abstention_value_for_schema("number") == "N/A"


def test_evaluate_abstention_no_rules_never_applies():
    p = _minimal_processor(abstain_on_validation_fail=False)
    out = p._evaluate_abstention([{"combined_score": 0.01}], {"validation_passed": False, "confidence": 0.0})
    assert out["apply"] is False
    assert out["reasons"] == []


def test_evaluate_abstention_validation_fail():
    p = _minimal_processor(abstain_on_validation_fail=True)
    out = p._evaluate_abstention([], {"validation_passed": False})
    assert out["apply"] is True
    assert "validation_failed" in out["reasons"]


def test_evaluate_abstention_low_confidence():
    p = _minimal_processor(abstain_min_confidence=0.5)
    out = p._evaluate_abstention([], {"confidence": 0.2, "validation_passed": True})
    assert out["apply"] is True
    assert "low_confidence" in out["reasons"]


def test_evaluate_abstention_weak_retrieval():
    p = _minimal_processor(abstain_min_retrieval_strength=0.9)
    out = p._evaluate_abstention([{"combined_score": 0.1}], {"confidence": 1.0, "validation_passed": True})
    assert out["apply"] is True
    assert "weak_retrieval" in out["reasons"]


def test_evaluate_abstention_confidence_missing_treated_as_zero():
    p = _minimal_processor(abstain_min_confidence=0.01)
    out = p._evaluate_abstention([], {"validation_passed": True})
    assert out["apply"] is True
    assert "low_confidence" in out["reasons"]
