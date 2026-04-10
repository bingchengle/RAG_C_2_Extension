"""Unit tests for answer text normalization and citation verification (no API)."""

import pytest

from src.answer_generator_optimized import (
    AnswerGeneratorOptimized,
    _normalize_answer_text_for_verify,
)


@pytest.mark.parametrize(
    "raw, check",
    [
        (None, lambda s: s == ""),
        ("plain", lambda s: s == "plain"),
        ({"revenue": 96.8}, lambda s: "96.8" in s),
        ([2023, 2024], lambda s: "2023" in s),
    ],
)
def test_normalize_answer_text_for_verify(raw, check):
    out = _normalize_answer_text_for_verify(raw)
    assert check(out)


def test_verify_citation_accepts_dict_answer_without_regex_error():
    """Regression: dict answer must not reach re.findall as non-string."""
    gen = AnswerGeneratorOptimized.__new__(AnswerGeneratorOptimized)
    page = "2023 fiscal year revenue was 96.8 billion USD."
    ok = gen._verify_citation_content({"year": 2023, "amt": "96.8"}, page)
    assert ok is True
