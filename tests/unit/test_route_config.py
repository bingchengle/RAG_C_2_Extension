"""Unit tests for multi-route RunConfig presets."""

from dataclasses import replace

import pytest

from src.pipeline import RunConfig
from src.route_config import (
    ROUTE_BALANCED,
    ROUTE_ECONOMY,
    ROUTE_QUALITY,
    RouteUsageStats,
    apply_route_to_run_config,
    effective_model,
)


def test_effective_model_prefers_explicit():
    assert effective_model("  gpt-x  ", "fallback") == "gpt-x"


def test_effective_model_falls_back():
    assert effective_model("", "fallback") == "fallback"
    assert effective_model("   ", "fallback") == "fallback"


def test_balanced_route_no_op():
    base = RunConfig(route=ROUTE_BALANCED, answering_model="custom")
    out = apply_route_to_run_config(base)
    assert out is base


def test_economy_sets_mini_models_and_caps_verification():
    base = RunConfig(
        route=ROUTE_ECONOMY,
        enable_similarity_check=True,
        similarity_mode="openai_large",
        max_verification_rounds=5,
    )
    out = apply_route_to_run_config(base)
    assert out.rewrite_model == "gpt-4o-mini-2024-07-18"
    assert out.answer_model == "gpt-4o-mini-2024-07-18"
    assert out.llm_rerank_model == "gpt-4o-mini-2024-07-18"
    assert out.max_verification_rounds == 1
    assert out.similarity_mode == "same_as_query"


def test_economy_similarity_off_when_check_disabled():
    base = RunConfig(
        route=ROUTE_ECONOMY,
        enable_similarity_check=False,
        similarity_mode="openai_large",
    )
    out = apply_route_to_run_config(base)
    assert out.similarity_mode == "off"


def test_economy_preserves_non_openai_similarity_mode_when_check_on():
    base = RunConfig(
        route=ROUTE_ECONOMY,
        enable_similarity_check=True,
        similarity_mode="same_as_query",
    )
    out = apply_route_to_run_config(base)
    assert out.similarity_mode == "same_as_query"


def test_quality_sets_strong_models_and_floor_verification():
    base = RunConfig(route=ROUTE_QUALITY, max_verification_rounds=1)
    out = apply_route_to_run_config(base)
    assert out.answer_model == "gpt-4o-2024-08-06"
    assert out.max_verification_rounds == 3
    assert out.similarity_mode == "openai_large"


def test_quality_raises_verification_floor_not_lowers():
    base = RunConfig(route=ROUTE_QUALITY, max_verification_rounds=5)
    out = apply_route_to_run_config(base)
    assert out.max_verification_rounds == 5


def test_unknown_route_raises():
    bad = replace(RunConfig(), route="nope")
    with pytest.raises(ValueError, match="Unknown route"):
        apply_route_to_run_config(bad)


def test_route_usage_threaded_increments():
    stats = RouteUsageStats()
    errors = []

    def worker():
        try:
            for _ in range(100):
                stats.record_rewrite()
                stats.record_llm_rerank(1)
        except Exception as e:
            errors.append(e)

    import threading

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    d = stats.to_dict()
    assert d["rewrite_calls"] == 400
    assert d["llm_rerank_llm_calls"] == 400
