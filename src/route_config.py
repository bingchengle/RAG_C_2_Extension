"""Multi-route presets (cost/quality) and optional usage accounting."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional


ROUTE_BALANCED = "balanced"
ROUTE_ECONOMY = "economy"
ROUTE_QUALITY = "quality"
ROUTE_IDS = (ROUTE_BALANCED, ROUTE_ECONOMY, ROUTE_QUALITY)


@dataclass
class RouteUsageStats:
    """Thread-safe counters for comparing routes (approximate API usage)."""

    rewrite_calls: int = 0
    comparative_rephrase_calls: int = 0
    llm_rerank_llm_calls: int = 0
    bge_rerank_calls: int = 0
    answer_generation_calls: int = 0
    answer_rewrite_calls: int = 0
    verification_calls: int = 0
    similarity_openai_embedding_batches: int = 0
    similarity_query_embedding_batches: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_rewrite(self) -> None:
        with self._lock:
            self.rewrite_calls += 1

    def record_comparative_rephrase(self) -> None:
        with self._lock:
            self.comparative_rephrase_calls += 1

    def record_llm_rerank(self, n: int = 1) -> None:
        with self._lock:
            self.llm_rerank_llm_calls += n

    def record_bge_rerank(self, n: int = 1) -> None:
        with self._lock:
            self.bge_rerank_calls += n

    def record_answer_generation(self) -> None:
        with self._lock:
            self.answer_generation_calls += 1

    def record_answer_rewrite(self) -> None:
        with self._lock:
            self.answer_rewrite_calls += 1

    def record_verification(self) -> None:
        with self._lock:
            self.verification_calls += 1

    def record_similarity_openai(self, n: int = 1) -> None:
        with self._lock:
            self.similarity_openai_embedding_batches += n

    def record_similarity_query_embedding(self, n: int = 1) -> None:
        with self._lock:
            self.similarity_query_embedding_batches += n

    def to_dict(self) -> Dict[str, int]:
        with self._lock:
            return {
                "rewrite_calls": self.rewrite_calls,
                "comparative_rephrase_calls": self.comparative_rephrase_calls,
                "llm_rerank_llm_calls": self.llm_rerank_llm_calls,
                "bge_rerank_calls": self.bge_rerank_calls,
                "answer_generation_calls": self.answer_generation_calls,
                "answer_rewrite_calls": self.answer_rewrite_calls,
                "verification_calls": self.verification_calls,
                "similarity_openai_embedding_batches": self.similarity_openai_embedding_batches,
                "similarity_query_embedding_batches": self.similarity_query_embedding_batches,
            }


def apply_route_to_run_config(run: Any) -> Any:
    """
    Apply route preset on top of current RunConfig (after --config / --profile).
    `run` must be a RunConfig dataclass instance.
    """
    rid = (getattr(run, "route", None) or ROUTE_BALANCED).lower().strip()
    if rid not in ROUTE_IDS:
        raise ValueError(f"Unknown route {rid!r}; expected one of {ROUTE_IDS}")
    if rid == ROUTE_BALANCED:
        return run

    if rid == ROUTE_ECONOMY:
        sim = getattr(run, "similarity_mode", "openai_large")
        if getattr(run, "enable_similarity_check", False) and sim == "openai_large":
            sim = "same_as_query"
        elif not getattr(run, "enable_similarity_check", False):
            sim = "off"
        return replace(
            run,
            rewrite_model="gpt-4o-mini-2024-07-18",
            answer_model="gpt-4o-mini-2024-07-18",
            verification_model="gpt-4o-mini-2024-07-18",
            comparative_rephrase_model="gpt-4o-mini-2024-07-18",
            llm_rerank_model="gpt-4o-mini-2024-07-18",
            max_verification_rounds=1,
            similarity_mode=sim,
        )


    return replace(
        run,
        rewrite_model="gpt-4o-2024-08-06",
        answer_model="gpt-4o-2024-08-06",
        verification_model="gpt-4o-2024-08-06",
        comparative_rephrase_model="gpt-4o-2024-08-06",
        llm_rerank_model="gpt-4o-2024-08-06",
        max_verification_rounds=max(getattr(run, "max_verification_rounds", 2), 3),
        similarity_mode="openai_large",
    )


def effective_model(explicit: str, fallback: str) -> str:
    return explicit.strip() if explicit and explicit.strip() else fallback
