"""Bundled dataset roots (single place under data/datasets/)."""
from pathlib import Path

from pyprojroot import here

DATASETS_DIR: Path = here() / "data" / "datasets"
TEST_SET_DIR: Path = DATASETS_DIR / "test_set"
BENCHMARK_ROUND1_MINI_DIR: Path = DATASETS_DIR / "benchmark_round1_mini"
BENCHMARK_ROUND1_SOFT_DIR: Path = DATASETS_DIR / "benchmark_round1_soft"
BENCHMARK_ROUND1_FULL_DIR: Path = DATASETS_DIR / "benchmark_round1_full"
ENTERPRISE_RAG_ROUND2_DIR: Path = DATASETS_DIR / "enterprise_rag_round2"
ERC2_SET_DIR: Path = DATASETS_DIR / "erc2_set"
