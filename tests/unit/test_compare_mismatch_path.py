"""benchmark_compare_round1.mismatch_out_path_for_run"""

from pathlib import Path

import pytest

from src.benchmark import compare_round1 as bc


@pytest.fixture
def bench(tmp_path: Path) -> Path:
    d = tmp_path / "bench"
    d.mkdir()
    return d


def test_explicit_write_mismatches_wins(bench: Path):
    explicit = bench / "custom.json"
    assert bc.mismatch_out_path_for_run(bench, explicit, True) == explicit


def test_export_mismatches_default_file(bench: Path):
    assert bc.mismatch_out_path_for_run(bench, None, True) == bench / "baseline_vs_new_mismatches.json"


def test_no_export_returns_none(bench: Path):
    assert bc.mismatch_out_path_for_run(bench, None, False) is None
