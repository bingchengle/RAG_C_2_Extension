"""Unit tests for retrieval grid helpers in scripts/compare_baseline_vs_new_params.py."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_compare_script():
    path = ROOT / "scripts" / "compare_baseline_vs_new_params.py"
    spec = importlib.util.spec_from_file_location("compare_baseline_vs_new_params", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parse_csv_ints_and_floats():
    m = _load_compare_script()
    assert m._parse_csv_ints("14, 17,20") == [14, 17, 20]
    assert m._parse_csv_floats("0.22,0.28") == [0.22, 0.28]
    assert m._parse_csv_strs(" llm , bge ") == ["llm", "bge"]


def test_grid_config_suffix_and_combo_count():
    m = _load_compare_script()
    assert m._grid_config_suffix(14, 0.22, "llm", None) == "_grid_t14_b220_llm"
    assert m._grid_config_suffix(14, 0.22, "llm", 0) == "_grid_t14_b220_llm_r1"
    combos = m._iter_grid_combos([14, 20], [0.25], ["llm"], repeats=1)
    assert len(combos) == 2
    combos_r = m._iter_grid_combos([14], [0.25], ["llm"], repeats=2)
    assert len(combos_r) == 2 and combos_r[0][3] == 0 and combos_r[1][3] == 1
