"""sample_benchmark_subset: deterministic sampling and file writes (no real PDFs)."""

import csv
import json
from pathlib import Path

import pytest

from scripts.sample_benchmark_subset import _quoted_company_names, _round1_needed_shas


def test_quoted_names():
    t = ['What is "Acme Corp" revenue vs "Beta Ltd"?', "No quotes here"]
    assert _quoted_company_names([t[0]]) == {"Acme Corp", "Beta Ltd"}
    assert _quoted_company_names([t[1]]) == set()


def test_round1_needed_shas(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "subset.csv").write_text(
        "sha1,company_name\n"
        "aaa111,Acme Corp\n"
        "bbb222,Beta Ltd\n",
        encoding="utf-8",
    )
    shas, rows = _round1_needed_shas(src, ['Revenue of "Acme Corp"?'])
    assert shas == {"aaa111"}
    assert rows == [{"sha1": "aaa111", "company_name": "Acme Corp"}]


def test_manifest_written(tmp_path: Path):
    """End-to-end tiny round1-like folder."""
    src = tmp_path / "bench"
    src.mkdir()
    g = [
        {"question": 'Revenue "Acme Corp"?', "schema": "number", "answer": [1.0]},
        {"question": 'Name "Beta Ltd"?', "schema": "name", "answer": ["x"]},
    ]
    q = [
        {"text": 'Revenue "Acme Corp"?', "kind": "number"},
        {"text": 'Name "Beta Ltd"?', "kind": "name"},
    ]
    (src / "gold_answers.json").write_text(json.dumps(g), encoding="utf-8")
    (src / "questions.json").write_text(json.dumps(q), encoding="utf-8")
    (src / "subset.csv").write_text(
        "sha1,company_name\naaa,Acme Corp\nbbb,Beta Ltd\n", encoding="utf-8"
    )
    pdf = src / "pdf_reports"
    pdf.mkdir()
    (pdf / "aaa.pdf").write_bytes(b"%PDF-1.4 minimal")
    (pdf / "bbb.pdf").write_bytes(b"%PDF-1.4 minimal")

    out = tmp_path / "out"
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[2] / "scripts" / "sample_benchmark_subset.py"
    r = subprocess.run(
        [
            sys.executable,
            str(script),
            "--source-dir",
            str(src),
            "--n",
            "2",
            "--seed",
            "0",
            "--out-dir",
            str(out),
            "--skip-chunked",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    man = json.loads((out / "sample_manifest.json").read_text(encoding="utf-8"))
    assert man["n_sampled"] == 2
    assert len(json.loads((out / "gold_answers.json").read_text(encoding="utf-8"))) == 2
