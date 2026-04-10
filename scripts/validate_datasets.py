"""Validate dataset folders for duplicate sha1, question text, or PDF names."""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_paths import DATASETS_DIR


def _check_subset_csv(path: Path, errors: list[str]) -> None:
    if not path.is_file():
        return
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    sha1s = [r.get("sha1", "").strip() for r in rows]
    dup = [s for s, c in Counter(sha1s).items() if c > 1 and s]
    if dup:
        errors.append(f"{path}: duplicate sha1 in subset.csv: {dup[:5]}")
    names = [r.get("company_name", "").strip() for r in rows]
    dupn = [n for n, c in Counter(names).items() if c > 1 and n]
    if dupn:
        errors.append(f"{path}: duplicate company_name: {dupn[:5]}")


def _check_questions(path: Path, errors: list[str]) -> None:
    if not path.is_file():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return
    texts = [item.get("text") or item.get("question", "") for item in data]
    dup = [t for t, c in Counter(texts).items() if c > 1 and t]
    if dup:
        errors.append(f"{path}: duplicate question text ({len(dup)}), e.g. {dup[0][:100]}...")


def _check_pdfs(dir_path: Path, errors: list[str]) -> None:
    if not dir_path.is_dir():
        return
    names = [f.name.lower() for f in dir_path.glob("*.pdf")]
    dup = [n for n, c in Counter(names).items() if c > 1]
    if dup:
        errors.append(f"{dir_path}: duplicate pdf filenames: {dup}")


def _check_subset_has_chunked_reports(ds: Path, errors: list[str]) -> None:
    """Each sha1 in subset.csv must have chunked_reports/{sha1}.json (RAG retrieval depends on it)."""
    if not ds.name.startswith("benchmark_round1"):

        return
    subset_path = ds / "subset.csv"
    chunked_dir = ds / "databases" / "chunked_reports"
    if not subset_path.is_file() or not chunked_dir.is_dir():
        return
    if not any(chunked_dir.glob("*.json")):

        return
    rows = list(csv.DictReader(subset_path.open(encoding="utf-8")))
    missing: list[str] = []
    for r in rows:
        sha = (r.get("sha1") or "").strip()
        name = (r.get("company_name") or "").strip()
        if not sha:
            continue
        if not (chunked_dir / f"{sha}.json").is_file():
            missing.append(f"{sha} ({name})")
    if missing:
        errors.append(
            f"{ds.name}: subset.csv has {len(missing)} companies without databases/chunked_reports/<sha1>.json: "
            + "; ".join(missing[:12])
            + (" ..." if len(missing) > 12 else "")
        )


def _check_subset_json_csv(er: Path, errors: list[str]) -> None:
    js, cs = er / "subset.json", er / "subset.csv"
    if js.is_file() and cs.is_file():
        sj = json.loads(js.read_text(encoding="utf-8"))
        rows = list(csv.DictReader(cs.open(encoding="utf-8")))
        if len(sj) != len(rows):
            errors.append(
                f"{er}: subset.json ({len(sj)}) vs subset.csv ({len(rows)}) length mismatch"
            )


def main() -> int:
    errors: list[str] = []
    if not DATASETS_DIR.is_dir():
        print(f"Missing {DATASETS_DIR}", file=sys.stderr)
        return 1

    for ds in sorted(DATASETS_DIR.iterdir()):
        if not ds.is_dir():
            continue
        _check_subset_csv(ds / "subset.csv", errors)
        _check_subset_has_chunked_reports(ds, errors)
        _check_questions(ds / "questions.json", errors)
        _check_pdfs(ds / "pdf_reports", errors)
        if ds.name == "enterprise_rag_round2":
            _check_subset_json_csv(ds, errors)

    if errors:
        print("FAILED", file=sys.stderr)
        for e in errors:
            print(e, file=sys.stderr)
        return 1
    print("OK: no internal duplicates in scanned datasets under", DATASETS_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
