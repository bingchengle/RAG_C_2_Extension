"""
Randomly sample N questions from a benchmark folder and write a smaller self-contained dataset.

Copies only the PDFs (and optional chunked_reports) needed for the sampled questions so you can
run the pipeline or score on a cheap smoke test.

Examples:
  python scripts/sample_benchmark_subset.py \\
    --source-dir data/datasets/benchmark_round1_full --n 5 --seed 42 \\
    --out-dir data/datasets/benchmark_round1_sample_n5

  python scripts/sample_benchmark_subset.py \\
    --source-dir data/datasets/enterprise_rag_round2 --n 8 --seed 1 \\
    --out-dir data/datasets/erc2_sample_n8

  python scripts/sample_benchmark_subset.py \\
    --source-dir data/datasets/benchmark_answerable_69 --n 10 --seed 2025 \\
    --out-dir data/datasets/benchmark_answerable_69_sample10 --skip-chunked
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyPDF2 import PdfReader


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _quoted_company_names(question_texts: List[str]) -> Set[str]:
    names: Set[str] = set()
    for t in question_texts:
        names.update(re.findall(r'"([^"]+)"', t))
    return names


def _round1_needed_shas(source_dir: Path, selected_texts: List[str]) -> Tuple[Set[str], List[Dict[str, str]]]:
    name_to_sha: Dict[str, str] = {}
    with (source_dir / "subset.csv").open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name_to_sha[row["company_name"]] = row["sha1"]
    needed_shas: Set[str] = set()
    rows_out: List[Dict[str, str]] = []
    for name in sorted(_quoted_company_names(selected_texts)):
        sha = name_to_sha.get(name)
        if sha:
            needed_shas.add(sha)
            rows_out.append({"sha1": sha, "company_name": name})
        else:
            print(f"[WARN] No sha in subset.csv for quoted name: {name!r}", file=sys.stderr)
    rows_out.sort(key=lambda r: r["sha1"])
    return needed_shas, rows_out


def _merged_benchmark_needed_shas(source_dir: Path, selected_texts: List[str]) -> Tuple[Set[str], List[Dict[str, str]]]:
    """Round1 + ERC2 mixed folders: quoted names plus company_name substring match (ERC-style questions)."""
    rows = list(csv.DictReader((source_dir / "subset.csv").open(encoding="utf-8", newline="")))
    name_to_sha = {r["company_name"]: r["sha1"] for r in rows if r.get("company_name")}
    needed: Set[str] = set()
    for t in selected_texts:
        for name in re.findall(r'"([^"]+)"', t):
            sha = name_to_sha.get(name)
            if sha:
                needed.add(sha)
            else:
                print(f"[WARN] No sha in subset.csv for quoted name: {name!r}", file=sys.stderr)
        for r in rows:
            cn = r.get("company_name") or ""
            if cn and cn in t:
                needed.add(r["sha1"])
    rows_out: List[Dict[str, str]] = []
    for r in sorted(rows, key=lambda x: x.get("sha1", "")):
        if r.get("sha1") in needed:
            rows_out.append({"sha1": r["sha1"], "company_name": r.get("company_name", "")})
    return needed, rows_out


def _erc2_needed_shas(source_dir: Path, indices: List[int]) -> Tuple[Set[str], List[Dict[str, str]]]:
    answers_obj = _load_json(source_dir / "answers.json")
    questions = _load_json(source_dir / "questions.json")
    subset_rows: List[Dict[str, str]] = []
    with (source_dir / "subset.csv").open("r", encoding="utf-8", newline="") as f:
        subset_rows = list(csv.DictReader(f))

    needed_shas: Set[str] = set()
    for i in indices:
        text = questions[i]["text"]
        payload = answers_obj.get(text)
        if not payload:
            print(f"[WARN] Missing answers.json entry for index {i}", file=sys.stderr)
            continue
        q_shas: Set[str] = set()
        for pool in payload.get("reference_pools") or []:
            for ref in pool:
                if isinstance(ref, str) and ":" in ref:
                    q_shas.add(ref.split(":", 1)[0])
        if not q_shas:
            for row in subset_rows:
                cn = row.get("company_name") or ""
                if cn and cn in text:
                    q_shas.add(row["sha1"])
        needed_shas |= q_shas
    sha_set = set(needed_shas)
    rows_out: List[Dict[str, str]] = []
    for row in subset_rows:
        if row.get("sha1") in sha_set:
            rows_out.append(row)
    rows_out.sort(key=lambda r: r["sha1"])
    return needed_shas, rows_out


def _write_subset_csv_round1(out_path: Path, rows: List[Dict[str, str]]) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sha1", "company_name"])
        w.writeheader()
        w.writerows(rows)


def _write_subset_csv_erc2(out_path: Path, full_rows: List[Dict[str, str]], needed_shas: Set[str]) -> None:
    picked = [r for r in full_rows if r.get("sha1") in needed_shas]
    picked.sort(key=lambda r: r["sha1"])
    if not picked:
        raise SystemExit("No subset rows matched sampled questions (missing PDF mapping).")
    fieldnames = list(picked[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(picked)


def _chunked_doc_from_pdf(pdf_path: Path, sha: str, company_name: str) -> dict:
    reader = PdfReader(str(pdf_path))
    pages = []
    chunks = []
    for idx, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        pages.append({"page": idx, "text": text})
        chunks.append({"page": idx, "text": text})
    return {
        "metainfo": {
            "company_name": company_name,
            "sha1_name": sha,
            "sha1": sha,
        },
        "content": {"pages": pages, "chunks": chunks},
    }


def _copy_or_build_chunked(
    *,
    source_dir: Path,
    out_dir: Path,
    sha: str,
    company_name: str,
) -> None:
    src_chunk = source_dir / "databases" / "chunked_reports" / f"{sha}.json"
    dst_dir = out_dir / "databases" / "chunked_reports"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{sha}.json"
    if src_chunk.is_file():
        shutil.copy2(src_chunk, dst)
        return
    pdf = out_dir / "pdf_reports" / f"{sha}.pdf"
    if not pdf.is_file():
        print(f"[WARN] No PDF for {sha}; skip chunked json", file=sys.stderr)
        return
    try:
        doc = _chunked_doc_from_pdf(pdf, sha, company_name)
        _write_json(dst, doc)
    except Exception as exc:
        print(f"[WARN] Chunk failed for {sha} ({pdf.name}): {exc}", file=sys.stderr)


def _copy_retrieval_sidecars(*, source_dir: Path, out_dir: Path, sha: str) -> None:
    """Copy per-report FAISS + BM25 if present under source (skip API rebuild on the sample)."""
    v_src = source_dir / "databases" / "vector_dbs"
    v_dst = out_dir / "databases" / "vector_dbs"
    v_dst.mkdir(parents=True, exist_ok=True)
    for name in (f"{sha}.faiss", f"{sha}_bge.faiss"):
        p = v_src / name
        if p.is_file():
            shutil.copy2(p, v_dst / name)
    b_src = source_dir / "databases" / "bm25_dbs"
    b_dst = out_dir / "databases" / "bm25_dbs"
    b_dst.mkdir(parents=True, exist_ok=True)
    bp = b_src / f"{sha}.pkl"
    if bp.is_file():
        shutil.copy2(bp, b_dst / f"{sha}.pkl")


def main() -> None:
    p = argparse.ArgumentParser(description="Random sample of benchmark questions + minimal PDFs/chunks")
    p.add_argument("--source-dir", type=Path, required=True, help="benchmark_round1_* or enterprise_rag_round2")
    p.add_argument("--n", type=int, default=5, help="Number of questions to sample")
    p.add_argument("--seed", type=int, default=42, help="RNG seed (reproducible)")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory (created)")
    p.add_argument(
        "--skip-chunked",
        action="store_true",
        help="Only copy PDFs + JSON metadata; do not build databases/chunked_reports from PDF (faster for large ERC2 PDFs; run pipeline parse+ingest later).",
    )
    p.add_argument(
        "--merged-benchmark",
        action="store_true",
        help="Use merged resolution (quoted + company substring) for subset/PDFs; auto-on for benchmark_answerable_69.",
    )
    args = p.parse_args()

    source_dir = args.source_dir.resolve()
    out_dir = args.out_dir.resolve()
    gold_path = source_dir / "gold_answers.json"
    q_path = source_dir / "questions.json"
    if not gold_path.is_file() or not q_path.is_file():
        raise SystemExit(f"Need gold_answers.json and questions.json under {source_dir}")

    gold: List[dict] = _load_json(gold_path)
    questions: List[dict] = _load_json(q_path)
    if len(gold) != len(questions):
        raise SystemExit(f"Length mismatch: gold {len(gold)} vs questions {len(questions)}")
    for i in range(len(gold)):
        if gold[i].get("question") != questions[i].get("text"):
            raise SystemExit(f"Row {i}: gold question text does not match questions.json")

    n = max(1, min(args.n, len(gold)))
    rng = random.Random(args.seed)
    indices = sorted(rng.sample(range(len(gold)), n))

    selected_gold = [gold[i] for i in indices]
    selected_questions = [questions[i] for i in indices]

    ans_path = source_dir / "answers.json"
    answers_looks_erc2 = ans_path.is_file() and ans_path.read_text(encoding="utf-8").lstrip().startswith("{")
    is_erc2 = "enterprise_rag_round2" in source_dir.name.lower() or answers_looks_erc2
    is_merged_style_dataset = source_dir.name.lower() in (
        "benchmark_answerable_69",
        "benchmark_answerable_68",
    ) or args.merged_benchmark

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pdf_reports").mkdir(parents=True, exist_ok=True)
    (out_dir / "databases" / "vector_dbs").mkdir(parents=True, exist_ok=True)
    (out_dir / "databases" / "bm25_dbs").mkdir(parents=True, exist_ok=True)

    selected_texts = [q["text"] for q in selected_questions]

    if is_erc2:
        needed_shas, _ = _erc2_needed_shas(source_dir, indices)
        full_subset = list(csv.DictReader((source_dir / "subset.csv").open(encoding="utf-8", newline="")))
        _write_subset_csv_erc2(out_dir / "subset.csv", full_subset, needed_shas)
        sha_to_name = {r["sha1"]: r.get("company_name", "") for r in full_subset}
    elif is_merged_style_dataset:
        needed_shas, subset_rows = _merged_benchmark_needed_shas(source_dir, selected_texts)
        _write_subset_csv_round1(out_dir / "subset.csv", subset_rows)
        sha_to_name = {r["sha1"]: r["company_name"] for r in subset_rows}
    else:
        needed_shas, subset_rows = _round1_needed_shas(source_dir, selected_texts)
        _write_subset_csv_round1(out_dir / "subset.csv", subset_rows)
        sha_to_name = {r["sha1"]: r["company_name"] for r in subset_rows}

    if not needed_shas:
        print("[WARN] No PDFs resolved; check subset.csv vs question text.", file=sys.stderr)

    for sha in sorted(needed_shas):
        src_pdf = source_dir / "pdf_reports" / f"{sha}.pdf"
        dst_pdf = out_dir / "pdf_reports" / f"{sha}.pdf"
        if not src_pdf.is_file():
            print(f"[WARN] Missing PDF: {src_pdf}", file=sys.stderr)
            continue
        shutil.copy2(src_pdf, dst_pdf)
        if not args.skip_chunked:
            cn = sha_to_name.get(sha, "")
            _copy_or_build_chunked(source_dir=source_dir, out_dir=out_dir, sha=sha, company_name=cn)
            _copy_retrieval_sidecars(source_dir=source_dir, out_dir=out_dir, sha=sha)

    _write_json(out_dir / "gold_answers.json", selected_gold)
    _write_json(out_dir / "questions.json", selected_questions)

    manifest = {
        "source_dir": str(source_dir),
        "seed": args.seed,
        "n_requested": args.n,
        "n_sampled": n,
        "indices": indices,
        "dataset_kind": "enterprise_rag_round2" if is_erc2 else "round1",
    }
    _write_json(out_dir / "sample_manifest.json", manifest)

    print(f"Wrote {out_dir}")
    print(f"Sampled {n} question(s); indices={indices}")
    print(f"PDFs copied: {len(needed_shas)} sha(s).")
    if args.skip_chunked:
        print("Next: build chunked_reports + vector_dbs (e.g. scripts/build_benchmark_chunks_and_vector_dbs.py), then main.py process-questions from this folder.")
    else:
        print("Next: cd to this folder and run main.py process-questions (indexes copied when present under source).")


if __name__ == "__main__":
    main()
