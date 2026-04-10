"""
Build the full merged 140-question benchmark (Round1 40 + ERC2 100):

  - questions.json + gold_answers.json (same order as data/datasets/merged_benchmark_140_qa.json)
  - subset.csv + pdf_reports + databases/chunked_reports (for run_bench_answers / process-questions)

Usage:
  python scripts/build_benchmark_merged_140.py
  python scripts/build_benchmark_merged_140.py --out-dir data/datasets/benchmark_merged_140

Then (from repo root) index + answer:
  python scripts/run_bench_answers.py --bench-dir data/datasets/benchmark_merged_140

Or score existing predictions vs gold:
  python scripts/compare_baseline_vs_new_params.py score \\
    --gold data/datasets/benchmark_merged_140/gold_answers.json \\
    --baseline path/to/answers_a.json --new path/to/answers_b.json

  python benchmark_compare_round1.py --bench-dir data/datasets/benchmark_merged_140 --score-only
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyPDF2 import PdfReader

from src.data_paths import DATASETS_DIR, ENTERPRISE_RAG_ROUND2_DIR

CHALLENGE_R1 = ROOT / "_challenge_source" / "round1"
R1_CHUNK_SOURCE = DATASETS_DIR / "benchmark_round1_full" / "databases" / "chunked_reports"
ERC2_CHUNK_SOURCE = ENTERPRISE_RAG_ROUND2_DIR / "databases" / "chunked_reports"


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def _quoted_names(text: str) -> Set[str]:
    import re

    return set(re.findall(r'"([^"]+)"', text))


def _r1_name_to_sha() -> Dict[str, str]:
    m: Dict[str, str] = {}
    with (CHALLENGE_R1 / "dataset.csv").open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            m[row["name"]] = row["sha1"]
    return m


def _erc2_shas_for_indices(indices: List[int]) -> Tuple[Set[str], List[dict]]:
    answers_obj = _load(ENTERPRISE_RAG_ROUND2_DIR / "answers.json")
    questions = _load(ENTERPRISE_RAG_ROUND2_DIR / "questions.json")
    subset_rows = list(
        csv.DictReader((ENTERPRISE_RAG_ROUND2_DIR / "subset.csv").open(encoding="utf-8", newline=""))
    )
    needed: Set[str] = set()
    for i in indices:
        text = questions[i]["text"]
        payload = answers_obj.get(text)
        if not payload:
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
        needed |= q_shas
    return needed, subset_rows


def _chunked_from_pdf(pdf_path: Path, sha: str, company_name: str) -> dict:
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
        "metainfo": {"company_name": company_name, "sha1_name": sha, "sha1": sha},
        "content": {"pages": pages, "chunks": chunks},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build merged 140-question benchmark folder")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=DATASETS_DIR / "benchmark_merged_140",
        help="Output directory",
    )
    ap.add_argument(
        "--skip-chunked",
        action="store_true",
        help="Do not write databases/chunked_reports (run ingest later).",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 2 if any PDF in subset is missing from source trees.",
    )
    args = ap.parse_args()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "pdf_reports").mkdir(parents=True, exist_ok=True)
    (out / "databases" / "chunked_reports").mkdir(parents=True, exist_ok=True)
    (out / "databases" / "vector_dbs").mkdir(parents=True, exist_ok=True)
    (out / "databases" / "bm25_dbs").mkdir(parents=True, exist_ok=True)

    r1_all = _load(CHALLENGE_R1 / "answers.json")
    if len(r1_all) != 40:
        raise SystemExit(f"Expected 40 Round1 rows, got {len(r1_all)}")

    erc_indices = list(range(100))
    erc_gold_full = _load(ENTERPRISE_RAG_ROUND2_DIR / "gold_answers.json")
    erc_q_full = _load(ENTERPRISE_RAG_ROUND2_DIR / "questions.json")
    if len(erc_gold_full) != 100 or len(erc_q_full) != 100:
        raise SystemExit("Expected 100 ERC2 questions/gold rows.")

    questions_out: List[dict] = []
    gold_out: List[dict] = []

    for a in r1_all:
        questions_out.append({"text": a["question"], "kind": a["schema"]})
        gold_out.append(
            {
                "question": a["question"],
                "schema": a["schema"],
                "answer": a["answer"],
                **({"comment": a["comment"]} if a.get("comment") else {}),
                **({"category": a["category"]} if a.get("category") else {}),
            }
        )

    for g, q in zip(erc_gold_full, erc_q_full):
        questions_out.append({"text": q["text"], "kind": q["kind"]})
        gold_out.append(dict(g))

    subset_merged: Dict[str, str] = {}
    name_to_sha = _r1_name_to_sha()

    for a in r1_all:
        for name in _quoted_names(a["question"]):
            sha = name_to_sha.get(name)
            if sha:
                subset_merged[sha] = name
            else:
                print(f"[WARN] Round1 unknown quoted name (no PDF row): {name!r}", file=sys.stderr)

    erc_shas, erc_subset_table = _erc2_shas_for_indices(erc_indices)
    sha_to_company_erc = {r["sha1"]: r.get("company_name", "") for r in erc_subset_table}
    for sha in erc_shas:
        subset_merged[sha] = sha_to_company_erc.get(sha, "")

    rows_csv = [{"sha1": s, "company_name": subset_merged[s]} for s in sorted(subset_merged.keys())]
    with (out / "subset.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sha1", "company_name"])
        w.writeheader()
        w.writerows(rows_csv)

    missing_pdf: List[Dict[str, str]] = []
    for sha, cname in subset_merged.items():
        dst = out / "pdf_reports" / f"{sha}.pdf"
        src_r1 = CHALLENGE_R1 / "pdfs" / f"{sha}.pdf"
        src_e = ENTERPRISE_RAG_ROUND2_DIR / "pdf_reports" / f"{sha}.pdf"
        if src_r1.is_file():
            shutil.copy2(src_r1, dst)
        elif src_e.is_file():
            shutil.copy2(src_e, dst)
        else:
            missing_pdf.append({"sha1": sha, "company_name": cname})
            print(f"[WARN] Missing PDF for sha {sha} ({cname})", file=sys.stderr)

    if not args.skip_chunked:
        for sha, cname in subset_merged.items():
            dst_c = out / "databases" / "chunked_reports" / f"{sha}.json"
            if (R1_CHUNK_SOURCE / f"{sha}.json").is_file():
                shutil.copy2(R1_CHUNK_SOURCE / f"{sha}.json", dst_c)
                continue
            if (ERC2_CHUNK_SOURCE / f"{sha}.json").is_file():
                shutil.copy2(ERC2_CHUNK_SOURCE / f"{sha}.json", dst_c)
                continue
            pdf = out / "pdf_reports" / f"{sha}.pdf"
            if pdf.is_file():
                doc = _chunked_from_pdf(pdf, sha, cname)
                dst_c.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")

    (out / "questions.json").write_text(json.dumps(questions_out, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "gold_answers.json").write_text(json.dumps(gold_out, ensure_ascii=False, indent=2), encoding="utf-8")

    n_chunk = len(list((out / "databases" / "chunked_reports").glob("*.json")))
    manifest = {
        "total_questions": len(gold_out),
        "round1": len(r1_all),
        "erc2": len(erc_indices),
        "pdfs_expected": len(subset_merged),
        "pdfs_missing": len(missing_pdf),
        "missing_pdfs": missing_pdf,
        "chunked_json_count": n_chunk,
        "out_dir": str(out),
    }
    (out / "build_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("\nNext: python scripts/run_bench_answers.py --bench-dir", str(out))
    if missing_pdf:
        print(
            f"\n[WARN] {len(missing_pdf)} PDF(s) missing — copy them into pdf_reports/ then rebuild chunks/indexes if needed.",
            file=sys.stderr,
        )
        if args.strict:
            sys.exit(2)


if __name__ == "__main__":
    main()
