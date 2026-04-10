"""
Build databases/chunked_reports/*.json from pdf_reports/*.pdf, then build
databases/vector_dbs/*.faiss (and optionally databases/bm25_dbs/*.pkl).

Designed for self-contained benchmark folders (e.g. benchmark_answerable_69) where
PDFs exist but chunk/vector assets were never generated.

Usage (from repo root):
  python scripts/build_benchmark_chunks_and_vector_dbs.py \\
    --bench-dir data/datasets/benchmark_answerable_69 \\
    --embedding-provider openai --build-bm25

Requires API keys per src/embedding_clients.py (e.g. OPENAI_API_KEY for openai).
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyPDF2 import PdfReader
from tqdm import tqdm

import faiss

from src.ingestion import BM25Ingestor, VectorDBIngestor


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
        "metainfo": {"company_name": company_name, "sha1_name": sha, "sha1": sha},
        "content": {"pages": pages, "chunks": chunks},
    }


def _load_sha_names(subset_csv: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with subset_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            sha = row.get("sha1") or ""
            if sha:
                out.setdefault(sha, (row.get("company_name") or "").strip())
    return out


def _chunks_nonempty(report: dict) -> bool:
    for c in report.get("content", {}).get("chunks") or []:
        if (c.get("text") or "").strip():
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Chunk PDFs + build FAISS/BM25 for a benchmark folder")
    ap.add_argument(
        "--bench-dir",
        type=Path,
        default=ROOT / "data" / "datasets" / "benchmark_answerable_69",
        help="Folder with pdf_reports/ and subset.csv",
    )
    ap.add_argument(
        "--skip-chunks",
        action="store_true",
        help="Only build vector (and optional BM25); assume chunked_reports already populated",
    )
    ap.add_argument(
        "--skip-vector",
        action="store_true",
        help="Only write chunked JSON from PDFs",
    )
    ap.add_argument(
        "--bm25-only",
        action="store_true",
        help="Only build databases/bm25_dbs/*.pkl from existing chunked_reports (no PDF, no FAISS)",
    )
    ap.add_argument(
        "--embedding-provider",
        choices=("openai", "bge_api"),
        default="openai",
        help="Must match process-questions / retrieval (openai -> stem.faiss, bge_api -> stem_bge.faiss)",
    )
    ap.add_argument("--embedding-model", default="", help="Optional; empty uses env default")
    ap.add_argument(
        "--build-bm25",
        action="store_true",
        help="Also write databases/bm25_dbs/*.pkl (skips reports with no chunk text)",
    )
    ap.add_argument(
        "--overwrite-chunks",
        action="store_true",
        help="Rebuild chunked JSON even if the file already exists",
    )
    args = ap.parse_args()

    bench = args.bench_dir.resolve()
    chunked_dir = bench / "databases" / "chunked_reports"
    vector_dir = bench / "databases" / "vector_dbs"
    bm25_dir = bench / "databases" / "bm25_dbs"

    if args.bm25_only:
        json_paths = sorted(chunked_dir.glob("*.json"))
        if not json_paths:
            raise SystemExit(f"No JSON under {chunked_dir}")
        bm25_dir.mkdir(parents=True, exist_ok=True)
        bm25 = BM25Ingestor()
        n_bm = 0
        for report_path in tqdm(json_paths, desc="chunked_reports -> bm25_dbs"):
            with open(report_path, "r", encoding="utf-8") as f:
                report_data = json.load(f)
            if not _chunks_nonempty(report_data):
                continue
            text_chunks = [c["text"] for c in report_data["content"]["chunks"]]
            idx = bm25.create_bm25_index(text_chunks)
            sha1_name = report_data["metainfo"]["sha1_name"]
            out_pkl = bm25_dir / f"{sha1_name}.pkl"
            with open(out_pkl, "wb") as f:
                pickle.dump(idx, f)
            n_bm += 1
        print(f"bm25_dbs: wrote {n_bm} *.pkl under {bm25_dir}")
        return

    pdf_dir = bench / "pdf_reports"
    subset_csv = bench / "subset.csv"

    if not subset_csv.is_file():
        raise SystemExit(f"Missing {subset_csv}")
    if not pdf_dir.is_dir():
        raise SystemExit(f"Missing {pdf_dir}")

    sha_names = _load_sha_names(subset_csv)

    if not args.skip_chunks:
        chunked_dir.mkdir(parents=True, exist_ok=True)
        pdfs = sorted(pdf_dir.glob("*.pdf"))
        ok = 0
        failed = 0
        for pdf_path in tqdm(pdfs, desc="PDF -> chunked_reports"):
            sha = pdf_path.stem
            dst = chunked_dir / f"{sha}.json"
            if dst.is_file() and not args.overwrite_chunks:
                ok += 1
                continue
            cname = sha_names.get(sha, "")
            try:
                doc = _chunked_doc_from_pdf(pdf_path, sha, cname)
            except Exception as exc:
                print(f"[WARN] Chunk failed {sha}: {exc}", file=sys.stderr)
                failed += 1
                continue
            if not _chunks_nonempty(doc):
                print(f"[WARN] No extractable text for {sha}; skip writing chunk json", file=sys.stderr)
                failed += 1
                continue
            dst.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            ok += 1
        print(f"chunked_reports: wrote/kept ok={ok} failed_or_empty={failed} (total pdfs={len(pdfs)})")

    if args.skip_vector:
        return

    model_kw = (args.embedding_model or "").strip() or None
    ingestor = VectorDBIngestor(
        embedding_provider=args.embedding_provider,
        embedding_model=model_kw,
    )

    json_paths = sorted(chunked_dir.glob("*.json"))
    if not json_paths:
        raise SystemExit(f"No JSON under {chunked_dir}; run without --skip-chunks first")

    vector_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_bge" if ingestor.embedding_provider in {"bge", "bge_api"} else ""
    n_vec = 0
    skipped = 0
    for report_path in tqdm(json_paths, desc="chunked_reports -> vector_dbs"):
        with open(report_path, "r", encoding="utf-8") as f:
            report_data = json.load(f)
        if not _chunks_nonempty(report_data):
            skipped += 1
            continue
        index = ingestor._process_report(report_data)
        sha1_name = report_data["metainfo"]["sha1_name"]
        faiss_file_path = vector_dir / f"{sha1_name}{suffix}.faiss"
        faiss.write_index(index, str(faiss_file_path))
        n_vec += 1
    print(f"vector_dbs: wrote {n_vec} *.faiss (skipped empty chunks: {skipped})")

    if args.build_bm25:
        bm25_dir.mkdir(parents=True, exist_ok=True)
        bm25 = BM25Ingestor()
        n_bm = 0
        for report_path in tqdm(json_paths, desc="chunked_reports -> bm25_dbs"):
            with open(report_path, "r", encoding="utf-8") as f:
                report_data = json.load(f)
            if not _chunks_nonempty(report_data):
                continue
            text_chunks = [c["text"] for c in report_data["content"]["chunks"]]
            idx = bm25.create_bm25_index(text_chunks)
            sha1_name = report_data["metainfo"]["sha1_name"]
            out_pkl = bm25_dir / f"{sha1_name}.pkl"
            with open(out_pkl, "wb") as f:
                pickle.dump(idx, f)
            n_bm += 1
        print(f"bm25_dbs: wrote {n_bm} *.pkl under {bm25_dir}")


if __name__ == "__main__":
    main()
