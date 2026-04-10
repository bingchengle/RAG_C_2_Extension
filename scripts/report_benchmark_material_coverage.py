"""
For a benchmark folder, check each question against:

  Layer 1 — pdf_reports/{sha}.pdf exists for every resolved sha (subset.csv + question text).
  Layer 2 — databases/chunked_reports/{sha}.json exists for every required sha.
  Layer 3 — databases/vector_dbs/{sha}.faiss (and optional {sha}_bge.faiss) for embedding provider.

Resolution rules match merged benchmark sampling: quoted company names + company_name substring.

Example:
  python scripts/report_benchmark_material_coverage.py --bench-dir data/datasets/benchmark_answerable_69
  python scripts/report_benchmark_material_coverage.py --bench-dir data/datasets/benchmark_answerable_69 \\
    --write-filtered-dir data/datasets/benchmark_answerable_68_chunk_ok --min-layer chunk
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _required_shas_for_text(text: str, subset_rows: List[Dict[str, str]]) -> Set[str]:
    name_to_sha = {r["company_name"]: r["sha1"] for r in subset_rows if r.get("company_name")}
    needed: Set[str] = set()
    for name in re.findall(r'"([^"]+)"', text):
        sha = name_to_sha.get(name)
        if sha:
            needed.add(sha)
    for r in subset_rows:
        cn = r.get("company_name") or ""
        if cn and cn in text:
            needed.add(r["sha1"])
    return needed


def _pdf_dir_shas(pdf_dir: Path) -> Set[str]:
    if not pdf_dir.is_dir():
        return set()
    return {p.stem for p in pdf_dir.glob("*.pdf")}


def _chunk_dir_shas(chunk_dir: Path) -> Set[str]:
    if not chunk_dir.is_dir():
        return set()
    return {p.stem for p in chunk_dir.glob("*.json")}


def _vector_dir_shas(vector_dir: Path) -> Set[str]:
    """Shas that have at least one .faiss index (openai or *_bge)."""
    if not vector_dir.is_dir():
        return set()
    shas: Set[str] = set()
    for p in vector_dir.glob("*.faiss"):
        stem = p.stem
        if stem.endswith("_bge"):
            shas.add(stem[: -len("_bge")])
        else:
            shas.add(stem)
    return shas


def analyze_layers(
    bench_dir: Path,
) -> Tuple[
    List[dict],
    List[dict],
    List[Set[str]],
    List[bool],
    List[bool],
    List[bool],
]:
    """
    Returns questions, gold, list of required shas per row, ok_pdf, ok_chunk, ok_vector.
    """
    gold = json.loads((bench_dir / "gold_answers.json").read_text(encoding="utf-8"))
    questions = json.loads((bench_dir / "questions.json").read_text(encoding="utf-8"))
    subset_rows = list(
        csv.DictReader((bench_dir / "subset.csv").open(encoding="utf-8", newline=""))
    )
    have_pdf = _pdf_dir_shas(bench_dir / "pdf_reports")
    have_chunk = _chunk_dir_shas(bench_dir / "databases" / "chunked_reports")
    have_vec = _vector_dir_shas(bench_dir / "databases" / "vector_dbs")

    if len(gold) != len(questions):
        raise SystemExit("gold and questions length mismatch")

    reqs: List[Set[str]] = []
    ok_pdf: List[bool] = []
    ok_chunk: List[bool] = []
    ok_vec: List[bool] = []

    for i, g in enumerate(gold):
        text = g.get("question") or questions[i].get("text") or ""
        req = _required_shas_for_text(text, subset_rows)
        reqs.append(req)
        if not req:
            ok_pdf.append(False)
            ok_chunk.append(False)
            ok_vec.append(False)
            continue
        op = req <= have_pdf
        oc = op and (req <= have_chunk)
        ov = oc and (req <= have_vec)
        ok_pdf.append(op)
        ok_chunk.append(oc)
        ok_vec.append(ov)

    return questions, gold, reqs, ok_pdf, ok_chunk, ok_vec


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Report PDF / chunked / vector DB coverage per question",
    )
    ap.add_argument("--bench-dir", type=Path, default=ROOT / "data/datasets/benchmark_answerable_69")
    ap.add_argument(
        "--write-filtered-dir",
        type=Path,
        default=None,
        help="Write a smaller benchmark keeping only questions that pass the chosen --min-layer.",
    )
    ap.add_argument(
        "--min-layer",
        type=str,
        choices=["pdf", "chunk", "vector"],
        default="pdf",
        help="Which layer must pass for --write-filtered-dir (default: pdf, same as before).",
    )
    args = ap.parse_args()
    bench = args.bench_dir.resolve()

    questions, gold, reqs, ok_pdf, ok_chunk, ok_vec = analyze_layers(bench)
    n = len(gold)

    have_pdf = _pdf_dir_shas(bench / "pdf_reports")
    have_chunk = _chunk_dir_shas(bench / "databases" / "chunked_reports")
    have_vec = _vector_dir_shas(bench / "databases" / "vector_dbs")

    n_pdf = sum(ok_pdf)
    n_chunk = sum(ok_chunk)
    n_vec = sum(ok_vec)
    n_nomapping = sum(1 for r in reqs if not r)

    print(f"Benchmark: {bench}")
    print(f"Total questions: {n}")
    vdir = bench / "databases" / "vector_dbs"
    n_faiss = len(list(vdir.glob("*.faiss"))) if vdir.is_dir() else 0
    print(f"Assets on disk: PDFs={len(have_pdf)} chunked_json={len(have_chunk)} faiss={n_faiss}")
    print()
    print("Layers (each question: all required shas must pass the layer):")
    print(f"  Layer 1 - PDF present:            {n_pdf} / {n}  ({100.0 * n_pdf / n:.1f}%)")
    print(f"  Layer 2 - + chunked_reports JSON: {n_chunk} / {n}  ({100.0 * n_chunk / n:.1f}%)")
    print(f"  Layer 3 - + vector_dbs (*.faiss):  {n_vec} / {n}  ({100.0 * n_vec / n:.1f}%)")
    print(f"  (no subset mapping for question): {n_nomapping}")


    for layer_name, ok_flags, have_set, label in [
        ("chunk", ok_chunk, have_chunk, "chunked JSON"),
        ("vector", ok_vec, have_vec, "FAISS index"),
    ]:
        if layer_name == "chunk":
            candidates = [(i, reqs[i]) for i in range(n) if ok_pdf[i] and not ok_flags[i]]
        else:
            candidates = [(i, reqs[i]) for i in range(n) if ok_chunk[i] and not ok_flags[i]]
        if candidates:
            print(f"\nExamples: pass PDF but fail {label} (first 5):")
            for i, req in candidates[:5]:
                missing = sorted(req - have_set)
                print(f"  idx={i} missing_{layer_name}={missing[:4]}")

    layer_ok = {"pdf": ok_pdf, "chunk": ok_chunk, "vector": ok_vec}[args.min_layer]
    keep_idx = [i for i in range(n) if layer_ok[i]]

    print()
    print(
        f"Using --min-layer {args.min_layer!r}: {len(keep_idx)} / {n} questions would remain after filter."
    )

    if args.write_filtered_dir:
        out = args.write_filtered_dir.resolve()
        out.mkdir(parents=True, exist_ok=True)
        (out / "pdf_reports").mkdir(parents=True, exist_ok=True)
        (out / "databases" / "chunked_reports").mkdir(parents=True, exist_ok=True)
        (out / "databases" / "vector_dbs").mkdir(parents=True, exist_ok=True)
        (out / "databases" / "bm25_dbs").mkdir(parents=True, exist_ok=True)

        fq = [questions[i] for i in keep_idx]
        fg = [gold[i] for i in keep_idx]

        all_needed: Set[str] = set()
        subset_rows = list(
            csv.DictReader((bench / "subset.csv").open(encoding="utf-8", newline=""))
        )
        for i in keep_idx:
            text = gold[i].get("question") or ""
            all_needed |= _required_shas_for_text(text, subset_rows)

        rows_out = []
        seen = set()
        for row in sorted(subset_rows, key=lambda r: r.get("sha1", "")):
            sha = row.get("sha1")
            if sha in all_needed and sha not in seen:
                rows_out.append({"sha1": sha, "company_name": row.get("company_name", "")})
                seen.add(sha)

        with (out / "subset.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["sha1", "company_name"])
            w.writeheader()
            w.writerows(rows_out)

        for sha in sorted(all_needed):
            src_pdf = bench / "pdf_reports" / f"{sha}.pdf"
            dst_pdf = out / "pdf_reports" / f"{sha}.pdf"
            if src_pdf.is_file():
                shutil.copy2(src_pdf, dst_pdf)
            src_ch = bench / "databases" / "chunked_reports" / f"{sha}.json"
            dst_ch = out / "databases" / "chunked_reports" / f"{sha}.json"
            if src_ch.is_file():
                shutil.copy2(src_ch, dst_ch)
            vdir = bench / "databases" / "vector_dbs"
            if vdir.is_dir():
                for pat in (f"{sha}.faiss", f"{sha}_bge.faiss"):
                    src_v = vdir / pat
                    if src_v.is_file():
                        shutil.copy2(src_v, out / "databases" / "vector_dbs" / pat)
                for ext in (".pkl",):
                    for p in vdir.glob(f"{sha}*{ext}"):
                        shutil.copy2(p, out / "databases" / "vector_dbs" / p.name)

        (out / "questions.json").write_text(json.dumps(fq, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "gold_answers.json").write_text(json.dumps(fg, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest = {
            "source": str(bench),
            "min_layer": args.min_layer,
            "kept_indices": keep_idx,
            "kept_count": len(keep_idx),
            "dropped_count": n - len(keep_idx),
            "counts": {"pdf_ok": n_pdf, "chunk_ok": n_chunk, "vector_ok": n_vec},
        }
        (out / "material_filter_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nWrote filtered benchmark: {out} ({len(keep_idx)} questions)")


if __name__ == "__main__":
    main()
