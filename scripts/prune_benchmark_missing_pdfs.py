"""
Remove from a benchmark folder any question whose retrieval needs a SHA that has no PDF on disk.

Reads `_challenge_source/round1` and `enterprise_rag_round2` to resolve which SHAs each
global question needs (same rules as build_benchmark_merged_140).

Usage:
  python scripts/prune_benchmark_missing_pdfs.py --bench-dir data/datasets/benchmark_merged_140 --dry-run
  python scripts/prune_benchmark_missing_pdfs.py --bench-dir data/datasets/benchmark_merged_140
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

R1 = ROOT / "_challenge_source" / "round1"
ERC = ROOT / "data" / "datasets" / "enterprise_rag_round2"


def _missing_shas(bench: Path) -> set[str]:
    pdf_dir = bench / "pdf_reports"
    out: set[str] = set()
    with (bench / "subset.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            sha = row.get("sha1") or ""
            if sha and not (pdf_dir / f"{sha}.pdf").is_file():
                out.add(sha)
    return out


def _r1_name_to_sha() -> dict[str, str]:
    m: dict[str, str] = {}
    with (R1 / "dataset.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            m[row["name"]] = row["sha1"]
    return m


def _r1_question_shas(text: str, name_to_sha: dict[str, str]) -> set[str]:
    return {name_to_sha[n] for n in re.findall(r'"([^"]+)"', text) if n in name_to_sha}


def _erc_question_shas(
    text: str,
    answers_obj: dict,
    subset_rows: list[dict],
) -> set[str]:
    payload = answers_obj.get(text)
    q_shas: set[str] = set()
    if payload:
        for pool in payload.get("reference_pools") or []:
            for ref in pool:
                if isinstance(ref, str) and ":" in ref:
                    q_shas.add(ref.split(":", 1)[0])
    if not q_shas:
        for row in subset_rows:
            cn = row.get("company_name") or ""
            if cn and cn in text:
                q_shas.add(row["sha1"])
    return q_shas


def _needed_shas_for_global(
    gi: int,
    *,
    r1_list: list[dict],
    name_to_sha: dict[str, str],
    questions_erc: list[dict],
    answers_obj: dict,
    subset_erc: list[dict],
) -> set[str]:
    if gi < 40:
        return _r1_question_shas(r1_list[gi]["question"], name_to_sha)
    return _erc_question_shas(questions_erc[gi - 40]["text"], answers_obj, subset_erc)


def main() -> None:
    ap = argparse.ArgumentParser(description="Drop questions that need a SHA with no local PDF")
    ap.add_argument("--bench-dir", type=Path, required=True)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print indices only; do not write files",
    )
    args = ap.parse_args()
    bench: Path = args.bench_dir.resolve()
    if not (bench / "questions.json").is_file():
        raise SystemExit(f"Missing questions.json: {bench}")

    miss = _missing_shas(bench)
    if not miss:
        print("No missing PDFs under pdf_reports; nothing to prune.")
        return

    name_to_sha = _r1_name_to_sha()
    r1_list = json.loads((R1 / "answers.json").read_text(encoding="utf-8"))
    answers_obj = json.loads((ERC / "answers.json").read_text(encoding="utf-8"))
    questions_erc = json.loads((ERC / "questions.json").read_text(encoding="utf-8"))
    subset_erc = list(csv.DictReader((ERC / "subset.csv").open(encoding="utf-8", newline="")))

    drop_global: list[int] = []
    for gi in range(140):
        sh = _needed_shas_for_global(
            gi,
            r1_list=r1_list,
            name_to_sha=name_to_sha,
            questions_erc=questions_erc,
            answers_obj=answers_obj,
            subset_erc=subset_erc,
        )
        if sh & miss:
            drop_global.append(gi)

    print(f"Missing-PDF SHAs ({len(miss)}): {sorted(miss)}")
    print(f"Questions to remove ({len(drop_global)}): global indices {drop_global}")

    if args.dry_run:
        return

    q_all = json.loads((bench / "questions.json").read_text(encoding="utf-8"))
    g_all = json.loads((bench / "gold_answers.json").read_text(encoding="utf-8"))
    if len(q_all) != len(g_all):
        raise SystemExit("questions.json and gold_answers.json length mismatch")
    drop_set = set(drop_global)
    keep_idx = [i for i in range(len(q_all)) if i not in drop_set]
    q_new = [q_all[i] for i in keep_idx]
    g_new = [g_all[i] for i in keep_idx]

    backup = bench / "_backup_before_prune_missing_pdf"
    if not backup.exists():
        backup.mkdir(parents=True)
        shutil.copy2(bench / "questions.json", backup / "questions.json")
        shutil.copy2(bench / "gold_answers.json", backup / "gold_answers.json")
        shutil.copy2(bench / "subset.csv", backup / "subset.csv")

    (bench / "questions.json").write_text(json.dumps(q_new, ensure_ascii=False, indent=2), encoding="utf-8")
    (bench / "gold_answers.json").write_text(json.dumps(g_new, ensure_ascii=False, indent=2), encoding="utf-8")

    needed: set[str] = set()
    for gi in keep_idx:
        needed |= _needed_shas_for_global(
            gi,
            r1_list=r1_list,
            name_to_sha=name_to_sha,
            questions_erc=questions_erc,
            answers_obj=answers_obj,
            subset_erc=subset_erc,
        )

    by_sha = {r["sha1"]: r for r in csv.DictReader((bench / "subset.csv").open(encoding="utf-8", newline=""))}
    rows_out = []
    for sha in sorted(needed):
        row = by_sha.get(sha)
        if row:
            rows_out.append(dict(row))
        else:
            print(f"[WARN] sha {sha} not in old subset.csv; skip row", file=sys.stderr)
    with (bench / "subset.csv").open("w", encoding="utf-8", newline="") as f:
        if not rows_out:
            raise SystemExit("No subset rows left after prune")
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    manifest = {
        "removed_global_indices": drop_global,
        "removed_count": len(drop_global),
        "remaining_count": len(q_new),
        "missing_pdf_shas": sorted(miss),
        "subset_rows": len(rows_out),
    }
    (bench / "prune_missing_pdf_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("\nBacked up previous JSON/CSV to", backup)
    print("Re-run from bench-dir: main.py rebuild-vector-dbs --embedding-provider openai --build-bm25")


if __name__ == "__main__":
    main()
