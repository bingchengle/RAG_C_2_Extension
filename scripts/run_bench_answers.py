"""
Build chunked_reports (if missing), vector+BM25 indexes, then run process_questions for a benchmark folder.

Run from repo root; uses main.py with cwd = bench directory (same as manual workflow).

Example:
  python scripts/run_bench_answers.py --bench-dir data/datasets/benchmark_round1_smoke_sample
  python scripts/run_bench_answers.py --bench-dir data/datasets/erc2_smoke_sample
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "main.py"


def _run(cmd: list[str], *, cwd: Path) -> None:
    print(f"\n$ {' '.join(cmd)}  (cwd={cwd})")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Rebuild indexes + process_questions for a benchmark dir")
    p.add_argument("--bench-dir", type=Path, required=True)
    p.add_argument(
        "--embedding-provider",
        default="openai",
        choices=["openai", "bge_api"],
        help="Must match your env keys (OPENAI vs BGE).",
    )
    p.add_argument("--reranker-type", default="llm", choices=["llm", "bge"])
    p.add_argument("--config", default="max_nst_o3m", help="Pipeline preset for process_questions")
    args = p.parse_args()

    bench = args.bench_dir.resolve()
    if not (bench / "questions.json").is_file():
        raise SystemExit(f"Missing questions.json under {bench}")

    chunked_dir = bench / "databases" / "chunked_reports"
    if not any(chunked_dir.glob("*.json")):
        raise SystemExit(
            f"No JSON in {chunked_dir}. Generate them first, e.g.:\n"
            f"  python scripts/sample_benchmark_subset.py --source-dir ... --out-dir {bench}\n"
            f"  (omit --skip-chunked), or run parse_pdf + process_reports from main.py."
        )

    _run(
        [
            sys.executable,
            str(MAIN),
            "rebuild-vector-dbs",
            "--embedding-provider",
            args.embedding_provider,
            "--build-bm25",
        ],
        cwd=bench,
    )
    _run(
        [
            sys.executable,
            str(MAIN),
            "process-questions",
            "--config",
            args.config,
            "--profile",
            "enhanced",
            "--embedding-provider",
            args.embedding_provider,
            "--reranker-type",
            args.reranker_type,
        ],
        cwd=bench,
    )
    print(f"\nDone. Check {bench} for answers_*.json")


if __name__ == "__main__":
    main()
