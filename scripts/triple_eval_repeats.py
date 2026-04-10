import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark import compare_round1 as bc
from src.data_paths import BENCHMARK_ROUND1_SOFT_DIR


BENCH_DIR = BENCHMARK_ROUND1_SOFT_DIR
RUNS_DIR = BENCH_DIR / "triple_runs"
UPSTREAM_MAIN = ROOT / "_upstream_source" / "main.py"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _score_file(pred_path: Path, gold_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    payload = _load_json(pred_path)
    pred_items = bc._unwrap_answers(payload)
    return bc._score(pred_items, gold_items)


def _run_source_once(run_idx: int) -> Path:
    before = {p.name for p in BENCH_DIR.glob("answers_max_nst_o3m*.json")}
    cmd = [sys.executable, str(UPSTREAM_MAIN), "process-questions", "--config", "max_nst_o3m"]
    subprocess.run(cmd, cwd=str(BENCH_DIR), check=True)
    candidates = [p for p in BENCH_DIR.glob("answers_max_nst_o3m*.json") if not p.name.endswith("_debug.json")]
    new_candidates = [p for p in candidates if p.name not in before]
    if new_candidates:
        src = max(new_candidates, key=lambda p: p.stat().st_mtime)
    else:
        src = max(candidates, key=lambda p: p.stat().st_mtime)
    out = RUNS_DIR / f"source_run{run_idx}.json"
    shutil.copy2(src, out)
    return out


def _run_improved_once(run_idx: int, provider: str, reranker: str) -> Path:
    cmd = [
        sys.executable,
        str(ROOT / "benchmark_compare_round1.py"),
        "--bench-dir",
        BENCH_DIR.relative_to(ROOT).as_posix(),
        "--improved-only",
        "--parallel-requests",
        "1",
        "--batch-size",
        "2",
        "--embedding-provider",
        provider,
        "--reranker-type",
        reranker,
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    src = BENCH_DIR / "answers_improved_bench_debug.json"
    out = RUNS_DIR / f"improved_{provider}_run{run_idx}.json"
    shutil.copy2(src, out)
    return out


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Triple-run benchmark_compare on soft benchmark")
    args = ap.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    gold = _load_json(BENCH_DIR / "gold_answers.json")

    existing_openai = BENCH_DIR / "answers_improved_openai_debug.json"
    existing_bge = BENCH_DIR / "answers_improved_bge_debug.json"
    if not existing_openai.exists() or not existing_bge.exists():
        raise FileNotFoundError(
            "Missing cached run-1 files for improved variants. "
            "Expected answers_improved_openai_debug.json and answers_improved_bge_debug.json"
        )
    shutil.copy2(existing_openai, RUNS_DIR / "improved_openai_run1.json")
    shutil.copy2(existing_bge, RUNS_DIR / "improved_bge_api_run1.json")

    _run_improved_once(2, "openai", "llm")
    _run_improved_once(3, "openai", "llm")
    _run_improved_once(2, "bge_api", "bge")
    _run_improved_once(3, "bge_api", "bge")

    _run_source_once(1)
    _run_source_once(2)
    _run_source_once(3)

    source_scores = []
    openai_scores = []
    bge_scores = []
    per_run = {"source": [], "improved_openai": [], "improved_bge": []}

    for i in [1, 2, 3]:
        s = _score_file(RUNS_DIR / f"source_run{i}.json", gold)
        o = _score_file(RUNS_DIR / f"improved_openai_run{i}.json", gold)
        b = _score_file(RUNS_DIR / f"improved_bge_api_run{i}.json", gold)

        source_scores.append(s["accuracy"])
        openai_scores.append(o["accuracy"])
        bge_scores.append(b["accuracy"])

        per_run["source"].append({"run": i, "accuracy": s["accuracy"], "correct": s["correct"], "total": s["total"]})
        per_run["improved_openai"].append({"run": i, "accuracy": o["accuracy"], "correct": o["correct"], "total": o["total"]})
        per_run["improved_bge"].append({"run": i, "accuracy": b["accuracy"], "correct": b["correct"], "total": b["total"]})

    summary = {
        "benchmark": str(BENCH_DIR),
        "score_protocol": "strict",
        "runs_dir": str(RUNS_DIR),
        "notes": "Improved run-1 reused from previous completed experiment; source run-1..3 freshly executed.",
        "per_run": per_run,
        "averages": {
            "source": _mean(source_scores),
            "improved_openai": _mean(openai_scores),
            "improved_bge": _mean(bge_scores),
            "delta_openai_minus_source": _mean(openai_scores) - _mean(source_scores),
            "delta_bge_minus_source": _mean(bge_scores) - _mean(source_scores),
            "delta_bge_minus_openai": _mean(bge_scores) - _mean(openai_scores),
        },
    }

    out = BENCH_DIR / "triple_compare_repeats_report.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Triple Compare (3 runs) ===")
    print("score_protocol=strict")
    print(f"Source avg:         {summary['averages']['source']:.2%}")
    print(f"Improved OpenAI avg:{summary['averages']['improved_openai']:.2%}")
    print(f"Improved BGE avg:   {summary['averages']['improved_bge']:.2%}")
    print(f"BGE - OpenAI:       {summary['averages']['delta_bge_minus_openai']:+.2%}")
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
