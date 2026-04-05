import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import benchmark_compare_round1 as bc


BENCH_DIR = ROOT / "data" / "benchmark_round1_soft"
RUNS_DIR = BENCH_DIR / "triple_runs"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _score_file(path: Path, gold: List[Dict[str, Any]]) -> Dict[str, Any]:
    payload = _load_json(path)
    return bc._score(bc._unwrap_answers(payload), gold)


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    gold = _load_json(BENCH_DIR / "gold_answers.json")

    source_outputs = [
        BENCH_DIR / "answers_max_nst_o3m.json",
        BENCH_DIR / "answers_max_nst_o3m_01.json",
        BENCH_DIR / "answers_max_nst_o3m_02.json",
    ]
    for idx, src in enumerate(source_outputs, start=1):
        if not src.exists():
            raise FileNotFoundError(f"Missing source output: {src}")
        shutil.copy2(src, RUNS_DIR / f"source_run{idx}.json")

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
        "runs_dir": str(RUNS_DIR),
        "notes": "Recomputed from existing outputs: source uses answers_max_nst_o3m, _01, _02.",
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

    out = BENCH_DIR / "triple_compare_repeats_report_fixed.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Triple Compare (fixed) ===")
    print(f"Source avg:         {summary['averages']['source']:.2%}")
    print(f"Improved OpenAI avg:{summary['averages']['improved_openai']:.2%}")
    print(f"Improved BGE avg:   {summary['averages']['improved_bge']:.2%}")
    print(f"BGE - OpenAI:       {summary['averages']['delta_bge_minus_openai']:+.2%}")
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
