"""
Run a single grid-style pipeline (same RunConfig as compare_baseline_vs_new_params grid) and
optionally print strict accuracy vs an existing baseline answers file.

Example:
  python scripts/run_one_grid_combo_on_bench.py \\
    --bench-dir data/datasets/benchmark_merged_140_rand10 \\
    --top-n 14 --bm25 0.27 \\
    --compare-baseline data/datasets/benchmark_merged_140_rand10/answers_max_nst_o3m.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmark import compare_round1 as bc


def _load_compare_module():
    spec = importlib.util.spec_from_file_location(
        "compare_baseline_vs_new_params",
        ROOT / "scripts" / "compare_baseline_vs_new_params.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser(description="One grid combo run + optional score vs baseline")
    ap.add_argument("--bench-dir", type=Path, required=True)
    ap.add_argument("--top-n", type=int, required=True)
    ap.add_argument("--bm25", type=float, required=True, help="hybrid_merge_bm25_weight")
    ap.add_argument("--parallel-requests", type=int, default=6)
    ap.add_argument("--embedding-provider", type=str, default="openai")
    ap.add_argument("--compare-baseline", type=Path, default=None, help="Existing answers JSON to diff")
    args = ap.parse_args()

    bench: Path = args.bench_dir.resolve()
    mod = _load_compare_module()
    b = int(round(float(args.bm25) * 1000))
    suffix = f"_grid_t{int(args.top_n)}_b{b}_llm"
    rc = mod._benchmark_run_config(
        config_suffix=suffix,
        parallel_requests=max(1, args.parallel_requests),
        embedding_provider=args.embedding_provider,
        embedding_model="",
        reranker_type="llm",
        answering_model="o3-mini-2025-01-31",
        top_n_retrieval=max(1, int(args.top_n)),
        llm_reranking_sample_size=36,
        hybrid_rerank_llm_weight=0.7,
        hybrid_merge_bm25_weight=float(args.bm25),
        llm_rerank_documents_batch_size=2,
        enable_abstention_gate=False,
        abstain_min_retrieval_strength=None,
        abstain_min_confidence=None,
        abstain_on_validation_fail=False,
    )
    out = mod._run_one_pipeline(bench, rc)
    gold = json.loads((bench / "gold_answers.json").read_text(encoding="utf-8"))
    pred_new = bc._unwrap_answers(json.loads(out.read_text(encoding="utf-8")))
    s_new = bc._score(pred_new, gold)
    print(f"=== {suffix} ===")
    print(f"answers: {out}")
    print(f"strict: {s_new['accuracy']:.2%} ({s_new['correct']}/{s_new['total']})")
    if args.compare_baseline:
        bp = Path(args.compare_baseline).resolve()
        pred_b = bc._unwrap_answers(json.loads(bp.read_text(encoding="utf-8")))
        s_b = bc._score(pred_b, gold)
        print(f"--- vs baseline {bp.name} ---")
        print(f"baseline strict: {s_b['accuracy']:.2%} ({s_b['correct']}/{s_b['total']})")
        print(f"delta (new - baseline): {s_new['accuracy'] - s_b['accuracy']:+.2%}")


if __name__ == "__main__":
    main()
