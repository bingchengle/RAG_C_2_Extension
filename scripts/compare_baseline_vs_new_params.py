"""
Compare answer accuracy (vs gold) between two full RunConfig tunings (baseline vs new).

All hybrid / top_k / rerank-pool / abstention knobs exposed for both sides on ``run``.
New side: any flag omitted inherits the baseline value for that field (so you can change one knob or many).

Usage (no API):
  python scripts/compare_baseline_vs_new_params.py score \\
    --gold path/to/gold_answers.json \\
    --baseline path/to/answers_baseline.json \\
    --new path/to/answers_new.json \\
    --write-mismatches mismatches.json

Usage (two pipeline runs + score vs gold_answers.json in bench-dir):
  python scripts/compare_baseline_vs_new_params.py run --bench-dir path/to/benchmark_round1_mini \\
    --new-hybrid-llm-weight 0.6 --new-top-n-retrieval 12

  # After ``run``, also write ``<bench-dir>/baseline_vs_new_mismatches.json`` (same as ``--write-mismatches`` to that path):
  python scripts/compare_baseline_vs_new_params.py run --bench-dir path/to/benchmark_round1_mini --export-mismatches

Step-2 retrieval grid (top_n x bm25_weight x reranker; score each vs gold):
  python scripts/compare_baseline_vs_new_params.py grid --dry-run
  python scripts/compare_baseline_vs_new_params.py grid --bench-dir path/to/benchmark_round1_full \\
    --top-n-list 14,20 --bm25-weight-list 0.25 --reranker-types llm,bge

Gold format: benchmark_prepare_round1 style (question, schema, answer list).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark import compare_round1 as bc
from src.pipeline import Pipeline, RunConfig


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_csv_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_csv_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_csv_strs(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _grid_config_suffix(top_n: int, bm25: float, reranker: str, repeat_idx: Optional[int]) -> str:
    """Short unique suffix for answers filename (config_suffix)."""
    b = int(round(bm25 * 1000))
    base = f"_grid_t{top_n}_b{b}_{reranker}"
    if repeat_idx is not None:
        return f"{base}_r{repeat_idx + 1}"
    return base


def _iter_grid_combos(
    top_ns: List[int],
    bm25s: List[float],
    rerankers: List[str],
    repeats: int,
) -> List[Tuple[int, float, str, Optional[int]]]:
    out: List[Tuple[int, float, str, Optional[int]]] = []
    for tn, bm, rt in itertools.product(top_ns, bm25s, rerankers):
        if repeats <= 1:
            out.append((tn, bm, rt, None))
        else:
            for r in range(repeats):
                out.append((tn, bm, rt, r))
    return out


def cmd_grid(args: argparse.Namespace) -> None:
    """Step-2 style sweep: top_n × BM25 merge weight × reranker type; score each vs gold."""
    top_ns = _parse_csv_ints(args.top_n_list)
    bm25s = _parse_csv_floats(args.bm25_weight_list)
    rerankers = _parse_csv_strs(args.reranker_types)
    for rt in rerankers:
        if rt not in ("llm", "bge"):
            raise SystemExit(f"Invalid reranker type {rt!r}; use llm or bge.")
    if not top_ns or not bm25s or not rerankers:
        raise SystemExit("top_n_list, bm25_weight_list, and reranker_types must each be non-empty.")

    combos = _iter_grid_combos(top_ns, bm25s, rerankers, args.repeats)
    if args.dry_run:
        print(
            f"[dry-run] {len(combos)} pipeline run(s) would execute "
            "(top_n x bm25_weight x reranker [x repeats])."
        )
        for tn, bm, rt, rep in combos[:20]:
            suf = _grid_config_suffix(tn, bm, rt, rep if args.repeats > 1 else None)
            print(f"  {suf}  (top_n={tn}, hybrid_merge_bm25_weight={bm}, reranker_type={rt})")
        if len(combos) > 20:
            print(f"  ... and {len(combos) - 20} more.")
        return

    bench_dir = Path(args.bench_dir).resolve()
    gold_path = bench_dir / "gold_answers.json"
    if not gold_path.exists():
        raise FileNotFoundError(
            f"Missing {gold_path}. Use --dry-run to preview combos without data."
        )

    gold_items: List[Dict[str, Any]] = _load_json(gold_path)
    runs_out: List[Dict[str, Any]] = []
    print(f"=== Retrieval param grid ({len(combos)} runs) ===")
    print(f"Benchmark: {bench_dir}")
    print("Score protocol: strict (all gold rows)")

    for i, (tn, bm, rt, rep_idx) in enumerate(combos, start=1):
        repeat_idx = rep_idx if args.repeats > 1 else None
        suffix = _grid_config_suffix(tn, bm, rt, repeat_idx)
        print(f"\n--- [{i}/{len(combos)}] {suffix} ---")
        rc = _benchmark_run_config(
            config_suffix=suffix,
            parallel_requests=args.parallel_requests,
            embedding_provider=args.embedding_provider,
            embedding_model=args.embedding_model or "",
            reranker_type=rt,
            answering_model=args.answering_model,
            top_n_retrieval=tn,
            llm_reranking_sample_size=args.llm_reranking_sample_size,
            hybrid_rerank_llm_weight=float(args.hybrid_llm_weight),
            hybrid_merge_bm25_weight=bm,
            llm_rerank_documents_batch_size=args.llm_rerank_batch_size,
            enable_abstention_gate=bool(args.enable_abstention_gate),
            abstain_min_retrieval_strength=args.abstain_min_retrieval_strength,
            abstain_min_confidence=args.abstain_min_confidence,
            abstain_on_validation_fail=bool(args.abstain_on_validation_fail),
        )
        snap = _tunable_snapshot(rc)
        snap["reranker_type"] = rt
        print(f"tunables={json.dumps(snap, ensure_ascii=False)}")
        out_path = _run_one_pipeline(bench_dir, rc)
        print(f"Saved: {out_path}")
        pred = bc._unwrap_answers(_load_json(out_path))
        s = bc._score(pred, gold_items)
        metrics = {k: v for k, v in s.items() if k != "rows"}
        runs_out.append(
            {
                "config_suffix": suffix,
                "answers_output": str(out_path),
                "tunables": {
                    "top_n_retrieval": tn,
                    "hybrid_merge_bm25_weight": bm,
                    "reranker_type": rt,
                    "repeat_index": repeat_idx,
                    **snap,
                },
                "metrics": metrics,
            }
        )
        print(f"accuracy={s['accuracy']:.2%} ({s['correct']}/{s['total']})")


    runs_sorted = sorted(
        runs_out,
        key=lambda r: (r["metrics"]["accuracy"], r["metrics"].get("correct", 0)),
        reverse=True,
    )
    print("\n=== Grid summary (best first) ===")
    hdr = f"{'rank':<5} {'accuracy':>10} {'corr/tot':>10}  suffix"
    print(hdr)
    print("-" * len(hdr))
    for rank, r in enumerate(runs_sorted, start=1):
        m = r["metrics"]
        print(
            f"{rank:<5} {m['accuracy']:>9.2%} {m['correct']:>4}/{m['total']:<4}  {r['config_suffix']}"
        )

    report_path = Path(args.write_report) if args.write_report else bench_dir / "retrieval_param_grid_report.json"
    report_path = report_path.resolve()
    report_path.write_text(
        json.dumps(
            {
                "benchmark": str(bench_dir),
                "score_protocol": "strict",
                "grid": {
                    "top_n_list": top_ns,
                    "bm25_weight_list": bm25s,
                    "reranker_types": rerankers,
                    "repeats": args.repeats,
                },
                "runs": runs_out,
                "runs_best_first": runs_sorted,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nReport: {report_path}")


def _write_baseline_new_mismatches(
    path: Path,
    *,
    gold_path: Path,
    baseline_path: Path,
    new_path: Path,
    s_base: Dict[str, Any],
    s_new: Dict[str, Any],
    delta: float,
) -> None:
    diff = bc.diff_score_rows(s_base, s_new)
    payload = {
        "gold": str(gold_path),
        "score_protocol": "strict",
        "baseline_file": str(baseline_path),
        "new_file": str(new_path),
        "delta_accuracy": delta,
        "baseline_metrics": {k: v for k, v in s_base.items() if k != "rows"},
        "new_metrics": {k: v for k, v in s_new.items() if k != "rows"},
        **diff,
        "wrong_rows": {
            "baseline": [r for r in s_base["rows"] if not r["correct"]],
            "new": [r for r in s_new["rows"] if not r["correct"]],
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _benchmark_run_config(
    *,
    config_suffix: str,
    parallel_requests: int,
    embedding_provider: str,
    embedding_model: str,
    reranker_type: str,
    answering_model: str,
    top_n_retrieval: int,
    llm_reranking_sample_size: int,
    hybrid_rerank_llm_weight: float,
    hybrid_merge_bm25_weight: float,
    llm_rerank_documents_batch_size: int,
    enable_abstention_gate: bool,
    abstain_min_retrieval_strength: Optional[float],
    abstain_min_confidence: Optional[float],
    abstain_on_validation_fail: bool,
) -> RunConfig:
    """Same backbone as benchmark_compare_round1._run_improved; tunables fully parameterized."""
    return RunConfig(
        use_serialized_tables=False,
        parent_document_retrieval=True,
        llm_reranking=True,
        llm_reranking_sample_size=max(1, int(llm_reranking_sample_size)),
        top_n_retrieval=max(1, int(top_n_retrieval)),
        parallel_requests=max(1, parallel_requests),
        api_provider="openai",
        answering_model=answering_model,
        submission_file=False,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model or "",
        reranker_type=reranker_type,
        domain="finance",
        finance_metric_expansion=True,
        finance_normalize_numeric=True,
        finance_currency_consistency_check=True,
        finance_unit_conversion=True,
        enable_query_rewrite=True,
        enable_similarity_check=True,
        enable_multi_turn=False,
        config_suffix=config_suffix,
        hybrid_rerank_llm_weight=float(hybrid_rerank_llm_weight),
        hybrid_merge_bm25_weight=float(hybrid_merge_bm25_weight),
        llm_rerank_documents_batch_size=max(1, int(llm_rerank_documents_batch_size)),
        enable_abstention_gate=enable_abstention_gate,
        abstain_min_retrieval_strength=abstain_min_retrieval_strength,
        abstain_min_confidence=abstain_min_confidence,
        abstain_on_validation_fail=abstain_on_validation_fail,
    )


def _tunable_snapshot(rc: RunConfig) -> Dict[str, Any]:
    return {
        "hybrid_rerank_llm_weight": rc.hybrid_rerank_llm_weight,
        "hybrid_merge_bm25_weight": rc.hybrid_merge_bm25_weight,
        "llm_rerank_documents_batch_size": rc.llm_rerank_documents_batch_size,
        "top_n_retrieval": rc.top_n_retrieval,
        "llm_reranking_sample_size": rc.llm_reranking_sample_size,
        "enable_abstention_gate": rc.enable_abstention_gate,
        "abstain_min_retrieval_strength": rc.abstain_min_retrieval_strength,
        "abstain_min_confidence": rc.abstain_min_confidence,
        "abstain_on_validation_fail": rc.abstain_on_validation_fail,
    }


def _pick(new_v: Optional[Any], base_v: Any) -> Any:
    return base_v if new_v is None else new_v


def _baseline_from_args(args: argparse.Namespace) -> RunConfig:
    return _benchmark_run_config(
        config_suffix=args.baseline_suffix,
        parallel_requests=args.parallel_requests,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model or "",
        reranker_type=args.reranker_type,
        answering_model=args.answering_model,
        top_n_retrieval=args.baseline_top_n_retrieval,
        llm_reranking_sample_size=args.baseline_llm_reranking_sample_size,
        hybrid_rerank_llm_weight=args.baseline_hybrid_llm_weight,
        hybrid_merge_bm25_weight=args.baseline_hybrid_bm25_weight,
        llm_rerank_documents_batch_size=args.baseline_llm_rerank_batch_size,
        enable_abstention_gate=args.baseline_enable_abstention_gate,
        abstain_min_retrieval_strength=args.baseline_abstain_min_retrieval_strength,
        abstain_min_confidence=args.baseline_abstain_min_confidence,
        abstain_on_validation_fail=args.baseline_abstain_on_validation_fail,
    )


def _new_from_baseline(base: RunConfig, args: argparse.Namespace) -> RunConfig:
    gate = base.enable_abstention_gate
    if args.new_enable_abstention_gate:
        gate = True
    elif args.new_disable_abstention_gate:
        gate = False
    vfail = base.abstain_on_validation_fail
    if args.new_abstain_on_validation_fail:
        vfail = True
    elif args.new_no_abstain_on_validation_fail:
        vfail = False
    return replace(
        base,
        config_suffix=args.new_suffix,
        hybrid_rerank_llm_weight=float(_pick(args.new_hybrid_llm_weight, base.hybrid_rerank_llm_weight)),
        hybrid_merge_bm25_weight=float(_pick(args.new_hybrid_bm25_weight, base.hybrid_merge_bm25_weight)),
        llm_rerank_documents_batch_size=max(
            1, int(_pick(args.new_llm_rerank_batch_size, base.llm_rerank_documents_batch_size))
        ),
        top_n_retrieval=max(1, int(_pick(args.new_top_n_retrieval, base.top_n_retrieval))),
        llm_reranking_sample_size=max(
            1, int(_pick(args.new_llm_reranking_sample_size, base.llm_reranking_sample_size))
        ),
        enable_abstention_gate=gate,
        abstain_min_retrieval_strength=_pick(
            args.new_abstain_min_retrieval_strength, base.abstain_min_retrieval_strength
        ),
        abstain_min_confidence=_pick(args.new_abstain_min_confidence, base.abstain_min_confidence),
        abstain_on_validation_fail=vfail,
    )


def cmd_score(args: argparse.Namespace) -> None:
    gold_path = Path(args.gold)
    base_path = Path(args.baseline)
    new_path = Path(args.new)
    gold_items: List[Dict[str, Any]] = _load_json(gold_path)
    pred_baseline = bc._unwrap_answers(_load_json(base_path))
    pred_new = bc._unwrap_answers(_load_json(new_path))
    s_base = bc._score(pred_baseline, gold_items)
    s_new = bc._score(pred_new, gold_items)
    delta = s_new["accuracy"] - s_base["accuracy"]
    print("=== Accuracy vs gold (protocol=strict, all gold rows) ===")
    print(f"Gold: {gold_path} ({len(gold_items)} items)")
    print(f"Baseline: {base_path}")
    print(f"  accuracy {s_base['accuracy']:.2%} ({s_base['correct']}/{s_base['total']})")
    print(f"New:      {new_path}")
    print(f"  accuracy {s_new['accuracy']:.2%} ({s_new['correct']}/{s_new['total']})")
    print(f"Delta (new - baseline): {delta:+.2%}")
    if args.write_report:
        report = {
            "gold": str(gold_path),
            "score_protocol": "strict",
            "baseline_file": str(base_path),
            "new_file": str(new_path),
            "baseline_metrics": {k: v for k, v in s_base.items() if k != "rows"},
            "new_metrics": {k: v for k, v in s_new.items() if k != "rows"},
            "delta_accuracy": delta,
        }
        Path(args.write_report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Report written: {args.write_report}")
    if getattr(args, "write_mismatches", None):
        mp = Path(args.write_mismatches)
        _write_baseline_new_mismatches(
            mp,
            gold_path=gold_path,
            baseline_path=base_path,
            new_path=new_path,
            s_base=s_base,
            s_new=s_new,
            delta=delta,
        )
        print(f"Mismatches written: {mp}")


def _resolve_answers_output_path(pipeline: Pipeline) -> Path:
    """Resolve path after process_questions (submission_file=False writes *_debug.json)."""
    base = pipeline.paths.answers_file_path
    parent = base.parent
    stem = base.stem
    candidates = [
        p
        for p in parent.glob(f"{stem}*.json")
        if not p.name.endswith("_route_usage.json")
    ]
    if not candidates:
        raise FileNotFoundError(f"No answers JSON matching {stem}*.json under {parent}")
    non_debug = [p for p in candidates if "_debug" not in p.name]
    pool = non_debug if non_debug else candidates
    return max(pool, key=lambda p: p.stat().st_mtime)


def _run_one_pipeline(bench_dir: Path, run_config: RunConfig) -> Path:
    pipeline = Pipeline(bench_dir, run_config=run_config)
    pipeline.process_questions()
    return _resolve_answers_output_path(pipeline)


def cmd_run(args: argparse.Namespace) -> None:
    bench_dir = Path(args.bench_dir).resolve()
    gold_path = bench_dir / "gold_answers.json"
    if not gold_path.exists():
        raise FileNotFoundError(
            f"Missing {gold_path}. Prepare a benchmark folder with gold_answers.json "
            "(e.g. python benchmark_prepare_round1.py) or use the score subcommand with explicit --gold."
        )

    base_cfg = _baseline_from_args(args)
    new_cfg = _new_from_baseline(base_cfg, args)
    snap_b = _tunable_snapshot(base_cfg)
    snap_n = _tunable_snapshot(new_cfg)
    differs = snap_b != snap_n
    if args.warn_if_no_overrides and not differs:
        print(
            "[WARN] Baseline and new tunables are identical. Expect ~0 accuracy delta "
            "(stochastic APIs may still differ slightly).",
            file=sys.stderr,
        )

    print("--- Baseline pipeline run ---")
    print(f"config_suffix={base_cfg.config_suffix!r} tunables={json.dumps(snap_b, ensure_ascii=False)}")
    out_base = _run_one_pipeline(bench_dir, base_cfg)
    print(f"Saved: {out_base}")

    print("--- New-params pipeline run ---")
    print(f"config_suffix={new_cfg.config_suffix!r} tunables={json.dumps(snap_n, ensure_ascii=False)}")
    out_new = _run_one_pipeline(bench_dir, new_cfg)
    print(f"Saved: {out_new}")

    gold_items = _load_json(gold_path)
    s_base = bc._score(bc._unwrap_answers(_load_json(out_base)), gold_items)
    s_new = bc._score(bc._unwrap_answers(_load_json(out_new)), gold_items)
    delta = s_new["accuracy"] - s_base["accuracy"]
    report_path = bench_dir / "baseline_vs_new_params_report.json"
    report_path.write_text(
        json.dumps(
            {
                "benchmark": str(bench_dir),
                "score_protocol": "strict",
                "baseline_output": str(out_base),
                "new_output": str(out_new),
                "baseline_tunables": snap_b,
                "new_tunables": snap_n,
                "baseline_metrics": {k: v for k, v in s_base.items() if k != "rows"},
                "new_metrics": {k: v for k, v in s_new.items() if k != "rows"},
                "delta_accuracy": delta,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("=== Scored vs gold (strict) ===")
    print(f"Baseline accuracy: {s_base['accuracy']:.2%} ({s_base['correct']}/{s_base['total']})")
    print(f"New accuracy:      {s_new['accuracy']:.2%} ({s_new['correct']}/{s_new['total']})")
    print(f"Delta (new - baseline): {delta:+.2%}")
    print(f"Report: {report_path}")
    mp = bc.mismatch_out_path_for_run(
        bench_dir,
        getattr(args, "write_mismatches", None),
        bool(getattr(args, "export_mismatches", False)),
    )
    if mp:
        _write_baseline_new_mismatches(
            mp,
            gold_path=gold_path,
            baseline_path=out_base,
            new_path=out_new,
            s_base=s_base,
            s_new=s_new,
            delta=delta,
        )
        print(f"Mismatches written: {mp}")


def _add_tunable_args(p: argparse.ArgumentParser, *, prefix: str) -> None:
    """prefix is 'baseline' or 'new' (new_* args use None default to mean inherit when prefix=='new')."""
    is_base = prefix == "baseline"

    def add(name: str, **kw):
        p.add_argument(f"--{prefix}-{name}", **kw)

    add(
        "hybrid-llm-weight",
        type=float,
        default=0.7 if is_base else None,
        help="Hybrid rerank: LLM/BGE vs vector weight [0,1]."
        + ("" if is_base else " Omit = inherit baseline."),
    )
    add(
        "hybrid-bm25-weight",
        type=float,
        default=0.25 if is_base else None,
        help="Hybrid merge: BM25 vs vector weight [0,1]."
        + ("" if is_base else " Omit = inherit baseline."),
    )
    add(
        "llm-rerank-batch-size",
        type=int,
        default=2 if is_base else None,
        help="LLM reranker documents per batch." + ("" if is_base else " Omit = inherit baseline."),
    )
    add(
        "top-n-retrieval",
        type=int,
        default=14 if is_base else None,
        help="Final top-k after rerank." + ("" if is_base else " Omit = inherit baseline."),
    )
    add(
        "llm-reranking-sample-size",
        type=int,
        default=36 if is_base else None,
        help="Hybrid rerank candidate pool size." + ("" if is_base else " Omit = inherit baseline."),
    )
    if is_base:
        add(
            "enable-abstention-gate",
            action="store_true",
            help="Turn on multi-signal abstention gate (default off for baseline).",
        )
        add(
            "abstain-min-retrieval-strength",
            type=float,
            default=None,
            help="Abstain if best retrieval signal < this (0-1).",
        )
        add(
            "abstain-min-confidence",
            type=float,
            default=None,
            help="Abstain if confidence < this.",
        )
        add(
            "abstain-on-validation-fail",
            action="store_true",
            help="Abstain when validation fails.",
        )
    else:
        g = p.add_mutually_exclusive_group()
        g.add_argument(
            "--new-enable-abstention-gate",
            action="store_true",
            help="Force abstention gate on (overrides baseline).",
        )
        g.add_argument(
            "--new-disable-abstention-gate",
            action="store_true",
            help="Force abstention gate off (overrides baseline).",
        )
        add(
            "abstain-min-retrieval-strength",
            type=float,
            default=None,
            help="Omit = inherit baseline.",
        )
        add(
            "abstain-min-confidence",
            type=float,
            default=None,
            help="Omit = inherit baseline.",
        )
        g2 = p.add_mutually_exclusive_group()
        g2.add_argument(
            "--new-abstain-on-validation-fail",
            action="store_true",
            help="Force abstain-on-validation-fail on.",
        )
        g2.add_argument(
            "--new-no-abstain-on-validation-fail",
            action="store_true",
            help="Force abstain-on-validation-fail off.",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline vs new: full hybrid / top_k / rerank / abstention tunables vs same gold"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_score = sub.add_parser("score", help="Score two existing prediction files against the same gold")
    p_score.add_argument("--gold", type=Path, required=True)
    p_score.add_argument("--baseline", type=Path, required=True)
    p_score.add_argument("--new", type=Path, required=True)
    p_score.add_argument("--write-report", type=Path, default=None)
    p_score.add_argument(
        "--write-mismatches",
        type=Path,
        default=None,
        help="Write fixed_by_new / regressed / both_wrong + per-run wrong_rows (JSON).",
    )
    p_score.set_defaults(func=cmd_score)

    p_run = sub.add_parser("run", help="Run pipeline twice (baseline + new) and score vs gold_answers.json")
    p_run.add_argument("--bench-dir", type=Path, required=True)
    p_run.add_argument("--parallel-requests", type=int, default=6)
    p_run.add_argument("--embedding-provider", type=str, default="openai")
    p_run.add_argument("--embedding-model", type=str, default="")
    p_run.add_argument("--reranker-type", type=str, default="llm", choices=["llm", "bge"])
    p_run.add_argument(
        "--answering-model",
        type=str,
        default="o3-mini-2025-01-31",
        help="Answering model (match benchmark_compare_round1._run_improved unless you change it).",
    )
    p_run.add_argument("--baseline-suffix", type=str, default="_eval_baseline_params")
    p_run.add_argument("--new-suffix", type=str, default="_eval_new_params")
    _add_tunable_args(p_run, prefix="baseline")
    _add_tunable_args(p_run, prefix="new")
    p_run.add_argument(
        "--no-warn-if-no-overrides",
        action="store_false",
        dest="warn_if_no_overrides",
        default=True,
        help="Suppress warning when baseline and new tunables are identical.",
    )
    p_run.add_argument(
        "--write-mismatches",
        type=Path,
        default=None,
        help="Write mismatch JSON to this path (overrides --export-mismatches).",
    )
    p_run.add_argument(
        "--export-mismatches",
        action="store_true",
        help="Write mismatch JSON to <bench-dir>/baseline_vs_new_mismatches.json (ignored if --write-mismatches is set).",
    )
    p_run.set_defaults(func=cmd_run)

    p_grid = sub.add_parser(
        "grid",
        help="Step-2 sweep: Cartesian grid of top_n_retrieval × hybrid_merge_bm25_weight × reranker_type; run + score each vs gold",
    )
    p_grid.add_argument(
        "--bench-dir",
        type=Path,
        default=None,
        help="Benchmark folder with gold_answers.json (omit with --dry-run to only list combinations).",
    )
    p_grid.add_argument(
        "--top-n-list",
        type=str,
        default="14,17,20",
        help="Comma-separated top_n_retrieval values (e.g. 14,20).",
    )
    p_grid.add_argument(
        "--bm25-weight-list",
        type=str,
        default="0.22,0.28",
        help="Comma-separated hybrid_merge_bm25_weight values (e.g. 0.2,0.25,0.3).",
    )
    p_grid.add_argument(
        "--reranker-types",
        type=str,
        default="llm,bge",
        help="Comma-separated reranker types: llm and/or bge.",
    )
    p_grid.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeat each combo N times (suffix _r1, _r2, ...) for variance checks; default 1.",
    )
    p_grid.add_argument(
        "--dry-run",
        action="store_true",
        help="Print combination count and sample list; do not read gold or call APIs.",
    )
    p_grid.add_argument("--parallel-requests", type=int, default=6)
    p_grid.add_argument("--embedding-provider", type=str, default="openai")
    p_grid.add_argument("--embedding-model", type=str, default="")
    p_grid.add_argument("--answering-model", type=str, default="o3-mini-2025-01-31")
    p_grid.add_argument("--hybrid-llm-weight", type=float, default=0.7)
    p_grid.add_argument("--llm-reranking-sample-size", type=int, default=36)
    p_grid.add_argument("--llm-rerank-batch-size", type=int, default=2)
    p_grid.add_argument(
        "--write-report",
        type=Path,
        default=None,
        help="JSON report path (default: <bench-dir>/retrieval_param_grid_report.json).",
    )
    p_grid.add_argument(
        "--enable-abstention-gate",
        action="store_true",
        help="Enable multi-signal abstention gate for all grid runs (default off).",
    )
    p_grid.add_argument("--abstain-min-retrieval-strength", type=float, default=None)
    p_grid.add_argument("--abstain-min-confidence", type=float, default=None)
    p_grid.add_argument(
        "--abstain-on-validation-fail",
        action="store_true",
        help="Abstain when validation fails (all grid runs).",
    )
    p_grid.set_defaults(func=cmd_grid)

    args = parser.parse_args()
    if getattr(args, "command", None) == "grid":
        if not getattr(args, "dry_run", False) and getattr(args, "bench_dir", None) is None:
            parser.error("grid: --bench-dir is required unless --dry-run is set")
    args.func(args)


if __name__ == "__main__":
    main()
