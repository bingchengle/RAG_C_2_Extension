import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.data_paths import BENCHMARK_ROUND1_MINI_DIR
from src.pipeline import Pipeline, RunConfig
from src.questions_processing import QuestionsProcessor

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_MAIN = ROOT / "_upstream_source" / "main.py"
_DEFAULT_BENCH = BENCHMARK_ROUND1_MINI_DIR.relative_to(ROOT).as_posix()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _unwrap_answers(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        if "questions" in payload and isinstance(payload["questions"], list):
            return payload["questions"]
        if "answers" in payload and isinstance(payload["answers"], list):
            return payload["answers"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported answers format: {type(payload)}")


def _question_text(item: Dict[str, Any]) -> str:
    return (
        item.get("question_text")
        or item.get("text")
        or item.get("question")
        or ""
    ).strip()


def _pred_value(item: Dict[str, Any]) -> Any:
    if item is None:
        return None
    if "value" in item:
        value = item.get("value")
        if value is None and "error" in item and "No report found with" in str(item["error"]):

            return "N/A"
        return value
    if "answer" in item:
        return item.get("answer")
    return None


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _to_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None

    m = re.search(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?", s)
    if not m:
        return None
    candidate = m.group(0).replace(",", "")
    try:
        return float(candidate)
    except ValueError:
        return None


def _is_na(value: Any) -> bool:
    normalized = _normalize_text(value)
    return normalized in {"n/a", "na", "信息不足", "insufficient information", "not available"}


def _pred_abstention_applied(item: Optional[Dict[str, Any]]) -> Optional[bool]:
    if not item:
        return None
    a = item.get("abstention")
    if not isinstance(a, dict):
        return None
    applied = a.get("applied")
    if applied is None:
        return None
    return bool(applied)


def _match_number(pred: Any, gold_candidates: List[Any], rel_tol: float = 0.01) -> bool:
    if _is_na(pred):
        return any(_is_na(c) for c in gold_candidates)
    p = _to_float(pred)
    if p is None:
        return False
    for c in gold_candidates:
        g = _to_float(c)
        if g is None:
            continue
        denom = max(abs(g), 1.0)
        if abs(p - g) / denom <= rel_tol:
            return True
    return False


def _match_name(pred: Any, gold_candidates: List[Any]) -> bool:
    if _is_na(pred):
        return any(_is_na(c) for c in gold_candidates)
    p = _normalize_text(pred)
    return any(p == _normalize_text(c) for c in gold_candidates if not _is_na(c))


def _score(
    pred_items: List[Dict[str, Any]],
    gold_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Strict scoring: every gold row counts (including gold-only-N/A rows: prediction N/A matches).
    """
    pred_by_q = {_question_text(item): item for item in pred_items if _question_text(item)}
    rows = []
    by_kind: Dict[str, Dict[str, int]] = {}
    total = 0
    correct = 0
    gold_total = len(gold_items)

    for g in gold_items:
        q = g["question"]
        kind = g["schema"]
        gold_candidates = g.get("answer", [])

        pred_item = pred_by_q.get(q)
        pred_value = _pred_value(pred_item) if pred_item else None

        if kind == "number":
            ok = _match_number(pred_value, gold_candidates)
        else:
            ok = _match_name(pred_value, gold_candidates)

        total += 1
        correct += int(ok)
        by_kind.setdefault(kind, {"total": 0, "correct": 0})
        by_kind[kind]["total"] += 1
        by_kind[kind]["correct"] += int(ok)
        rows.append(
            {
                "question": q,
                "kind": kind,
                "pred": pred_value,
                "gold": gold_candidates,
                "correct": ok,
                "pred_abstention": _pred_abstention_applied(pred_item),
            }
        )

    abstention_rows = [r for r in rows if r.get("pred_abstention") is True]
    abstention_applied_count = len(abstention_rows)
    abstention_applied_correct = sum(1 for r in abstention_rows if r["correct"])

    return {
        "protocol": "strict",
        "gold_total": gold_total,
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "abstention_applied_count": abstention_applied_count,
        "abstention_applied_correct": abstention_applied_correct,
        "abstention_applied_accuracy": (abstention_applied_correct / abstention_applied_count)
        if abstention_applied_count
        else None,
        "by_kind": {
            k: {
                **v,
                "accuracy": (v["correct"] / v["total"]) if v["total"] else 0.0,
            }
            for k, v in by_kind.items()
        },
        "rows": rows,
    }


def diff_score_rows(s_base: Dict[str, Any], s_new: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given two _score() results on the same gold, classify questions by baseline/new correctness.
    Keys: fixed_by_new, regressed, both_wrong (each a list of detail dicts with question, kind, preds, gold).
    """
    by_q_b = {r["question"]: r for r in s_base["rows"]}
    by_q_n = {r["question"]: r for r in s_new["rows"]}
    common = sorted(set(by_q_b.keys()) & set(by_q_n.keys()))
    fixed_by_new: List[Dict[str, Any]] = []
    regressed: List[Dict[str, Any]] = []
    both_wrong: List[Dict[str, Any]] = []
    for q in common:
        rb, rn = by_q_b[q], by_q_n[q]
        cb, cn = rb["correct"], rn["correct"]
        if not cb and cn:
            fixed_by_new.append(
                {
                    "question": q,
                    "kind": rb["kind"],
                    "baseline_pred": rb["pred"],
                    "new_pred": rn["pred"],
                    "gold": rb["gold"],
                }
            )
        elif cb and not cn:
            regressed.append(
                {
                    "question": q,
                    "kind": rb["kind"],
                    "baseline_pred": rb["pred"],
                    "new_pred": rn["pred"],
                    "gold": rb["gold"],
                }
            )
        elif not cb and not cn:
            both_wrong.append(
                {
                    "question": q,
                    "kind": rb["kind"],
                    "baseline_pred": rb["pred"],
                    "new_pred": rn["pred"],
                    "gold": rb["gold"],
                }
            )
    return {
        "fixed_by_new": fixed_by_new,
        "regressed": regressed,
        "both_wrong": both_wrong,
        "counts": {
            "fixed_by_new": len(fixed_by_new),
            "regressed": len(regressed),
            "both_wrong": len(both_wrong),
        },
    }


def mismatch_out_path_for_run(
    bench_dir: Path,
    write_mismatches: Optional[Path],
    export_mismatches: bool,
) -> Optional[Path]:
    """
    Resolve mismatch JSON path for compare_baseline_vs_new_params ``run``.
    Explicit ``write_mismatches`` overrides ``export_mismatches``.
    """
    if write_mismatches is not None:
        return Path(write_mismatches)
    if export_mismatches:
        return bench_dir / "baseline_vs_new_mismatches.json"
    return None


def _run_improved(
    bench_dir: Path,
    parallel_requests: int = 6,
    embedding_provider: str = "openai",
    embedding_model: Optional[str] = None,
    reranker_type: str = "llm",
) -> Path:
    run_config = RunConfig(
        use_serialized_tables=False,
        parent_document_retrieval=True,
        llm_reranking=True,
        llm_reranking_sample_size=36,
        top_n_retrieval=14,
        hybrid_merge_bm25_weight=0.25,
        parallel_requests=max(1, parallel_requests),
        api_provider="openai",
        answering_model="o3-mini-2025-01-31",
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
        config_suffix="_improved_bench",
    )
    Pipeline(bench_dir, run_config=run_config).process_questions()
    debug_path = bench_dir / "answers_improved_bench_debug.json"
    normal_path = bench_dir / "answers_improved_bench.json"
    return debug_path if debug_path.exists() else normal_path


def _run_improved_batched(
    bench_dir: Path,
    parallel_requests: int = 1,
    batch_size: int = 2,
    resume: bool = True,
    embedding_provider: str = "openai",
    embedding_model: Optional[str] = None,
    reranker_type: str = "llm",
) -> Path:
    questions_path = bench_dir / "questions.json"
    all_questions = _load_json(questions_path)
    if not isinstance(all_questions, list):
        raise ValueError(f"Unexpected questions format in {questions_path}")

    state_path = bench_dir / "answers_improved_bench_batched_state.json"
    output_path = bench_dir / "answers_improved_bench_debug.json"

    processed_by_text: Dict[str, Dict[str, Any]] = {}
    if resume and state_path.exists():
        state = _load_json(state_path)
        for item in state.get("questions", []):
            qtext = _question_text(item)
            if qtext:
                processed_by_text[qtext] = item

    vector_db_dir = bench_dir / "databases" / "vector_dbs"
    bm25_db_dir = bench_dir / "databases" / "bm25_dbs"
    documents_dir = bench_dir / "databases" / "chunked_reports"
    subset_path = bench_dir / "subset.csv"
    processor = QuestionsProcessor(
        vector_db_dir=vector_db_dir,
        bm25_db_dir=bm25_db_dir,
        documents_dir=documents_dir,
        questions_file_path=None,
        new_challenge_pipeline=True,
        subset_path=subset_path,
        parent_document_retrieval=True,
        llm_reranking=True,
        llm_reranking_sample_size=36,
        top_n_retrieval=14,
        hybrid_merge_bm25_weight=0.25,
        parallel_requests=max(1, parallel_requests),
        api_provider="openai",
        answering_model="o3-mini-2025-01-31",
        full_context=False,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        reranker_type=reranker_type,
        domain="finance",
        finance_metric_expansion=True,
        finance_normalize_numeric=True,
        finance_currency_consistency_check=True,
        finance_unit_conversion=True,
        enable_query_rewrite=True,
        enable_similarity_check=True,
        enable_multi_turn=False,
        conversation_max_turns=6,
    )

    remaining = [q for q in all_questions if _question_text(q) not in processed_by_text]
    if not remaining:

        payload = {"questions": [processed_by_text[_question_text(q)] for q in all_questions]}
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return output_path

    batch_size = max(1, batch_size)
    for i in range(0, len(remaining), batch_size):
        batch = remaining[i : i + batch_size]
        result = processor.process_questions_list(batch, output_path=None, submission_file=False)
        for item in result.get("questions", []):
            qtext = _question_text(item)
            if qtext:
                processed_by_text[qtext] = item


        checkpoint_questions = [processed_by_text[_question_text(q)] for q in all_questions if _question_text(q) in processed_by_text]
        checkpoint = {
            "questions": checkpoint_questions,
            "processed_count": len(checkpoint_questions),
            "total_count": len(all_questions),
        }
        state_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")

    final_questions = [processed_by_text.get(_question_text(q), q) for q in all_questions]
    output_payload = {"questions": final_questions}
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def _run_source(bench_dir: Path) -> Path:
    cmd = [
        sys.executable,
        str(UPSTREAM_MAIN),
        "process-questions",
        "--config",
        "max_nst_o3m",
    ]
    subprocess.run(cmd, cwd=str(bench_dir), check=True)
    debug_path = bench_dir / "answers_max_nst_o3m_debug.json"
    normal_path = bench_dir / "answers_max_nst_o3m.json"
    return normal_path if normal_path.exists() else debug_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare improved project with upstream on benchmark set")
    parser.add_argument("--bench-dir", type=str, default=_DEFAULT_BENCH, help="Benchmark directory")
    parser.add_argument("--score-only", action="store_true", help="Only score existing outputs")
    parser.add_argument("--improved-only", action="store_true", help="Run only improved project")
    parser.add_argument("--parallel-requests", type=int, default=6, help="Parallel requests for improved run")
    parser.add_argument("--batch-size", type=int, default=0, help="If >0, run improved model in small batches")
    parser.add_argument("--resume", action="store_true", help="Resume improved batched run from checkpoint")
    parser.add_argument("--embedding-provider", type=str, default="openai", help="Embedding provider for improved run")
    parser.add_argument("--embedding-model", type=str, default="", help="Optional embedding model override")
    parser.add_argument("--reranker-type", type=str, default="llm", help="Reranker type: llm or bge")
    args = parser.parse_args()
    bench_dir = ROOT / args.bench_dir

    if not bench_dir.exists():
        raise FileNotFoundError(f"Missing benchmark dir: {bench_dir}")

    gold_items = _load_json(bench_dir / "gold_answers.json")

    if args.score_only:
        improved_path = bench_dir / "answers_improved_bench_debug.json"
        source_path = bench_dir / "answers_max_nst_o3m.json"
        if not source_path.exists():
            alt_source = bench_dir / "answers_max_nst_o3m_debug.json"
            source_path = alt_source if alt_source.exists() else improved_path
    else:
        if args.batch_size > 0:
            improved_path = _run_improved_batched(
                bench_dir=bench_dir,
                parallel_requests=args.parallel_requests,
                batch_size=args.batch_size,
                resume=args.resume,
                embedding_provider=args.embedding_provider,
                embedding_model=args.embedding_model or None,
                reranker_type=args.reranker_type,
            )
        else:
            improved_path = _run_improved(
                bench_dir,
                parallel_requests=args.parallel_requests,
                embedding_provider=args.embedding_provider,
                embedding_model=args.embedding_model or None,
                reranker_type=args.reranker_type,
            )
        if args.improved_only:
            source_path = bench_dir / "answers_max_nst_o3m.json"
            if not source_path.exists():
                source_path = bench_dir / "answers_max_nst_o3m_debug.json"
                if not source_path.exists():
                    source_path = improved_path
        else:
            source_path = _run_source(bench_dir)

    improved = _score(_unwrap_answers(_load_json(improved_path)), gold_items)
    source = _score(_unwrap_answers(_load_json(source_path)), gold_items)

    report = {
        "benchmark": str(bench_dir),
        "gold_count": len(gold_items),
        "improved": {
            "file": str(improved_path),
            "metrics": {k: v for k, v in improved.items() if k != "rows"},
        },
        "source": {
            "file": str(source_path),
            "metrics": {k: v for k, v in source.items() if k != "rows"},
        },
        "delta_accuracy": improved["accuracy"] - source["accuracy"],
        "improved_rows": improved["rows"],
        "source_rows": source["rows"],
    }

    out = bench_dir / "comparison_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Benchmark Comparison ===")
    print(f"Score protocol: strict (all gold rows)")
    print(f"Gold questions: {len(gold_items)}")
    print(f"Source accuracy:   {source['accuracy']:.2%} ({source['correct']}/{source['total']})")
    print(f"Improved accuracy: {improved['accuracy']:.2%} ({improved['correct']}/{improved['total']})")
    print(f"Delta: {report['delta_accuracy']:+.2%}")
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
