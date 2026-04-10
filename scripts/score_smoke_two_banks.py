"""
Smoke-score a random subset of each bank using existing prediction JSONs (no API).

1) Round1: gold from sampled dir vs benchmark_round1_full/answers_max_nst_o3m.json (filtered).
2) ERC2: gold from sampled dir vs erc2_set/answers_1st_place_o3-mini.json (filtered).

Step 1 writes sample dirs via sample_benchmark_subset.py; step 2 scores.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark import compare_round1 as bc
from src.data_paths import (
    BENCHMARK_ROUND1_FULL_DIR,
    DATASETS_DIR,
    ENTERPRISE_RAG_ROUND2_DIR,
    ERC2_SET_DIR,
)

DEFAULT_ROUND1_PRED = BENCHMARK_ROUND1_FULL_DIR / "answers_max_nst_o3m.json"
DEFAULT_ERC2_PRED = ERC2_SET_DIR / "answers_1st_place_o3-mini.json"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _filter_preds(pred_items: list, question_set: set[str]) -> list:
    out = []
    for p in pred_items:
        t = bc._question_text(p)
        if t in question_set:
            out.append(p)
    return out


def _score_bank(*, name: str, gold_path: Path, pred_path: Path) -> None:
    gold_items = _load(gold_path)
    qs = {g["question"] for g in gold_items}
    payload = _load(pred_path)
    pred_items = bc._unwrap_answers(payload)
    filtered = _filter_preds(pred_items, qs)
    print(f"\n=== {name} ===")
    print(f"Gold file: {gold_path} ({len(gold_items)} questions)")
    print(f"Pred file: {pred_path}")
    print(f"Pred rows matched to sample: {len(filtered)} / {len(gold_items)}")
    if len(filtered) < len(gold_items):
        print("[WARN] Some sampled questions missing from pred file; those score as empty.", file=sys.stderr)
    s = bc._score(filtered, gold_items)
    print(f"  strict: {s['accuracy']:.2%} ({s['correct']}/{s['total']})")


def main() -> None:
    p = argparse.ArgumentParser(description="Sample + score two banks using on-disk predictions")
    p.add_argument("--n", type=int, default=4, help="Questions per bank")
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--round1-out", type=Path, default=DATASETS_DIR / "benchmark_round1_smoke_sample")
    p.add_argument("--erc2-out", type=Path, default=DATASETS_DIR / "erc2_smoke_sample")
    p.add_argument("--round1-pred", type=Path, default=DEFAULT_ROUND1_PRED)
    p.add_argument("--erc2-pred", type=Path, default=DEFAULT_ERC2_PRED)
    p.add_argument("--skip-sample", action="store_true", help="Assume sample dirs already exist")
    args = p.parse_args()

    samp = ROOT / "scripts" / "sample_benchmark_subset.py"
    if not args.skip_sample:
        for src, out in [
            (BENCHMARK_ROUND1_FULL_DIR, args.round1_out),
            (ENTERPRISE_RAG_ROUND2_DIR, args.erc2_out),
        ]:
            subprocess.run(
                [
                    sys.executable,
                    str(samp),
                    "--source-dir",
                    str(src),
                    "--n",
                    str(args.n),
                    "--seed",
                    str(args.seed),
                    "--out-dir",
                    str(out),
                    "--skip-chunked",
                ],
                check=True,
            )

    _score_bank(
        name="Round1 (sample vs answers_max_nst_o3m)",
        gold_path=args.round1_out / "gold_answers.json",
        pred_path=args.round1_pred,
    )
    _score_bank(
        name="Enterprise RAG Round2 (sample vs o3-mini ref run)",
        gold_path=args.erc2_out / "gold_answers.json",
        pred_path=args.erc2_pred,
    )


if __name__ == "__main__":
    main()
