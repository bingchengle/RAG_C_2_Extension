"""
List questions where predictions disagree with gold (for README examples / manual review).

Examples (from repo root, PowerShell; subcommand is ``single`` or ``compare``):

  # Single run: wrong rows only -> stdout + optional JSON
  python scripts/list_gold_mismatches.py single ^
    --gold data/datasets/benchmark_round1_mini/gold_answers.json ^
    --pred data/datasets/benchmark_round1_mini/answers_eval_baseline_params_debug.json ^
    --out wrong_baseline.json

  # Baseline vs new: fixed / regressed / both_wrong
  python scripts/list_gold_mismatches.py compare ^
    --gold data/datasets/benchmark_round1_mini/gold_answers.json ^
    --baseline data/datasets/benchmark_round1_mini/answers_eval_baseline_params_debug.json ^
    --new data/datasets/benchmark_round1_mini/answers_eval_new_params_debug.json ^
    --compare all --out compare_mismatches.json

  # Scoring uses strict protocol (all gold rows; see benchmark_compare_round1._score).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark import compare_round1 as bc


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _wrong_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in rows if not r["correct"]]


def cmd_single(args: argparse.Namespace) -> None:
    gold = _load(Path(args.gold))
    pred = bc._unwrap_answers(_load(Path(args.pred)))
    s = bc._score(pred, gold)
    bad = _wrong_rows(s["rows"])
    print(
        f"protocol={s['protocol']}  scored={s['total']}/{s['gold_total']}  "
        f"wrong={len(bad)}"
    )
    for i, r in enumerate(bad, 1):
        print(f"\n--- [{i}] {r['kind']} ---")
        print(r["question"][:500] + ("..." if len(r["question"]) > 500 else ""))
        print(f"  pred: {r['pred']!r}")
        print(f"  gold: {r['gold']}")
        if r.get("pred_abstention") is not None:
            print(f"  pred_abstention: {r['pred_abstention']}")
    if args.out:
        out_path = Path(args.out)
        out_path.write_text(
            json.dumps(
                {
                    "score_protocol": s["protocol"],
                    "metrics": {k: v for k, v in s.items() if k != "rows"},
                    "wrong_rows": bad,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nWrote {out_path}")


def cmd_compare(args: argparse.Namespace) -> None:
    gold = _load(Path(args.gold))
    base_items = bc._unwrap_answers(_load(Path(args.baseline)))
    new_items = bc._unwrap_answers(_load(Path(args.new)))
    sb = bc._score(base_items, gold)
    sn = bc._score(new_items, gold)
    diff = bc.diff_score_rows(sb, sn)
    fixed = diff["fixed_by_new"]
    regressed = diff["regressed"]
    both_wrong = diff["both_wrong"]

    print(
        f"protocol={sb['protocol']}  gold_total={sb['gold_total']}  "
        f"baseline acc {sb['accuracy']:.2%} ({sb['correct']}/{sb['total']})  "
        f"new acc {sn['accuracy']:.2%} ({sn['correct']}/{sn['total']})"
    )
    print(f"fixed_by_new (baseline wrong -> new correct): {len(fixed)}")
    print(f"regressed (baseline correct -> new wrong): {len(regressed)}")
    print(f"both_wrong: {len(both_wrong)}")

    if args.compare in ("all", "fixed", "regressed", "both_wrong"):
        show = []
        if args.compare == "all":
            show = [("fixed_by_new", fixed), ("regressed", regressed), ("both_wrong", both_wrong)]
        elif args.compare == "fixed":
            show = [("fixed_by_new", fixed)]
        elif args.compare == "regressed":
            show = [("regressed", regressed)]
        else:
            show = [("both_wrong", both_wrong)]

        for label, rows in show:
            if not rows:
                continue
            print(f"\n======== {label} ({len(rows)}) ========")
            for i, row in enumerate(rows, 1):
                q = row["question"]
                print(f"\n--- [{i}] {row['kind']} ---")
                print(q[:600] + ("..." if len(q) > 600 else ""))
                print(f"  baseline pred: {row['baseline_pred']!r}")
                print(f"  new pred:      {row['new_pred']!r}")
                print(f"  gold:          {row['gold']}")

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "score_protocol": sb["protocol"],
                    "baseline_metrics": {k: v for k, v in sb.items() if k != "rows"},
                    "new_metrics": {k: v for k, v in sn.items() if k != "rows"},
                    **diff,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nWrote {args.out}")


def main() -> None:
    p = argparse.ArgumentParser(description="List pred vs gold mismatches (and optional baseline vs new diff).")
    p.add_argument("--gold", type=Path, required=True)
    sub = p.add_subparsers(dest="mode", required=True)

    one = sub.add_parser("single", help="One prediction file vs gold")
    one.add_argument("--pred", type=Path, required=True)
    one.add_argument("--out", type=Path, default=None, help="Write JSON with wrong_rows + metrics")
    one.set_defaults(func=cmd_single)

    two = sub.add_parser("compare", help="Baseline and new vs gold (fixed / regressed / both_wrong)")
    two.add_argument("--baseline", type=Path, required=True)
    two.add_argument("--new", type=Path, required=True)
    two.add_argument(
        "--compare",
        default="all",
        choices=["all", "fixed", "regressed", "both_wrong"],
        help="Which groups to print in detail (default: all).",
    )
    two.add_argument("--out", type=Path, default=None, help="Write JSON with fixed/regressed/both_wrong lists")
    two.set_defaults(func=cmd_compare)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
