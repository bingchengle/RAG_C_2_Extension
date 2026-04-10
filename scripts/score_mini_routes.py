"""Score two answer JSON files on benchmark_round1_mini against gold (benchmark_compare_round1 logic)."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmark import compare_round1 as bc

BENCH = ROOT / "data" / "datasets" / "benchmark_round1_mini"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bench-dir", type=Path, default=BENCH)
    p.add_argument("--economy-file", default="answers_base_04.json")
    p.add_argument("--quality-file", default="answers_base_05.json")
    args = p.parse_args()
    bench = args.bench_dir
    gold = json.loads((bench / "gold_answers.json").read_text(encoding="utf-8"))
    pairs = [
        ("economy", bench / args.economy_file),
        ("quality", bench / args.quality_file),
    ]
    print(f"score_protocol=strict  gold_total={len(gold)}")
    for label, path in pairs:
        pred = bc._unwrap_answers(json.loads(path.read_text(encoding="utf-8")))
        s = bc._score(pred, gold)
        pct = 100.0 * s["accuracy"]
        print(f"{label}: {pct:.2f}% ({s['correct']}/{s['total']})")
        if s["abstention_applied_count"]:
            ac = s["abstention_applied_correct"]
            aa = s["abstention_applied_accuracy"]
            pct_str = f"{aa:.2%}" if aa is not None else "n/a"
            print(f"  abstention: {ac}/{s['abstention_applied_count']} correct ({pct_str})")


if __name__ == "__main__":
    main()
