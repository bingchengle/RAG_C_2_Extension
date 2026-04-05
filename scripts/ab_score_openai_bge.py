import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import benchmark_compare_round1 as bc


def main() -> None:
    base = ROOT / "data" / "benchmark_round1_soft"
    gold = json.loads((base / "gold_answers.json").read_text(encoding="utf-8"))

    openai_payload = json.loads((base / "answers_improved_openai_debug.json").read_text(encoding="utf-8"))
    bge_payload = json.loads((base / "answers_improved_bge_debug.json").read_text(encoding="utf-8"))

    openai = bc._score(bc._unwrap_answers(openai_payload), gold)
    bge = bc._score(bc._unwrap_answers(bge_payload), gold)
    delta = bge["accuracy"] - openai["accuracy"]

    report = {
        "benchmark": str(base.resolve()),
        "openai": {k: v for k, v in openai.items() if k != "rows"},
        "bge": {k: v for k, v in bge.items() if k != "rows"},
        "delta_bge_minus_openai": delta,
    }

    out = base / "ab_report_openai_vs_bge.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("OPENAI", openai["correct"], openai["total"], f"{openai['accuracy']:.2%}")
    print("BGE", bge["correct"], bge["total"], f"{bge['accuracy']:.2%}")
    print("DELTA", f"{delta:+.2%}")
    print("REPORT", out)


if __name__ == "__main__":
    main()
