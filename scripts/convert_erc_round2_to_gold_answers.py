"""
Build benchmark_compare_round1-compatible gold_answers.json from Enterprise RAG Round2 answers.json.

Round2 uses question text as keys and kind in {number, boolean, names}. Scoring expects schema
number vs non-number (name/boolean both use _match_name-style matching in benchmark_compare_round1).

Usage:
  python scripts/convert_erc_round2_to_gold_answers.py
  python scripts/convert_erc_round2_to_gold_answers.py --dataset-dir data/datasets/enterprise_rag_round2 --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_paths import ENTERPRISE_RAG_ROUND2_DIR


def _kind_to_schema(kind: str) -> str:
    k = (kind or "").strip().lower()
    if k == "names":
        return "name"
    if k in ("number", "boolean", "name"):
        return k
    return k


def convert(
    *,
    dataset_dir: Path,
    questions_path: Path | None,
    answers_path: Path | None,
) -> list[dict]:
    q_path = questions_path or (dataset_dir / "questions.json")
    a_path = answers_path or (dataset_dir / "answers.json")
    questions = json.loads(q_path.read_text(encoding="utf-8"))
    answers_obj = json.loads(a_path.read_text(encoding="utf-8"))
    if not isinstance(answers_obj, dict):
        raise ValueError(f"Expected answers.json to be a dict keyed by question, got {type(answers_obj)}")

    gold: list[dict] = []
    for q in questions:
        text = q.get("text") or q.get("question")
        if not text:
            continue
        payload = answers_obj.get(text)
        if payload is None:
            raise KeyError(f"No answer entry for question: {text[:120]!r}...")
        kind = payload.get("kind", "number")
        schema = _kind_to_schema(str(kind))
        ans_list = payload.get("answers", [])
        row = {"question": text, "schema": schema, "answer": ans_list}
        gold.append(row)
    return gold


def main() -> None:
    p = argparse.ArgumentParser(description="Convert ERC Round2 answers.json to gold_answers.json list format")
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=ENTERPRISE_RAG_ROUND2_DIR,
        help="Folder with questions.json and answers.json",
    )
    p.add_argument("--questions", type=Path, default=None, help="Override path to questions.json")
    p.add_argument("--answers", type=Path, default=None, help="Override path to answers.json")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output gold_answers.json (default: <dataset-dir>/gold_answers.json)",
    )
    p.add_argument("--dry-run", action="store_true", help="Print counts only; do not write")
    args = p.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    out_path = Path(args.out) if args.out else dataset_dir / "gold_answers.json"

    gold = convert(
        dataset_dir=dataset_dir,
        questions_path=args.questions,
        answers_path=args.answers,
    )
    print(f"Built {len(gold)} gold rows from {dataset_dir}")
    if args.dry_run:
        return
    out_path.write_text(json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
