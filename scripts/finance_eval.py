import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import click


def _load_submission(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "answers" in payload:
        return payload["answers"]
    if isinstance(payload, dict) and "questions" in payload:
        return payload["questions"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported file format: {path}")


def _to_value(item: Dict[str, Any]) -> Any:
    if "value" in item:
        return item.get("value")
    return item.get("answer")


def _is_na(value: Any) -> bool:
    return isinstance(value, str) and value.strip().upper() == "N/A"


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float))


@click.command()
@click.option("--pred", "pred_path", required=True, type=click.Path(exists=True, path_type=Path), help="Prediction json path.")
@click.option("--gold", "gold_path", required=False, type=click.Path(exists=True, path_type=Path), help="Optional gold json path.")
def main(pred_path: Path, gold_path: Optional[Path]):
    """Lightweight finance QA evaluation helper."""
    preds = _load_submission(pred_path)
    total = len(preds)
    values = [_to_value(item) for item in preds]

    na_count = sum(1 for v in values if _is_na(v))
    numeric_count = sum(1 for v in values if _is_numeric(v))
    with_refs_count = sum(1 for item in preds if bool(item.get("references")))

    print("=== Finance Evaluation (Basic) ===")
    print(f"Total: {total}")
    print(f"Numeric answers: {numeric_count} ({(numeric_count / total * 100) if total else 0:.1f}%)")
    print(f"N/A answers: {na_count} ({(na_count / total * 100) if total else 0:.1f}%)")
    print(f"Answers with references: {with_refs_count} ({(with_refs_count / total * 100) if total else 0:.1f}%)")

    if not gold_path:
        return

    gold_items = _load_submission(gold_path)
    if len(gold_items) != len(preds):
        print(f"[WARN] gold/pred size mismatch: {len(gold_items)} vs {len(preds)}")

    compare_count = min(len(gold_items), len(preds))
    exact_match = 0
    numeric_close = 0
    for idx in range(compare_count):
        pv = _to_value(preds[idx])
        gv = _to_value(gold_items[idx])
        if pv == gv:
            exact_match += 1
            continue
        if isinstance(pv, (int, float)) and isinstance(gv, (int, float)):
            base = max(abs(float(gv)), 1.0)
            if abs(float(pv) - float(gv)) / base <= 0.01:
                numeric_close += 1

    print("\n=== Against Gold ===")
    print(f"Compared: {compare_count}")
    print(f"Exact match: {exact_match} ({(exact_match / compare_count * 100) if compare_count else 0:.1f}%)")
    print(f"Numeric <=1% relative error: {numeric_close} ({(numeric_close / compare_count * 100) if compare_count else 0:.1f}%)")


if __name__ == "__main__":
    main()
