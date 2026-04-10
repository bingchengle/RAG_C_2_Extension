"""Check merged 140 benchmark: each question resolves to >=1 SHA (Round1: quoted names in dataset.csv; ERC2: answers.json pools or subset substring)."""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

R1 = ROOT / "_challenge_source" / "round1"
ERC = ROOT / "data" / "datasets" / "enterprise_rag_round2"


def main() -> None:
    name_to_sha: dict[str, str] = {}
    with (R1 / "dataset.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            name_to_sha[row["name"]] = row["sha1"]

    def r1_resolve(text: str) -> tuple[list[str], list[str]]:
        names = re.findall(r'"([^"]+)"', text)
        if not names:
            return [], ["no_quoted_company"]
        shas: list[str] = []
        miss: list[str] = []
        for n in names:
            sha = name_to_sha.get(n)
            if sha:
                shas.append(sha)
            else:
                miss.append(n)
        return shas, miss

    answers_obj = json.loads((ERC / "answers.json").read_text(encoding="utf-8"))
    questions_erc = json.loads((ERC / "questions.json").read_text(encoding="utf-8"))
    subset_rows = list(csv.DictReader((ERC / "subset.csv").open(encoding="utf-8", newline="")))

    def erc_resolve(i: int) -> tuple[list[str], list[str]]:
        text = questions_erc[i]["text"]
        payload = answers_obj.get(text)
        if not payload:
            return [], ["missing_answers_json_key"]
        q_shas: set[str] = set()
        for pool in payload.get("reference_pools") or []:
            for ref in pool:
                if isinstance(ref, str) and ":" in ref:
                    q_shas.add(ref.split(":", 1)[0])
        if not q_shas:
            for row in subset_rows:
                cn = row.get("company_name") or ""
                if cn and cn in text:
                    q_shas.add(row["sha1"])
        if not q_shas:
            return [], ["empty_pools_and_no_substring_match"]
        return list(q_shas), []

    r1g = json.loads((R1 / "answers.json").read_text(encoding="utf-8"))
    problems: list[tuple[str, int, str, object]] = []
    at_least_one_sha = 0

    partial_r1: list[tuple] = []
    for i in range(40):
        text = r1g[i]["question"]
        shas, miss = r1_resolve(text)
        if shas:
            at_least_one_sha += 1
        if not shas:
            problems.append(("round1", i, text[:90], miss or "unknown"))
        elif miss:
            partial_r1.append((i, miss, text[:80]))

    for i in range(100):
        shas, err = erc_resolve(i)
        if shas:
            at_least_one_sha += 1
        if not shas:
            t = questions_erc[i]["text"]
            problems.append(("erc2", i, t[:90], err))

    print(f"Total questions: 140")
    print(f"With >=1 resolved SHA (under this heuristic): {at_least_one_sha}")
    print(f"With NO SHA: {len(problems)}")
    for src, idx, tshort, detail in problems:
        print(f"--- {src} local_index={idx} ---")
        print(f"    detail: {detail}")
        print(f"    Q: {tshort}...")
    if partial_r1:
        print()
        print(f"Round1 questions with >=1 SHA but some unknown quoted names: {len(partial_r1)}")
        for idx, miss, tshort in partial_r1[:15]:
            print(f"  idx={idx} unknown={miss!r} Q={tshort}...")
        if len(partial_r1) > 15:
            print(f"  ... and {len(partial_r1) - 15} more")


if __name__ == "__main__":
    main()
