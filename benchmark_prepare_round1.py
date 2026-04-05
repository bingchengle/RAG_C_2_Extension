import csv
import argparse
import json
import re
from pathlib import Path

from PyPDF2 import PdfReader


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare round1 benchmark dataset")
    parser.add_argument("--num", type=int, default=15, help="Number of questions to include")
    parser.add_argument("--out-dir", type=str, default="data/benchmark_round1_mini", help="Output benchmark directory")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    challenge = root / "_challenge_source" / "round1"
    out = root / args.out_dir

    (out / "pdf_reports").mkdir(parents=True, exist_ok=True)
    (out / "databases" / "chunked_reports").mkdir(parents=True, exist_ok=True)
    (out / "databases" / "vector_dbs").mkdir(parents=True, exist_ok=True)
    (out / "databases" / "bm25_dbs").mkdir(parents=True, exist_ok=True)

    questions = json.loads((challenge / "questions.json").read_text(encoding="utf-8"))
    answers = json.loads((challenge / "answers.json").read_text(encoding="utf-8"))

    selected_q = questions[: max(1, min(args.num, len(questions)))]
    selected_texts = {q["question"] for q in selected_q}
    selected_answers = [a for a in answers if a.get("question") in selected_texts]

    sha_to_name = {}
    name_to_sha = {}
    with (challenge / "dataset.csv").open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sha = row["sha1"]
            name = row["name"]
            sha_to_name[sha] = name
            name_to_sha[name] = sha

    quoted = set()
    for q in selected_q:
        quoted.update(re.findall(r'"([^"]+)"', q["question"]))

    needed = []
    for name in quoted:
        if name in name_to_sha:
            needed.append((name_to_sha[name], name))

    subset_rows = [{"sha1": sha, "company_name": name} for sha, name in sorted(set(needed))]
    with (out / "subset.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sha1", "company_name"])
        writer.writeheader()
        writer.writerows(subset_rows)

    project_questions = [{"text": q["question"], "kind": q["schema"]} for q in selected_q]
    (out / "questions.json").write_text(json.dumps(project_questions, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "gold_answers.json").write_text(json.dumps(selected_answers, ensure_ascii=False, indent=2), encoding="utf-8")

    for sha, name in sorted(set(needed)):
        pdf_src = challenge / "pdfs" / f"{sha}.pdf"
        if not pdf_src.exists():
            continue
        pdf_dst = out / "pdf_reports" / f"{sha}.pdf"
        pdf_dst.write_bytes(pdf_src.read_bytes())

        reader = PdfReader(str(pdf_src))
        pages = []
        chunks = []
        for idx, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            pages.append({"page": idx, "text": text})
            chunks.append({"page": idx, "text": text})

        doc = {
            "metainfo": {
                "company_name": name,
                "sha1_name": sha,
                "sha1": sha,
            },
            "content": {
                "pages": pages,
                "chunks": chunks,
            },
        }
        (out / "databases" / "chunked_reports" / f"{sha}.json").write_text(
            json.dumps(doc, ensure_ascii=False),
            encoding="utf-8",
        )

    print(f"selected_questions={len(selected_q)}")
    print(f"gold_entries={len(selected_answers)}")
    print(f"needed_companies={len(set(needed))}")
    print(f"output={out}")


if __name__ == "__main__":
    main()
