import json
import os
from pathlib import Path

import pytest

from src.questions_processing import QuestionsProcessor


RUN_INTEGRATION = os.getenv("RUN_INTEGRATION_TESTS") == "1"
pytestmark = pytest.mark.integration

QUESTIONS = [
    {"text": "特斯拉2023年营收是多少", "kind": "number"},
]


def _run_provider(embedding_provider: str) -> dict:
    temp_questions = Path("test_questions.json")
    temp_questions.write_text(json.dumps(QUESTIONS, ensure_ascii=False, indent=2), encoding="utf-8")

    processor = QuestionsProcessor(
        vector_db_dir="./databases/vector_dbs",
        documents_dir="./databases/chunked_reports",
        questions_file_path=str(temp_questions),
        new_challenge_pipeline=True,
        subset_path="./subset.csv",
        parent_document_retrieval=True,
        llm_reranking=True,
        llm_reranking_sample_size=20,
        top_n_retrieval=6,
        parallel_requests=1,
        api_provider="openai",
        answering_model="gpt-4o-mini-2024-07-18",
        embedding_provider=embedding_provider,
    )
    return processor.process_all_questions(
        output_path=f"answers_{embedding_provider}.json",
        team_email="test@example.com",
        submission_name=f"{embedding_provider} test",
        submission_file=True,
        pipeline_details=f"Integration fallback test for {embedding_provider}",
    )


@pytest.mark.skipif(not RUN_INTEGRATION, reason="Set RUN_INTEGRATION_TESTS=1 to run API integration tests.")
def test_openai_and_bge_provider_paths_run():
    result_openai = _run_provider("openai")
    result_bge = _run_provider("bge_api")
    assert result_openai["statistics"]["total_questions"] == len(QUESTIONS)
    assert result_bge["statistics"]["total_questions"] == len(QUESTIONS)


if __name__ == "__main__":
    print(_run_provider("openai")["statistics"])
    print(_run_provider("bge_api")["statistics"])
