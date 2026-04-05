import json
import os
from pathlib import Path

import pytest

from src.questions_processing import QuestionsProcessor


RUN_INTEGRATION = os.getenv("RUN_INTEGRATION_TESTS") == "1"
pytestmark = pytest.mark.integration

QUESTIONS = [
    {"text": "特斯拉2023年营收是多少", "kind": "number"},
    {"text": "特斯拉2023年净利润是多少", "kind": "number"},
]


def _run_for_provider(embedding_provider: str, output_path: str) -> dict:
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
        llm_reranking_sample_size=30,
        top_n_retrieval=10,
        parallel_requests=2,
        api_provider="openai",
        answering_model="gpt-4o-mini-2024-07-18",
        full_context=False,
        embedding_provider=embedding_provider,
        reranker_type="bge" if embedding_provider == "bge_api" else "llm",
    )
    return processor.process_all_questions(
        output_path=output_path,
        team_email="test@example.com",
        submission_name=f"{embedding_provider} Embeddings Test",
        submission_file=True,
        pipeline_details=f"Integration test with {embedding_provider} embeddings",
    )


@pytest.mark.skipif(not RUN_INTEGRATION, reason="Set RUN_INTEGRATION_TESTS=1 to run API integration tests.")
def test_embedding_comparison_integration():
    result_openai = _run_for_provider("openai", "answers_openai.json")
    result_bge = _run_for_provider("bge_api", "answers_bge.json")
    assert result_openai["statistics"]["total_questions"] == len(QUESTIONS)
    assert result_bge["statistics"]["total_questions"] == len(QUESTIONS)


if __name__ == "__main__":
    print(_run_for_provider("openai", "answers_openai.json")["statistics"])
    print(_run_for_provider("bge_api", "answers_bge.json")["statistics"])
