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
    {"text": "特斯拉2023年收入增长率是多少", "kind": "number"},
    {"text": "特斯拉2023年净利润增长率是多少", "kind": "number"},
]


def _run_test() -> dict:
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
        parallel_requests=10,
        api_provider="openai",
        answering_model="gpt-4o-mini-2024-07-18",
        full_context=False,
    )

    return processor.process_all_questions(
        output_path="test_answers.json",
        team_email="test@example.com",
        submission_name="Test API Answer Generator",
        submission_file=True,
        pipeline_details="Integration test for API answer generation",
    )


@pytest.mark.skipif(not RUN_INTEGRATION, reason="Set RUN_INTEGRATION_TESTS=1 to run API integration tests.")
def test_api_answer_generator_integration():
    result = _run_test()
    assert "statistics" in result
    assert result["statistics"]["total_questions"] == len(QUESTIONS)


if __name__ == "__main__":
    result = _run_test()
    print(result["statistics"])
