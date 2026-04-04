import json
import time
from src.questions_processing import QuestionsProcessor
from pathlib import Path

# 测试问题
questions = [
    {
        "text": "特斯拉2023年营收是多少",
        "kind": "number"
    },
    {
        "text": "特斯拉2023年净利润是多少",
        "kind": "number"
    }
]

# 保存测试问题到文件
with open("test_questions.json", "w", encoding="utf-8") as f:
    json.dump(questions, f, ensure_ascii=False, indent=2)

# 测试配置
vector_db_dir = "./databases/vector_dbs"
documents_dir = "./databases/chunked_reports"
questions_file_path = "test_questions.json"
subset_path = "./subset.csv"

# 测试：使用 OpenAI 嵌入（应该正常工作）
print("=== Testing with OpenAI Embeddings ===")
start_time = time.time()

try:
    processor_openai = QuestionsProcessor(
        vector_db_dir=vector_db_dir,
        documents_dir=documents_dir,
        questions_file_path=questions_file_path,
        new_challenge_pipeline=True,
        subset_path=subset_path,
        parent_document_retrieval=True,
        llm_reranking=True,
        llm_reranking_sample_size=30,
        top_n_retrieval=10,
        parallel_requests=10,
        api_provider="openai",
        answering_model="gpt-4o-mini-2024-07-18",
        full_context=False,
        use_bge=False
    )

    result_openai = processor_openai.process_all_questions(
        output_path="answers_openai.json",
        team_email="test@example.com",
        submission_name="OpenAI Embeddings Test",
        submission_file=True,
        pipeline_details="Using OpenAI embeddings (text-embedding-3-large)"
    )

    openai_time = time.time() - start_time
    print(f"OpenAI embeddings test completed in {openai_time:.2f} seconds")
    print(f"Success count: {result_openai['statistics']['success_count']}")
    print(f"Error count: {result_openai['statistics']['error_count']}")
    
except Exception as e:
    print(f"Error with OpenAI embeddings: {str(e)}")

# 测试：尝试使用 BGE 嵌入（应该会回退到 OpenAI 或显示错误）
print("\n=== Testing with BGE Embeddings ===")
start_time = time.time()

try:
    processor_bge = QuestionsProcessor(
        vector_db_dir=vector_db_dir,
        documents_dir=documents_dir,
        questions_file_path=questions_file_path,
        new_challenge_pipeline=True,
        subset_path=subset_path,
        parent_document_retrieval=True,
        llm_reranking=True,
        llm_reranking_sample_size=30,
        top_n_retrieval=10,
        parallel_requests=10,
        api_provider="openai",
        answering_model="gpt-4o-mini-2024-07-18",
        full_context=False,
        use_bge=True
    )

    result_bge = processor_bge.process_all_questions(
        output_path="answers_bge.json",
        team_email="test@example.com",
        submission_name="BGE Embeddings Test",
        submission_file=True,
        pipeline_details="Using BGE embeddings (BAAI/bge-large-zh-v1.5)"
    )

    bge_time = time.time() - start_time
    print(f"BGE embeddings test completed in {bge_time:.2f} seconds")
    print(f"Success count: {result_bge['statistics']['success_count']}")
    print(f"Error count: {result_bge['statistics']['error_count']}")
    
except Exception as e:
    print(f"Error with BGE embeddings: {str(e)}")
    print("This is expected if BGE is not available. The system should fallback to OpenAI embeddings.")

print("\nTesting completed!")
