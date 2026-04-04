import json
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
    },
    {
        "text": "特斯拉2023年收入增长率是多少",
        "kind": "number"
    },
    {
        "text": "特斯拉2023年净利润增长率是多少",
        "kind": "number"
    }
]

# 保存测试问题到文件
with open("test_questions.json", "w", encoding="utf-8") as f:
    json.dump(questions, f, ensure_ascii=False, indent=2)

# 初始化 QuestionsProcessor
processor = QuestionsProcessor(
    vector_db_dir="./databases/vector_dbs",
    documents_dir="./databases/chunked_reports",
    questions_file_path="test_questions.json",
    new_challenge_pipeline=True,
    subset_path="./subset.csv",
    parent_document_retrieval=True,
    llm_reranking=True,
    llm_reranking_sample_size=30,
    top_n_retrieval=10,
    parallel_requests=10,
    api_provider="openai",
    answering_model="gpt-4o-mini-2024-07-18",
    full_context=False
)

# 处理所有问题
print("Processing questions...")
result = processor.process_all_questions(
    output_path="test_answers.json",
    team_email="test@example.com",
    submission_name="Test API Answer Generator",
    submission_file=True,
    pipeline_details="Using AnswerGeneratorOptimized with API"
)

# 打印结果
print("\nProcessing completed!")
print(f"Total questions: {result['statistics']['total_questions']}")
print(f"Successfully answered: {result['statistics']['success_count']}")
print(f"Errors: {result['statistics']['error_count']}")
print(f"N/A answers: {result['statistics']['na_count']}")

# 打印生成的答案
print("\nGenerated answers:")
for i, question in enumerate(result['questions']):
    print(f"\nQuestion {i+1}: {question['question_text']}")
    print(f"Answer: {question['value']}")
    if 'error' in question:
        print(f"Error: {question['error']}")
    if 'references' in question and question['references']:
        print(f"References: {question['references']}")
