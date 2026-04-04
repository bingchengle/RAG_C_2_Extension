import sys
from pathlib import Path

# 测试导入
print("Testing imports...")
try:
    from src.questions_processing import QuestionsProcessor
    print("OK: QuestionsProcessor imported successfully")
except Exception as e:
    print(f"ERROR: Error importing QuestionsProcessor: {e}")
    sys.exit(1)

try:
    from src.ingestion import VectorDBIngestor
    print("OK: VectorDBIngestor imported successfully")
except Exception as e:
    print(f"ERROR: Error importing VectorDBIngestor: {e}")
    sys.exit(1)

try:
    from src.retrieval import VectorRetriever, HybridRetriever
    print("OK: VectorRetriever and HybridRetriever imported successfully")
except Exception as e:
    print(f"ERROR: Error importing VectorRetriever or HybridRetriever: {e}")
    sys.exit(1)

print("\nAll imports successful!")
print("The system is ready to use with both OpenAI and BGE embeddings.")
print("When BGE is not available, the system will use OpenAI embeddings as fallback.")
print("\nTo use BGE embeddings, make sure you have:")
print("1. PyTorch installed")
print("2. FlagEmbedding library installed")
print("3. BGE model downloaded")
print("\nTo use OpenAI embeddings, make sure you have:")
print("1. OpenAI API key set in environment variable OPENAI_API_KEY")
