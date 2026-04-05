def test_core_imports():
    from src.questions_processing import QuestionsProcessor  # noqa: F401
    from src.ingestion import VectorDBIngestor  # noqa: F401
    from src.retrieval import VectorRetriever, HybridRetriever  # noqa: F401

    assert True
