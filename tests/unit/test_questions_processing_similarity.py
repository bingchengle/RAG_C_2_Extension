"""Unit tests for similarity_mode behavior in QuestionsProcessor (no external APIs)."""

from unittest.mock import MagicMock

import pytest

from src.questions_processing import QuestionsProcessor


@pytest.fixture
def processor_openai_similarity(tmp_path):
    return QuestionsProcessor(
        questions_file_path=None,
        vector_db_dir=tmp_path,
        bm25_db_dir=tmp_path,
        documents_dir=tmp_path,
        enable_similarity_check=True,
        similarity_mode="openai_large",
        answering_model="gpt-4o-mini-2024-07-18",
    )


def test_similarity_mode_off_skips_even_when_enabled():
    p = QuestionsProcessor(
        questions_file_path=None,
        vector_db_dir=".",
        bm25_db_dir=".",
        documents_dir=".",
        enable_similarity_check=True,
        similarity_mode="off",
    )
    assert p._calculate_answer_similarity("answer", [{"text": "ctx"}]) is None


def test_openai_large_records_usage(monkeypatch, processor_openai_similarity):
    p = processor_openai_similarity
    monkeypatch.setattr(
        "src.questions_processing.VectorRetriever.get_strings_cosine_similarity",
        staticmethod(lambda a, b: 0.42),
    )
    before = p.route_usage.similarity_openai_embedding_batches
    out = p._calculate_answer_similarity("a", [{"text": "b"}, {"text": "c"}])
    assert out is not None
    assert out["context_similarity"] == 0.42
    assert p.route_usage.similarity_openai_embedding_batches > before


def test_same_as_query_uses_embedding_client(monkeypatch, tmp_path):
    p = QuestionsProcessor(
        questions_file_path=None,
        vector_db_dir=tmp_path,
        bm25_db_dir=tmp_path,
        documents_dir=tmp_path,
        enable_similarity_check=True,
        similarity_mode="same_as_query",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
    )
    mock_client = MagicMock()
    mock_client.embed_texts.side_effect = [
        [[1.0, 0.0], [1.0, 0.0]],
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
    ]
    monkeypatch.setattr(
        "src.questions_processing.EmbeddingAPIClient",
        lambda provider, model: mock_client,
    )
    before = p.route_usage.similarity_query_embedding_batches
    out = p._calculate_answer_similarity("x", [{"text": "y"}, {"text": "z"}])
    assert out is not None
    assert mock_client.embed_texts.call_count == 2
    assert p.route_usage.similarity_query_embedding_batches == before + 2
    assert out["context_similarity"] == 1.0
    assert out["best_chunk_similarity"] == 1.0
