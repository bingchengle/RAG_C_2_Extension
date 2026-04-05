import os
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI


class EmbeddingAPIClient:
    """Unified embedding client for multiple API providers."""

    def __init__(self, provider: str = "openai", model: Optional[str] = None):
        load_dotenv()
        self.provider = (provider or "openai").lower()
        self.model = model or self._default_model_for_provider(self.provider)
        self.client = self._build_client()

    def _default_model_for_provider(self, provider: str) -> str:
        if provider in {"bge", "bge_api"}:
            return os.getenv("BGE_EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
        return os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

    def _build_client(self) -> OpenAI:
        if self.provider in {"bge", "bge_api"}:
            api_key = os.getenv("BGE_API_KEY")
            base_url = os.getenv("BGE_API_BASE_URL", "https://api.siliconflow.cn/v1")
            if not api_key:
                raise ValueError("BGE_API_KEY is required when embedding provider is 'bge_api'.")
            return OpenAI(api_key=api_key, base_url=base_url, timeout=None, max_retries=2)

        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when embedding provider is 'openai'.")
        if base_url:
            return OpenAI(api_key=api_key, base_url=base_url, timeout=None, max_retries=2)
        return OpenAI(api_key=api_key, timeout=None, max_retries=2)

    def _sanitize_text(self, text: str) -> str:
        cleaned = " ".join(text.split())
        if self.provider in {"bge", "bge_api"}:
            # Some BGE gateways require each input to be under ~512 tokens.
            # Char-level truncation is a robust proxy without extra tokenizer dependency.
            return cleaned[:350]
        return cleaned

    def embed_texts(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        if not texts:
            return []
        if any((not isinstance(text, str)) or (not text.strip()) for text in texts):
            raise ValueError("All embedding inputs must be non-empty strings.")
        texts = [self._sanitize_text(text) for text in texts]
        if batch_size is None:
            # Some OpenAI-compatible BGE gateways enforce small batch limits (e.g. 32).
            batch_size = 1 if self.provider in {"bge", "bge_api"} else 512

        embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(input=batch, model=self.model)
            embeddings.extend([item.embedding for item in response.data])
        return embeddings


class BGERerankerClient:
    """BGE API reranker client."""

    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("BGE_API_KEY")
        self.base_url = os.getenv("BGE_RERANK_BASE_URL", os.getenv("BGE_API_BASE_URL", "https://api.siliconflow.cn/v1")).rstrip("/")
        self.model = os.getenv("BGE_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
        if not self.api_key:
            raise ValueError("BGE_API_KEY is required for BGE reranking.")

    def rerank(self, query: str, documents: List[str]) -> List[float]:
        if not documents:
            return []

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
            "return_documents": False
        }

        response = requests.post(f"{self.base_url}/rerank", headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])

        # API returns only top_n ranked docs with original index.
        scores_map: Dict[int, float] = {}
        for item in results:
            idx = int(item.get("index", -1))
            score = float(item.get("relevance_score", 0.0))
            if idx >= 0:
                scores_map[idx] = score

        scores: List[float] = []
        for i in range(len(documents)):
            scores.append(scores_map.get(i, 0.0))
        return scores
