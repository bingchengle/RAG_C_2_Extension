import json
import pickle
from typing import List, Union
from pathlib import Path
from tqdm import tqdm

from rank_bm25 import BM25Okapi
import faiss
import numpy as np
from tenacity import retry, wait_fixed, stop_after_attempt
from src.embedding_clients import EmbeddingAPIClient


class BM25Ingestor:
    def __init__(self):
        pass

    def create_bm25_index(self, chunks: List[str]) -> BM25Okapi:
        """Create a BM25 index from a list of text chunks."""
        tokenized_chunks = [chunk.split() for chunk in chunks]
        return BM25Okapi(tokenized_chunks)

    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        """Process all reports and save individual BM25 indices.

        Args:
            all_reports_dir (Path): Directory containing the JSON report files
            output_dir (Path): Directory where to save the BM25 indices
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        all_report_paths = list(all_reports_dir.glob("*.json"))

        for report_path in tqdm(all_report_paths, desc="Processing reports for BM25"):

            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)


            text_chunks = [chunk['text'] for chunk in report_data['content']['chunks']]
            bm25_index = self.create_bm25_index(text_chunks)


            sha1_name = report_data["metainfo"]["sha1_name"]
            output_file = output_dir / f"{sha1_name}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(bm25_index, f)

        print(f"Processed {len(all_report_paths)} reports")

class VectorDBIngestor:
    def __init__(self, embedding_provider: str = "openai", embedding_model: str = None):
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.embedding_client = EmbeddingAPIClient(provider=embedding_provider, model=embedding_model)

    @retry(wait=wait_fixed(20), stop=stop_after_attempt(2))
    def _get_embeddings(self, text: Union[str, List[str]]) -> List[float]:
        if isinstance(text, str) and not text.strip():
            raise ValueError("Input text cannot be an empty string.")

        text_items = text if isinstance(text, list) else [text]
        return self.embedding_client.embed_texts(text_items)

    def _create_vector_db(self, embeddings: List[float]):
        embeddings_array = np.array(embeddings, dtype=np.float32)
        dimension = len(embeddings[0])
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_array)
        return index

    def _process_report(self, report: dict):
        text_chunks = [chunk['text'] for chunk in report['content']['chunks']]
        embeddings = self._get_embeddings(text_chunks)
        index = self._create_vector_db(embeddings)
        return index

    def process_reports(self, all_reports_dir: Path, output_dir: Path, use_bge: bool = False):

        if use_bge and self.embedding_provider == "openai":
            self.embedding_provider = "bge_api"
            self.embedding_client = EmbeddingAPIClient(provider=self.embedding_provider, model=self.embedding_model)

        all_report_paths = list(all_reports_dir.glob("*.json"))
        output_dir.mkdir(parents=True, exist_ok=True)

        for report_path in tqdm(all_report_paths, desc="Processing reports"):
            with open(report_path, 'r', encoding='utf-8') as file:
                report_data = json.load(file)
            index = self._process_report(report_data)
            sha1_name = report_data["metainfo"]["sha1_name"]
            suffix = "_bge" if self.embedding_provider in {"bge", "bge_api"} else ""
            faiss_file_path = output_dir / f"{sha1_name}{suffix}.faiss"
            faiss.write_index(index, str(faiss_file_path))

        print(f"Processed {len(all_report_paths)} reports")