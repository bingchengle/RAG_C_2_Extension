import os
import json
import pickle
from typing import List, Union
from pathlib import Path
from tqdm import tqdm

from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
from tenacity import retry, wait_fixed, stop_after_attempt

# 延迟导入 FlagEmbedding
FlagModel = None


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
            # Load the report
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
                
            # Extract text chunks and create BM25 index
            text_chunks = [chunk['text'] for chunk in report_data['content']['chunks']]
            bm25_index = self.create_bm25_index(text_chunks)
            
            # Save BM25 index
            sha1_name = report_data["metainfo"]["sha1_name"]
            output_file = output_dir / f"{sha1_name}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(bm25_index, f)
                
        print(f"Processed {len(all_report_paths)} reports")

class VectorDBIngestor:
    def __init__(self):
        self.llm = self._set_up_llm()
        self.bge_model = self._set_up_bge_model()

    def _set_up_llm(self):
        load_dotenv()
        llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=None,
            max_retries=2
        )
        return llm

    def _set_up_bge_model(self):
        global FlagModel
        if FlagModel is None:
            try:
                from FlagEmbedding import FlagModel
            except ImportError as e:
                raise ImportError(f"FlagEmbedding is not available: {e}. BGE embeddings cannot be used.")
        return FlagModel(
            "BAAI/bge-large-zh-v1.5",
            use_fp16=True
        )

    @retry(wait=wait_fixed(20), stop=stop_after_attempt(2))
    def _get_embeddings(self, text: Union[str, List[str]], model: str = "text-embedding-3-large") -> List[float]:
        if isinstance(text, str) and not text.strip():
            raise ValueError("Input text cannot be an empty string.")
        
        if isinstance(text, list):
            text_chunks = [text[i:i + 1024] for i in range(0, len(text), 1024)]
        else:
            text_chunks = [text]

        embeddings = []
        for chunk in text_chunks:
            response = self.llm.embeddings.create(input=chunk, model=model)
            embeddings.extend([embedding.embedding for embedding in response.data])
        
        return embeddings

    def _get_bge_embeddings(self, text: Union[str, List[str]]) -> List[float]:
        if isinstance(text, str) and not text.strip():
            raise ValueError("Input text cannot be an empty string.")
        
        if isinstance(text, list):
            embeddings = self.bge_model.encode(text, batch_size=32)
        else:
            embeddings = [self.bge_model.encode(text)]
        
        return embeddings

    def _create_vector_db(self, embeddings: List[float]):
        embeddings_array = np.array(embeddings, dtype=np.float32)
        dimension = len(embeddings[0])
        index = faiss.IndexFlatIP(dimension)  # Cosine distance
        index.add(embeddings_array)
        return index
    
    def _process_report(self, report: dict, use_bge: bool = False):
        text_chunks = [chunk['text'] for chunk in report['content']['chunks']]
        if use_bge:
            embeddings = self._get_bge_embeddings(text_chunks)
        else:
            embeddings = self._get_embeddings(text_chunks)
        index = self._create_vector_db(embeddings)
        return index

    def process_reports(self, all_reports_dir: Path, output_dir: Path, use_bge: bool = False):
        all_report_paths = list(all_reports_dir.glob("*.json"))
        output_dir.mkdir(parents=True, exist_ok=True)

        for report_path in tqdm(all_report_paths, desc="Processing reports"):
            with open(report_path, 'r', encoding='utf-8') as file:
                report_data = json.load(file)
            index = self._process_report(report_data, use_bge)
            sha1_name = report_data["metainfo"]["sha1_name"]
            suffix = "_bge" if use_bge else ""
            faiss_file_path = output_dir / f"{sha1_name}{suffix}.faiss"
            faiss.write_index(index, str(faiss_file_path))

        print(f"Processed {len(all_report_paths)} reports")