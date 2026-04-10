"""
Optimized text splitting with semantic chunking and context window enhancement
"""
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from FlagEmbedding import FlagModel


class SemanticTextSplitter:
    """See implementation."""

    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-large-zh-v1.5",
        similarity_threshold: float = 0.7,
        min_chunk_size: int = 100,
        max_chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """See implementation."""
        self.embedding_model = FlagModel(
            embedding_model_name,
            use_fp16=True
        )
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap


        self.base_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4o",
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap
        )

    def _split_into_sentences(self, text: str) -> List[str]:
        """See implementation."""

        import re

        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _get_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        """See implementation."""
        return self.embedding_model.encode(sentences, batch_size=32)

    def _find_semantic_boundaries(
        self,
        sentences: List[str],
        embeddings: np.ndarray
    ) -> List[int]:
        """See implementation."""
        if len(sentences) <= 1:
            return []

        boundaries = []


        for i in range(len(sentences) - 1):
            similarity = cosine_similarity(
                embeddings[i:i+1],
                embeddings[i+1:i+2]
            )[0][0]


            if similarity < self.similarity_threshold:
                boundaries.append(i + 1)

        return boundaries

    def _merge_small_chunks(
        self,
        chunks: List[str],
        min_size: int
    ) -> List[str]:
        """See implementation."""
        if not chunks:
            return chunks

        merged = []
        current_chunk = chunks[0]

        for chunk in chunks[1:]:
            current_tokens = self.count_tokens(current_chunk)
            chunk_tokens = self.count_tokens(chunk)


            if current_tokens < min_size and current_tokens + chunk_tokens < self.max_chunk_size:
                current_chunk += " " + chunk
            else:
                merged.append(current_chunk)
                current_chunk = chunk

        merged.append(current_chunk)
        return merged

    def semantic_split(self, text: str) -> List[str]:
        """See implementation."""

        sentences = self._split_into_sentences(text)
        if len(sentences) <= 1:
            return [text] if text else []


        embeddings = self._get_sentence_embeddings(sentences)


        boundaries = self._find_semantic_boundaries(sentences, embeddings)


        chunks = []
        start = 0
        for boundary in boundaries:
            chunk = " ".join(sentences[start:boundary])
            if chunk:
                chunks.append(chunk)
            start = boundary


        if start < len(sentences):
            chunk = " ".join(sentences[start:])
            if chunk:
                chunks.append(chunk)


        chunks = self._merge_small_chunks(chunks, self.min_chunk_size)


        final_chunks = []
        for chunk in chunks:
            if self.count_tokens(chunk) > self.max_chunk_size:

                sub_chunks = self.base_splitter.split_text(chunk)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)

        return final_chunks

    def count_tokens(self, string: str, encoding_name: str = "o200k_base") -> int:
        """See implementation."""
        import tiktoken
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(string))


class ContextEnhancedChunk:
    """See implementation."""

    def __init__(
        self,
        text: str,
        page: int,
        chunk_id: int,
        prev_context: str = "",
        next_context: str = "",
        metadata: Dict = None
    ):
        self.text = text
        self.page = page
        self.chunk_id = chunk_id
        self.prev_context = prev_context
        self.next_context = next_context
        self.metadata = metadata or {}

    def to_dict(self) -> Dict:
        """See implementation."""
        return {
            'id': self.chunk_id,
            'page': self.page,
            'text': self.text,
            'prev_context': self.prev_context,
            'next_context': self.next_context,
            'full_text': f"{self.prev_context}\n\n{self.text}\n\n{self.next_context}".strip(),
            'length_tokens': self._count_tokens(),
            **self.metadata
        }

    def _count_tokens(self) -> int:
        """See implementation."""
        import tiktoken
        encoding = tiktoken.get_encoding("o200k_base")
        return len(encoding.encode(self.text))


class ContextEnhancedTextSplitter:
    """See implementation."""

    def __init__(
        self,
        semantic_splitter: Optional[SemanticTextSplitter] = None,
        context_window_size: int = 100
    ):
        """See implementation."""
        self.semantic_splitter = semantic_splitter
        self.context_window_size = context_window_size

        if semantic_splitter is None:
            self.base_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                model_name="gpt-4o",
                chunk_size=300,
                chunk_overlap=50
            )

    def _extract_context(
        self,
        chunks: List[str],
        idx: int,
        window_size: int
    ) -> Tuple[str, str]:
        """See implementation."""
        prev_context = ""
        next_context = ""


        if idx > 0:
            prev_text = chunks[idx - 1]
            tokens = prev_text.split()
            if len(tokens) > window_size:
                prev_context = " ".join(tokens[-window_size:])
            else:
                prev_context = prev_text


        if idx < len(chunks) - 1:
            next_text = chunks[idx + 1]
            tokens = next_text.split()
            if len(tokens) > window_size:
                next_context = " ".join(tokens[:window_size])
            else:
                next_context = next_text

        return prev_context, next_context

    def split_page(self, page: Dict[str, any]) -> List[Dict]:
        """See implementation."""
        text = page['text']
        page_num = page['page']


        if self.semantic_splitter:
            chunks = self.semantic_splitter.semantic_split(text)
        else:
            chunks = self.base_splitter.split_text(text)


        enhanced_chunks = []
        for idx, chunk_text in enumerate(chunks):
            prev_context, next_context = self._extract_context(
                chunks, idx, self.context_window_size
            )

            enhanced_chunk = ContextEnhancedChunk(
                text=chunk_text,
                page=page_num,
                chunk_id=idx,
                prev_context=prev_context,
                next_context=next_context,
                metadata={
                    'type': 'content',
                    'has_prev': idx > 0,
                    'has_next': idx < len(chunks) - 1
                }
            )

            enhanced_chunks.append(enhanced_chunk.to_dict())

        return enhanced_chunks

    def split_report(
        self,
        file_content: Dict[str, any],
        serialized_tables_report_path: Optional[Path] = None
    ) -> Dict[str, any]:
        """See implementation."""
        chunks = []
        chunk_id = 0


        tables_by_page = {}
        if serialized_tables_report_path is not None:
            with open(serialized_tables_report_path, 'r', encoding='utf-8') as f:
                parsed_report = json.load(f)
            tables_by_page = self._get_serialized_tables_by_page(
                parsed_report.get('tables', [])
            )


        for page in file_content['content']['pages']:
            page_chunks = self.split_page(page)
            for chunk in page_chunks:
                chunk['id'] = chunk_id
                chunk_id += 1
                chunks.append(chunk)


            if tables_by_page and page['page'] in tables_by_page:
                for table in tables_by_page[page['page']]:
                    table['id'] = chunk_id
                    table['type'] = 'serialized_table'
                    chunk_id += 1
                    chunks.append(table)

        file_content['content']['chunks'] = chunks
        return file_content

    def _get_serialized_tables_by_page(self, tables: List[Dict]) -> Dict[int, List[Dict]]:
        """See implementation."""
        tables_by_page = {}
        for table in tables:
            if 'serialized' not in table:
                continue

            page = table['page']
            if page not in tables_by_page:
                tables_by_page[page] = []

            table_text = "\n".join(
                block["information_block"]
                for block in table["serialized"]["information_blocks"]
            )

            tables_by_page[page].append({
                "page": page,
                "text": table_text,
                "table_id": table["table_id"],
                "length_tokens": self._count_tokens(table_text),
                "type": "serialized_table"
            })

        return tables_by_page

    def _count_tokens(self, string: str) -> int:
        """See implementation."""
        import tiktoken
        encoding = tiktoken.get_encoding("o200k_base")
        return len(encoding.encode(string))

    def split_all_reports(
        self,
        all_report_dir: Path,
        output_dir: Path,
        serialized_tables_dir: Optional[Path] = None
    ):
        """See implementation."""
        all_report_paths = list(all_report_dir.glob("*.json"))

        for report_path in all_report_paths:
            serialized_tables_path = None
            if serialized_tables_dir is not None:
                serialized_tables_path = serialized_tables_dir / report_path.name

            with open(report_path, 'r', encoding='utf-8') as file:
                report_data = json.load(file)

            updated_report = self.split_report(report_data, serialized_tables_path)
            output_dir.mkdir(parents=True, exist_ok=True)

            with open(output_dir / report_path.name, 'w', encoding='utf-8') as file:
                json.dump(updated_report, file, indent=2, ensure_ascii=False)

        print(f"Split {len(all_report_paths)} files with semantic chunking")



TextSplitter = ContextEnhancedTextSplitter
