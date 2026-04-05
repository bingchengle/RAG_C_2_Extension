import json
import logging
from typing import List, Tuple, Dict
from rank_bm25 import BM25Okapi
import pickle
from pathlib import Path
import faiss
import numpy as np
from src.reranking import LLMReranker
from src.embedding_clients import EmbeddingAPIClient, BGERerankerClient

_log = logging.getLogger(__name__)
'''
logging:Python 内置的日志模块，提供完整的日志记录功能
getLogger():logging 模块的核心函数，作用是创建 / 获取一个「命名的日志记录器对象」
__name__:Python 内置的魔法变量，代表「当前模块的名字」：
- 如果是主运行模块（直接执行的文件），__name__ = __main__；
- 如果是被导入的模块（比如 import utils），__name__ = 模块名（如 utils）
'''
'''
table_serialization中的日志处理器是否可以这种形式实现，更加简洁？
'''

class BM25Retriever:
    def __init__(self, bm25_db_dir: Path, documents_dir: Path):
        self.bm25_db_dir = bm25_db_dir
        self.documents_dir = documents_dir#实例化数据库目录以及文档目录
        
    def retrieve_by_company_name(self, company_name: str, query: str, top_n: int = 3, return_parent_pages: bool = False) -> List[Dict]:
        document_path = None
        for path in self.documents_dir.glob("*.json"):
            with open(path, 'r', encoding='utf-8') as f:
                doc = json.load(f)
                if doc["metainfo"]["company_name"] == company_name:
                    document_path = path
                    document = doc
                    break#检索出对应公司名字的路径（目录下）以及对应内容
                    
        if document_path is None:
            raise ValueError(f"No report found with '{company_name}' company name.")#不存在抛出但不中止进程
            
        # Load corresponding BM25 index
        bm25_path = self.bm25_db_dir / f"{document['metainfo']['sha1_name']}.pkl"
        with open(bm25_path, 'rb') as f:
            bm25_index = pickle.load(f)#将拼接后的地址作为索引并转为字节流
            
        # Get the document content and BM25 index
        document = document
        chunks = document["content"]["chunks"]
        pages = document["content"]["pages"]#获取文档中的文本块和页码
        
        # Get BM25 scores for the query
        tokenized_query = query.split()
        scores = bm25_index.get_scores(tokenized_query)#或许可以采用不同的评估分数？
        '''
        .split():Python 字符串的分割方法，默认按空格分割字符串，返回列表。
        BM25 索引的核心方法，接收「分词后的查询列表」，计算查询与所有文档的相关性得分。
        '''

        actual_top_n = min(top_n, len(scores))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:actual_top_n]
        '''
        range(len(scores)):生成 0 到「文档数 - 1」的整数序列，代表每篇文档的原始索引（和文档库一一对应）
        排序的依据：以「索引 i 对应的得分 scores[i]」为关键字排序
        reverse=True:降序
        '''
        
        retrieval_results = []
        seen_pages = set()
        
        for index in top_indices:
            score = round(float(scores[index]), 4)
            #round(..., 4)：保留 4 位小数，避免得分是 1.2231435512 这类超长小数，让结果更整洁
            chunk = chunks[index]
            parent_page = next(page for page in pages if page["page"] == chunk["page"])
            
            if return_parent_pages:#判断是否需要返回整篇文章还是chunk
                if parent_page["page"] not in seen_pages:#返回整篇文档的情况
                    seen_pages.add(parent_page["page"])
                    result = {
                        "distance": score,
                        "page": parent_page["page"],
                        "text": parent_page["text"]
                    }
                    retrieval_results.append(result)
            else:#返回chunk的情况
                result = {
                    "distance": score,
                    "page": chunk["page"],
                    "text": chunk["text"]
                }
                retrieval_results.append(result)
        
        return retrieval_results



class VectorRetriever:
    def __init__(
        self,
        vector_db_dir: Path,
        documents_dir: Path,
        use_bge: bool = False,
        embedding_provider: str = "openai",
        embedding_model: str = None
    ):
        self.vector_db_dir = vector_db_dir
        self.documents_dir = documents_dir
        self.embedding_provider = "bge_api" if use_bge else (embedding_provider or "openai")
        self.embedding_model = embedding_model
        self.vector_db_suffix = "_bge" if self.embedding_provider in {"bge", "bge_api"} else ""
        self.embedding_client = EmbeddingAPIClient(provider=self.embedding_provider, model=self.embedding_model)
        self.all_dbs = self._load_dbs()

    def _load_dbs(self):
        all_dbs = []
        # Get list of JSON document paths
        all_documents_paths = list(self.documents_dir.glob('*.json'))
        vector_db_files = {}
        for db_path in self.vector_db_dir.glob(f"*{self.vector_db_suffix}.faiss"):
            stem = db_path.stem
            normalized_stem = stem[:-len(self.vector_db_suffix)] if self.vector_db_suffix and stem.endswith(self.vector_db_suffix) else stem
            vector_db_files[normalized_stem] = db_path
        
        for document_path in all_documents_paths:
            stem = document_path.stem# 获取当前文档的“文件名”
            if stem not in vector_db_files:
                _log.warning(f"No matching vector DB found for document {document_path.name}")
                continue
            try:
                with open(document_path, 'r', encoding='utf-8') as f:
                    document = json.load(f)
            except Exception as e:
                _log.error(f"Error loading JSON from {document_path.name}: {e}")
                continue
            
            # Validate that the document meets the expected schema
            if not (isinstance(document, dict) and "metainfo" in document and "content" in document):
                _log.warning(f"Skipping {document_path.name}: does not match the expected schema.")
                continue# 校验文档格式：必须是字典，且包含 "metainfo" 和 "content" 字段
            
            try:
                vector_db = faiss.read_index(str(vector_db_files[stem]))
                # faiss.read_index：读取 .faiss 文件，返回 FAISS 索引对象
            except Exception as e:
                _log.error(f"Error reading vector DB for {document_path.name}: {e}")
                continue
                
            report = {
                "name": stem,
                "vector_db": vector_db,
                "document": document
            }
            all_dbs.append(report)
        return all_dbs

    @staticmethod
    def get_strings_cosine_similarity(str1, str2):
        llm = EmbeddingAPIClient(provider="openai").client
        embeddings = llm.embeddings.create(input=[str1, str2], model="text-embedding-3-large")#可以切换模型
        embedding1 = embeddings.data[0].embedding
        embedding2 = embeddings.data[1].embedding
        similarity_score = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        # np.dot(embedding1, embedding2)：向量点积
        # np.linalg.norm(embedding1)：向量的模（长度）
        # 余弦相似度公式：cosθ = (A·B) / (||A|| × ||B||)
        similarity_score = round(similarity_score, 4)
        return similarity_score

    def retrieve_by_company_name(self, company_name: str, query: str, llm_reranking_sample_size: int = None, top_n: int = 3, return_parent_pages: bool = False) -> List[Tuple[str, float]]:
        target_report = None
        for report in self.all_dbs:
            document = report.get("document", {})
            metainfo = document.get("metainfo")
            if not metainfo:
                _log.error(f"Report '{report.get('name')}' is missing 'metainfo'!")
                raise ValueError(f"Report '{report.get('name')}' is missing 'metainfo'!")
            if metainfo.get("company_name") == company_name:
                target_report = report
                break
        
        if target_report is None:
            _log.error(f"No report found with '{company_name}' company name.")
            raise ValueError(f"No report found with '{company_name}' company name.")
        
        document = target_report["document"]
        vector_db = target_report["vector_db"]
        chunks = document["content"]["chunks"]
        pages = document["content"]["pages"]
        
        actual_top_n = min(top_n, len(chunks))
        
        embedding = self.embedding_client.embed_texts([query])[0]
        
        #embedding.data[0].embedding：因为 input=query 是单个字符串，所以 data 列表只有 1 个元素，取第一个即可
        embedding_array = np.array(embedding, dtype=np.float32).reshape(1, -1)
        #将一维向量（形状 [3072,]）转为二维矩阵（形状 [1, 3072]），因为 FAISS 的 search 方法支持批量查询
        distances, indices = vector_db.search(x=embedding_array, k=actual_top_n)
        # x=embedding_array：查询向量（2D 数组）
        # k=actual_top_n：要返回的最相似向量数量
        # 返回值：distances（相似度距离）、indices（相似向量在库中的索引）
    
        retrieval_results = []
        seen_pages = set()
        
        for distance, index in zip(distances[0], indices[0]):
            distance = round(float(distance), 4)
            chunk = chunks[index]
            parent_page = next(page for page in pages if page["page"] == chunk["page"])
            if return_parent_pages:
                if parent_page["page"] not in seen_pages:
                    seen_pages.add(parent_page["page"])
                    result = {
                        "distance": distance,
                        "page": parent_page["page"],
                        "text": parent_page["text"]
                    }
                    retrieval_results.append(result)
            else:
                result = {
                    "distance": distance,
                    "page": chunk["page"],
                    "text": chunk["text"]
                }
                retrieval_results.append(result)
            
        return retrieval_results

    def retrieve_all(self, company_name: str) -> List[Dict]:
        target_report = None
        for report in self.all_dbs:
            document = report.get("document", {})
            metainfo = document.get("metainfo")
            if not metainfo:
                continue
            if metainfo.get("company_name") == company_name:
                target_report = report
                break
        
        if target_report is None:
            _log.error(f"No report found with '{company_name}' company name.")
            raise ValueError(f"No report found with '{company_name}' company name.")
        
        document = target_report["document"]
        pages = document["content"]["pages"]
        
        all_pages = []
        for page in sorted(pages, key=lambda p: p["page"]):
            result = {
                "distance": 0.5,#没有实际意义，仅是设置了相同格式为了与其他检索保持相同，避免切片等操作时报错
                "page": page["page"],
                "text": page["text"]
            }
            all_pages.append(result)
            
        return all_pages


class HybridRetriever:
    def __init__(
        self,
        vector_db_dir: Path,
        bm25_db_dir: Path,
        documents_dir: Path,
        use_bge: bool = False,
        embedding_provider: str = "openai",
        embedding_model: str = None,
        reranker_type: str = "llm",
        domain: str = "general"
    ):
        self.vector_retriever = VectorRetriever(
            vector_db_dir,
            documents_dir,
            use_bge=use_bge,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model
        )
        self.bm25_retriever = BM25Retriever(bm25_db_dir, documents_dir)
        self.reranker_type = (reranker_type or "llm").lower()
        self.reranker = BGERerankerClient() if self.reranker_type == "bge" else LLMReranker(domain=domain)
        
    def retrieve_by_company_name(
        self, 
        company_name: str, 
        query: str, 
        llm_reranking_sample_size: int = 28,
        documents_batch_size: int = 2,
        top_n: int = 6,
        llm_weight: float = 0.7,
        bm25_weight: float = 0.3,
        return_parent_pages: bool = False
    ) -> List[Dict]:
        """
        Retrieve and rerank documents using hybrid approach.
        
        Args:
            company_name: Name of the company to search documents for
            query: Search query
            llm_reranking_sample_size: Number of initial results to retrieve from each retriever
            documents_batch_size: Number of documents to analyze in one LLM prompt
            top_n: Number of final results to return after reranking
            llm_weight: Weight given to LLM scores (0-1)
            bm25_weight: Weight given to BM25 scores (0-1)
            return_parent_pages: Whether to return full pages instead of chunks
            
        Returns:
            List of reranked document dictionaries with scores
        """
        # Get initial results from both retrievers
        vector_results = self.vector_retriever.retrieve_by_company_name(
            company_name=company_name,
            query=query,
            top_n=llm_reranking_sample_size,
            return_parent_pages=return_parent_pages
        )
        
        bm25_results = self.bm25_retriever.retrieve_by_company_name(
            company_name=company_name,
            query=query,
            top_n=llm_reranking_sample_size,
            return_parent_pages=return_parent_pages
        )
        
        # Merge and deduplicate results
        merged_results = self._merge_results(vector_results, bm25_results, bm25_weight)
        
        # Rerank results using LLM
        if self.reranker_type == "bge":
            reranked_results = self._rerank_with_bge(
                query=query,
                documents=merged_results,
                llm_weight=llm_weight
            )
        else:
            reranked_results = self.reranker.rerank_documents(
                query=query,
                documents=merged_results,
                documents_batch_size=documents_batch_size,
                llm_weight=llm_weight
            )
        
        return reranked_results[:top_n]

    def _rerank_with_bge(self, query: str, documents: List[Dict], llm_weight: float = 0.7) -> List[Dict]:
        if not documents:
            return []
        texts = [doc.get("text", "") for doc in documents]
        scores = self.reranker.rerank(query=query, documents=texts)
        vector_weight = 1 - llm_weight
        reranked = []
        for doc, score in zip(documents, scores):
            doc_with_score = doc.copy()
            base_vector_score = 1.0 - float(doc.get("distance", 0.0))
            doc_with_score["relevance_score"] = float(score)
            doc_with_score["combined_score"] = round(
                llm_weight * float(score) + vector_weight * base_vector_score,
                4
            )
            reranked.append(doc_with_score)
        reranked.sort(key=lambda x: x["combined_score"], reverse=True)
        return reranked
    
    def _merge_results(self, vector_results: List[Dict], bm25_results: List[Dict], bm25_weight: float) -> List[Dict]:
        """
        Merge and deduplicate results from vector and BM25 retrievers.
        
        Args:
            vector_results: Results from vector retriever
            bm25_results: Results from BM25 retriever
            bm25_weight: Weight given to BM25 scores
            
        Returns:
            Merged and deduplicated results
        """
        # Create a dictionary to store merged results by page
        merged_dict = {}
        
        # Add vector results
        for result in vector_results:
            page = result['page']
            if page not in merged_dict:
                merged_dict[page] = result.copy()
                # Normalize vector distance to a score (higher is better)
                merged_dict[page]['vector_score'] = 1.0 - result['distance']
            else:
                # If page already exists, keep the higher score
                current_score = 1.0 - result['distance']
                if current_score > merged_dict[page].get('vector_score', 0):
                    merged_dict[page]['vector_score'] = current_score
        
        # Add BM25 results
        for result in bm25_results:
            page = result['page']
            if page not in merged_dict:
                merged_dict[page] = result.copy()
                merged_dict[page]['bm25_score'] = result['distance']
            else:
                # If page already exists, keep the higher score
                current_score = result['distance']
                if current_score > merged_dict[page].get('bm25_score', 0):
                    merged_dict[page]['bm25_score'] = current_score
        
        # Calculate combined score for each result
        for page, result in merged_dict.items():
            vector_score = result.get('vector_score', 0)
            bm25_score = result.get('bm25_score', 0)
            
            # Calculate weighted combined score
            combined_score = (vector_score * (1 - bm25_weight)) + (bm25_score * bm25_weight)
            result['distance'] = 1.0 - combined_score  # Convert to distance for reranking
        
        # Convert back to list and sort by combined score (ascending distance)
        merged_list = list(merged_dict.values())
        merged_list.sort(key=lambda x: x['distance'])
        
        return merged_list
