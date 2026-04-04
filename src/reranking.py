import os
from dotenv import load_dotenv
from openai import OpenAI
import requests
#简洁易用的 HTTP 请求库，用于发送 HTTP/1.1 请求
import src.prompts as prompts
from concurrent.futures import ThreadPoolExecutor
#作用：提供线程池管理，实现多线程并发执行任务


class JinaReranker:
    def __init__(self):
        self.url = 'https://api.jina.ai/v1/rerank'
        self.headers = self.get_headers()
        
    def get_headers(self):#请求头包含 API 鉴权和数据格式信息，是调用 API 的必要参数。
        load_dotenv()
        jina_api_key = os.getenv("JINA_API_KEY")    
        headers = {'Content-Type': 'application/json',
                   'Authorization': f'Bearer {jina_api_key}'}
        #Content-Type：指定请求体的数据格式为 JSON，这是 API 要求的格式
        #Authorization：API 鉴权字段，格式为 Bearer + 密钥，是调用 Jina API 的身份凭证。
        return headers
    
    def rerank(self, query, documents, top_n = 10):
        data = {
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "top_n": top_n,
            "documents": documents
        }

        response = requests.post(url=self.url, headers=self.headers, json=data)

        return response.json()

class LLMReranker:
    def __init__(self):
        self.llm = self.set_up_llm()
        self.system_prompt_rerank_single_block = prompts.RerankingPrompt.system_prompt_rerank_single_block
        self.system_prompt_rerank_multiple_blocks = prompts.RerankingPrompt.system_prompt_rerank_multiple_blocks
        self.schema_for_single_block = prompts.RetrievalRankingSingleBlock
        self.schema_for_multiple_blocks = prompts.RetrievalRankingMultipleBlocks
      #schema_for_single_block/schema_for_multiple_blocks：是预定义的 Pydantic 模型（或 JSON Schema），用于约束 LLM 返回的格式（比如强制返回 relevance_score 字段），确保输出结构化、可解析

    def set_up_llm(self):
        load_dotenv()
        llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return llm
    
    def get_rank_for_single_block(self, query, retrieved_document):
        user_prompt = f'/nHere is the query:/n"{query}"/n/nHere is the retrieved text block:/n"""/n{retrieved_document}/n"""/n'
        
        completion = self.llm.beta.chat.completions.parse(
            model="gpt-4o-mini-2024-07-18",
            temperature=0,
            messages=[
                {"role": "system", "content": self.system_prompt_rerank_single_block},
                {"role": "user", "content": user_prompt},
            ],
            response_format=self.schema_for_single_block
            #response_format：OpenAI 的 parse 方法是结构化输出功能，能强制 LLM 返回符合 schema_for_single_block 定义的格式（比如必须包含 relevance_score 字段，且值为 0-1 之间的浮点数）
        )

        response = completion.choices[0].message.parsed
        response_dict = response.model_dump()# 转为字典格式，方便后续处理
        
        return response_dict

    def get_rank_for_multiple_blocks(self, query, retrieved_documents):
        formatted_blocks = "\n\n---\n\n".join([f'Block {i+1}:\n\n"""\n{text}\n"""' for i, text in enumerate(retrieved_documents)])
        user_prompt = (
            f"Here is the query: \"{query}\"\n\n"
            "Here are the retrieved text blocks:\n"
            f"{formatted_blocks}\n\n"
            f"You should provide exactly {len(retrieved_documents)} rankings, in order."
        )

        completion = self.llm.beta.chat.completions.parse(
            model="gpt-4o-mini-2024-07-18",
            temperature=0,
            messages=[
                {"role": "system", "content": self.system_prompt_rerank_multiple_blocks},
                {"role": "user", "content": user_prompt},
            ],
            response_format=self.schema_for_multiple_blocks
        )

        response = completion.choices[0].message.parsed
        response_dict = response.model_dump()
      
        return response_dict

    def rerank_documents(self, query: str, documents: list, documents_batch_size: int = 4, llm_weight: float = 0.7):
        """
        Rerank multiple documents using parallel processing with threading.
        Combines vector similarity and LLM relevance scores using weighted average.
        """
        # Create batches of documents
        doc_batches = [documents[i:i + documents_batch_size] for i in range(0, len(documents), documents_batch_size)]
        vector_weight = 1 - llm_weight
        
        if documents_batch_size == 1:# 如果批次大小为1，走单文档处理逻辑
            def process_single_doc(doc):
                # Get ranking for single document
                ranking = self.get_rank_for_single_block(query, doc['text'])
                
                doc_with_score = doc.copy()
                doc_with_score["relevance_score"] = ranking["relevance_score"]
                # 计算综合分数：加权平均（LLM分数*0.7 + 向量距离*0.3）
                # 注：向量 distance 越小表示越相似，这里直接用 distance 参与计算（代码注释已说明）
                # Calculate combined score - note that distance is inverted since lower is better
                doc_with_score["combined_score"] = round(
                    llm_weight * ranking["relevance_score"] + 
                    vector_weight * doc['distance'],
                    4
                )
                return doc_with_score

            # Process all documents in parallel using single-block method
            with ThreadPoolExecutor() as executor:
                all_results = list(executor.map(process_single_doc, documents))
                #ThreadPoolExecutor()：Python 内置的线程池管理器，自动管理线程的创建 / 销毁，避免手动线程操作的繁琐
                #executor.map(process_single_doc, documents)：把 documents 列表中的每个文档，分配给线程池中的线程，并行执行 process_single_doc 函数
                #list(...)：将线程池的执行结果（迭代器）转为列表，all_results 是所有文档打完分后的结果列表
                '''
                executor.map(func, *iterables, timeout=None, chunksize=1)
                本质：把 iterables（可迭代对象，比如列表）中的元素，分配给线程池 / 进程池中的线程 / 进程，并行执行 func 函数；
                chunksize：仅对 ProcessPoolExecutor 有效，控制每个进程一次性取多少个任务
                '''
        else:
            def process_batch(batch):
                texts = [doc['text'] for doc in batch]
                rankings = self.get_rank_for_multiple_blocks(query, texts)
                results = []
                block_rankings = rankings.get('block_rankings', [])
                
                if len(block_rankings) < len(batch):
                    print(f"\nWarning: Expected {len(batch)} rankings but got {len(block_rankings)}")
                    for i in range(len(block_rankings), len(batch)):#生成「缺失打分的文档索引」
                        doc = batch[i]
                        print(f"Missing ranking for document on page {doc.get('page', 'unknown')}:")
                        print(f"Text preview: {doc['text'][:100]}...\n")
                        #doc.get('page', 'unknown')：获取文档的页码（如果有），没有就显示 unknown，方便定位文档来源
                        #doc['text'][:100]：截取文档前 100 个字符预览，避免超长文本刷屏，同时能看到文档内容
                    
                    for _ in range(len(batch) - len(block_rankings)):
                        block_rankings.append({
                            "relevance_score": 0.0, 
                            "reasoning": "Default ranking due to missing LLM response"
                        })
                        #给 block_rankings 列表追加默认打分
                        #reasoning：标注分数是「默认值」，不是 LLM 真实评估的，方便后续分析结果时区分
                
                for doc, rank in zip(batch, block_rankings):
                    doc_with_score = doc.copy()
                    doc_with_score["relevance_score"] = rank["relevance_score"]
                    doc_with_score["combined_score"] = round(
                        llm_weight * rank["relevance_score"] + 
                        vector_weight * doc['distance'],
                        4
                    )
                    results.append(doc_with_score)
                return results

            # Process batches in parallel using threads
            with ThreadPoolExecutor() as executor:
                batch_results = list(executor.map(process_batch, doc_batches))
            
            # Flatten results
            all_results = []
            for batch in batch_results:
                all_results.extend(batch)
        
        # Sort results by combined score in descending order
        all_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return all_results
