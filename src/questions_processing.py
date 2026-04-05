import json
from typing import Union, Dict, List, Optional, Any
import re
from pathlib import Path
from src.retrieval import VectorRetriever, HybridRetriever
from src.api_requests import APIProcessor
from src.answer_generator_optimized import AnswerGeneratorOptimized
from tqdm import tqdm
import pandas as pd
import threading
import concurrent.futures

FINANCE_METRIC_SYNONYMS = {
    "revenue": ["sales", "turnover", "total revenue", "net sales"],
    "operating margin": ["operating profit margin", "operating income margin", "ebit margin"],
    "net income": ["net profit", "profit attributable", "profit for the year"],
    "total assets": ["assets", "asset base"],
    "shareholders' equity": ["stockholders' equity", "total equity", "net assets", "equity attributable to owners"],
    "eps": ["earnings per share", "diluted eps", "basic eps"],
    "cash flow": ["operating cash flow", "cash generated from operations", "cfo"],
}

FINANCE_CURRENCY_PATTERNS = {
    "USD": [r"\busd\b", r"\bus\$?\b", r"\$"],
    "EUR": [r"\beur\b", r"\beuro\b", r"€"],
    "GBP": [r"\bgbp\b", r"\bpound\b", r"£"],
    "CNY": [r"\bcny\b", r"\brmb\b", r"\byuan\b", r"人民币", r"元"],
    "JPY": [r"\bjpy\b", r"\byen\b", r"¥"],
}


class QuestionsProcessor:
    def __init__(
        self,
        vector_db_dir: Union[str, Path] = './vector_dbs',
        bm25_db_dir: Union[str, Path] = './bm25_dbs',
        documents_dir: Union[str, Path] = './documents',
        questions_file_path: Optional[Union[str, Path]] = None,
        new_challenge_pipeline: bool = False,
            #new_challenge_pipeline 是「严格问答流程」的开关，启用后系统会执行「页码验证 + 标准化参考文献生成」
        subset_path: Optional[Union[str, Path]] = None,
        parent_document_retrieval: bool = False,
        llm_reranking: bool = False,
        llm_reranking_sample_size: int = 20,
        top_n_retrieval: int = 10,
        parallel_requests: int = 10,
        api_provider: str = "openai",
        answering_model: str = "gpt-4o-2024-08-06",
        full_context: bool = False,
        use_bge: bool = False,
        embedding_provider: str = "openai",
        embedding_model: Optional[str] = None,
        reranker_type: str = "llm",
        domain: str = "general",
        finance_metric_expansion: bool = True,
        finance_normalize_numeric: bool = True,
        finance_currency_consistency_check: bool = True,
        finance_unit_conversion: bool = True,
        enable_query_rewrite: bool = False,
        enable_similarity_check: bool = True,
        enable_multi_turn: bool = False,
        conversation_max_turns: int = 6
            # Optional[...] 表示参数可以为 None（如 questions_file_path）
            #所有参数都加了类型注解（如 Union[str, Path] 表示支持字符串或 Path 对象）
    ):
        self.questions = self._load_questions(questions_file_path)
        self.documents_dir = Path(documents_dir)
        self.vector_db_dir = Path(vector_db_dir)
        self.bm25_db_dir = Path(bm25_db_dir)
        self.subset_path = Path(subset_path) if subset_path else None
        #路径标准化：将字符串路径转为 Path 对象（方便后续文件操作）

        
        self.new_challenge_pipeline = new_challenge_pipeline
        self.return_parent_pages = parent_document_retrieval
        self.llm_reranking = llm_reranking
        self.llm_reranking_sample_size = llm_reranking_sample_size
        self.top_n_retrieval = top_n_retrieval
        self.answering_model = answering_model
        self.parallel_requests = parallel_requests
        self.api_provider = api_provider
        self.use_bge = use_bge
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.reranker_type = reranker_type
        self.domain = (domain or "general").lower()
        self.finance_metric_expansion = finance_metric_expansion
        self.finance_normalize_numeric = finance_normalize_numeric
        self.finance_currency_consistency_check = finance_currency_consistency_check
        self.finance_unit_conversion = finance_unit_conversion
        self.enable_query_rewrite = enable_query_rewrite
        self.enable_similarity_check = enable_similarity_check
        self.enable_multi_turn = enable_multi_turn
        self.conversation_max_turns = max(1, conversation_max_turns)
        self.openai_processor = APIProcessor(provider=api_provider)
        #self.openai_processor 是一个封装好的 API 处理器，目的是屏蔽不同厂商（如 OpenAI、Anthropic）的 API 调用差异，让后续代码无需关心具体的 API 调用细节。
        self.answer_generator = AnswerGeneratorOptimized(domain=self.domain)
        self.full_context = full_context

        self.answer_details = []
        self.detail_counter = 0# 计数器：记录已处理的问答数量
        self._lock = threading.Lock()
        self._conversation_lock = threading.Lock()
        self._conversation_store: Dict[str, List[Dict[str, str]]] = {}
        #self._lock = threading.Lock() 是为了应对 parallel_requests 带来的多线程并发，防止 answer_details、detail_counter 等共享变量被多线程同时修改导致数据错乱

    @staticmethod
    def _is_na_like(value: Any) -> bool:
        if value is None:
            return True
        normalized = str(value).strip().lower()
        return normalized in {"n/a", "na", "信息不足", "insufficient information", "not available", ""}

    @staticmethod
    def _extract_numeric_value_for_comparison(value: Any) -> Optional[float]:
        if isinstance(value, (int, float)):
            return float(value)
        if not isinstance(value, str):
            return None
        text_raw = value.strip()
        text = text_raw.lower()
        if not text:
            return None
        # Avoid extracting years/numbers from long natural language sentences.
        numeric_like = re.fullmatch(
            r"[$€£¥￥]?\s*[-+]?\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|billion|million|thousand|bn|mn|k)?",
            text,
            flags=re.IGNORECASE,
        )
        if not numeric_like:
            return None
        match = re.search(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", text)
        if not match:
            return None
        raw = match.group(0).replace(",", "")
        try:
            number = float(raw)
        except ValueError:
            return None

        # Normalize common unit suffixes into comparable scalar values.
        if "billion" in text or re.search(r"\bbn\b", text):
            number *= 1_000_000_000.0
        elif "million" in text or re.search(r"\bmn\b", text):
            number *= 1_000_000.0
        elif "thousand" in text or re.search(r"\bk\b", text):
            number *= 1_000.0
        return number

    def _fallback_comparative_name_answer(self, question: str, individual_answers: Dict[str, dict]) -> Optional[str]:
        scored = []
        for company, answer in individual_answers.items():
            numeric_value = self._extract_numeric_value_for_comparison(answer.get("final_answer"))
            if numeric_value is None:
                continue
            scored.append((company, numeric_value))

        if not scored:
            return None

        question_lower = question.lower()
        prefer_min = any(token in question_lower for token in ["lower", "lowest", "smaller", "least", "minimum", "min "])
        prefer_max = any(token in question_lower for token in ["higher", "highest", "larger", "greater", "most", "maximum", "max "])
        if prefer_min:
            return min(scored, key=lambda x: x[1])[0]
        if prefer_max:
            return max(scored, key=lambda x: x[1])[0]

        # Default to maximum for "which company had ..." style comparisons.
        return max(scored, key=lambda x: x[1])[0]

    def _build_comparative_company_fallback_queries(self, original_question: str, company: str) -> List[str]:
        """Build deterministic per-company queries for comparative retries."""
        queries: List[str] = []
        question = (original_question or "").strip()
        if question:
            queries.append(question)

        metric_match = re.search(r"(?:higher|lower|highest|lowest|greater|smaller)\s+(.+?):", question, re.IGNORECASE)
        metric = metric_match.group(1).strip() if metric_match else ""

        period_match = re.search(r"(in\s+.+?)\??$", question, re.IGNORECASE)
        period = period_match.group(1).strip() if period_match else ""

        if metric:
            candidate = f'What was the {metric} of "{company}"'
            if period:
                candidate = f"{candidate} {period}"
            queries.append(candidate.rstrip(" ?") + "?")

            metric_lower = metric.lower()
            alias_terms: List[str] = []
            for canonical, aliases in FINANCE_METRIC_SYNONYMS.items():
                candidates = [canonical] + aliases
                if canonical in metric_lower or any(alias in metric_lower for alias in aliases):
                    alias_terms = candidates[:5]
                    break
            if metric not in alias_terms:
                alias_terms = [metric] + alias_terms

            # Add apostrophe-free variant to improve retrieval hit rate.
            alias_terms.extend([term.replace("'", "") for term in alias_terms if "'" in term])

            for alias_metric in alias_terms:
                alias_metric = alias_metric.strip()
                if not alias_metric:
                    continue
                alias_query = f'What was the {alias_metric} of "{company}"'
                if period:
                    alias_query = f"{alias_query} {period}"
                queries.append(alias_query.rstrip(" ?") + "?")

        unique_queries: List[str] = []
        seen = set()
        for item in queries:
            key = item.strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique_queries.append(item)
        return unique_queries

    def _choose_better_numeric_company_answer(self, candidate: dict, current: Optional[dict]) -> dict:
        if current is None:
            return candidate
        cand_na = self._is_na_like(candidate.get("final_answer"))
        curr_na = self._is_na_like(current.get("final_answer"))
        if curr_na and not cand_na:
            return candidate
        if cand_na and not curr_na:
            return current
        cand_refs = len(candidate.get("references", []) or [])
        curr_refs = len(current.get("references", []) or [])
        return candidate if cand_refs > curr_refs else current

    def _load_questions(self, questions_file_path: Optional[Union[str, Path]]) -> List[Dict[str, str]]:
        if questions_file_path is None:
            return []
        with open(questions_file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def _format_retrieval_results(self, retrieval_results) -> str:
        """Format vector retrieval results into RAG context string"""
        if not retrieval_results:
            return ""
        
        context_parts = []
        for result in retrieval_results:
            page_number = result['page']
            text = result['text']
            context_parts.append(f'Text retrieved from page {page_number}: \n"""\n{text}\n"""')
            
        return "\n\n---\n\n".join(context_parts)

    def _extract_references(self, pages_list: list, company_name: str) -> list:
        # Load companies data
        if self.subset_path is None:
            raise ValueError("subset_path is required for new challenge pipeline when processing references.")
        self.companies_df = pd.read_csv(self.subset_path)

        # Find the company's SHA1 from the subset CSV
        #pdf_sha1：是 PDF 文档的「唯一身份证」（不管文件名怎么改，SHA1 值不变），用来精准定位是哪个文档；
        matching_rows = self.companies_df[self.companies_df['company_name'] == company_name]
        if matching_rows.empty:
            company_sha1 = ""
        else:
            company_sha1 = matching_rows.iloc[0]['sha1']
            # 为什么用iloc[0]？防止CSV里有重名公司（比如两个「腾讯」），只取第一个，保证结果唯一。

        refs = []
        for page in pages_list:
            refs.append({"pdf_sha1": company_sha1, "page_index": page})
        return refs

    def _validate_page_references(self, claimed_pages: list, retrieval_results: list, min_pages: int = 2, max_pages: int = 8) -> list:
        """
        Validate that all page numbers mentioned in the LLM's answer are actually from the retrieval results.
        If fewer than min_pages valid references remain, add top pages from retrieval results.
        """
        if claimed_pages is None:#边界处理：若 LLM 未返回任何声称的页码，初始化为空列表
            claimed_pages = []
        
        retrieved_pages = [result['page'] for result in retrieval_results]
        
        validated_pages = [page for page in claimed_pages if page in retrieved_pages]
        
        if len(validated_pages) < len(claimed_pages):#提示幻觉：若有页码被过滤，打印警告（暴露 LLM 编造的页码）
            removed_pages = set(claimed_pages) - set(validated_pages)
            print(f"Warning: Removed {len(removed_pages)} hallucinated page references: {removed_pages}")
        
        if len(validated_pages) < min_pages and retrieval_results:#补充不足：若有效页码少于 min_pages，从检索结果中补充 Top-N 页码
            existing_pages = set(validated_pages)
            #转化为集合的原因：1、集合遍历速度更快；2、集合会自动去重，避免页码重复；3、综合以上两点最后同字典append做比对
            
            for result in retrieval_results:
                page = result['page']
                if page not in existing_pages:
                    validated_pages.append(page)
                    existing_pages.add(page)
                    
                    if len(validated_pages) >= min_pages:
                        break# 达到最小值则停止补充
        
        if len(validated_pages) > max_pages:#截断过多：若有效页码超过 max_pages，截断到最大值
            print(f"Trimming references from {len(validated_pages)} to {max_pages} pages")
            validated_pages = validated_pages[:max_pages]
        
        return validated_pages

    def _stringify_history(self, history: List[Dict[str, str]]) -> str:
        if not history:
            return "No previous turns."
        lines = []
        for turn in history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _get_conversation_history(
        self,
        conversation_id: Optional[str],
        provided_history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        if provided_history:
            return provided_history
        if not self.enable_multi_turn or not conversation_id:
            return []
        with self._conversation_lock:
            return list(self._conversation_store.get(conversation_id, []))

    def _store_conversation_turn(
        self,
        conversation_id: Optional[str],
        question: str,
        answer: str
    ) -> None:
        if not self.enable_multi_turn or not conversation_id:
            return
        with self._conversation_lock:
            history = self._conversation_store.get(conversation_id, [])
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})
            max_messages = self.conversation_max_turns * 2
            if len(history) > max_messages:
                history = history[-max_messages:]
            self._conversation_store[conversation_id] = history

    def _rewrite_query_for_retrieval(
        self,
        question: str,
        company_name: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        if not self.enable_query_rewrite and not conversation_history:
            return question
        history_text = self._stringify_history(conversation_history or [])
        system_prompt = (
            "You rewrite user questions for document retrieval. "
            "Preserve entities, timeframe, units, and metric intent. "
            "Output one single rewritten query only."
        )
        if self.domain == "finance":
            system_prompt += (
                " Finance mode: keep fiscal period, currency, unit scale (thousand/million), "
                "and exact metric wording; avoid replacing a metric with a related proxy."
            )
        user_prompt = (
            f"Company: {company_name}\n"
            f"Conversation history:\n{history_text}\n\n"
            f"Latest user question:\n{question}\n\n"
            "Return only the rewritten retrieval query."
        )
        try:
            rewritten = self.openai_processor.send_message(
                model=self.answering_model,
                temperature=0,
                system_content=system_prompt,
                human_content=user_prompt,
                is_structured=False
            )
            if not isinstance(rewritten, str):
                return question
            rewritten = rewritten.strip().strip('"').strip()
            rewritten = rewritten or question
            if self.domain == "finance" and self.finance_metric_expansion:
                rewritten = self._expand_finance_query_with_synonyms(rewritten)
            return rewritten
        except Exception:
            return question

    def _expand_finance_query_with_synonyms(self, query: str) -> str:
        """Append likely finance metric synonyms to improve recall."""
        lowered = query.lower()
        expansions = []
        for metric, aliases in FINANCE_METRIC_SYNONYMS.items():
            if metric in lowered or any(alias in lowered for alias in aliases):
                expansions.extend(alias for alias in aliases if alias not in lowered)
        if not expansions:
            return query
        unique_expansions = list(dict.fromkeys(expansions))[:6]
        return f"{query} | related terms: {', '.join(unique_expansions)}"

    def _normalize_finance_numeric_answer(self, answer: Any, schema: str) -> Any:
        """Normalize numeric answers in finance mode while preserving N/A."""
        if self.domain != "finance" or not self.finance_normalize_numeric or schema != "number":
            return answer
        if answer is None:
            return answer
        if isinstance(answer, (int, float)):
            return answer
        if isinstance(answer, str):
            stripped = answer.strip()
            if stripped.upper() == "N/A":
                return "N/A"
            normalized = stripped.replace(",", "")
            is_percent = normalized.endswith("%")
            if is_percent:
                normalized = normalized[:-1]
            scale = 1.0
            lowered = normalized.lower()
            if self.finance_unit_conversion:
                if "billion" in lowered or " bn" in lowered or lowered.endswith("b"):
                    scale = 1_000_000_000.0
                elif "million" in lowered or " mn" in lowered or lowered.endswith("m"):
                    scale = 1_000_000.0
                elif "thousand" in lowered or " k" in lowered or lowered.endswith("k"):
                    scale = 1_000.0
            normalized = re.sub(r"[a-zA-Z$€£¥￥ ]+", "", normalized)
            normalized = normalized.strip()
            if re.fullmatch(r"-?\d+", normalized):
                try:
                    value = int(normalized)
                    if scale != 1.0:
                        value = int(value * scale)
                    return value
                except ValueError:
                    return answer
            if re.fullmatch(r"-?\d+(?:\.\d+)?", normalized):
                try:
                    value = float(normalized)
                    if scale != 1.0:
                        value = value * scale
                    if is_percent:
                        return value
                    return int(value) if value.is_integer() else value
                except ValueError:
                    return answer
        return answer

    def _infer_question_currency(self, question: str) -> Optional[str]:
        lowered = question.lower()
        for currency, patterns in FINANCE_CURRENCY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, lowered, re.IGNORECASE):
                    return currency
        return None

    def _detect_context_currencies(self, retrieval_results: List[Dict[str, Any]]) -> List[str]:
        detected = set()
        corpus = "\n".join(item.get("text", "")[:2000] for item in retrieval_results[:8])
        for currency, patterns in FINANCE_CURRENCY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, corpus, re.IGNORECASE):
                    detected.add(currency)
                    break
        return sorted(detected)

    def _apply_finance_currency_consistency(
        self,
        answer: Any,
        question: str,
        retrieval_results: List[Dict[str, Any]],
        schema: str
    ) -> tuple[Any, Optional[Dict[str, Any]]]:
        if self.domain != "finance" or not self.finance_currency_consistency_check or schema != "number":
            return answer, None
        question_currency = self._infer_question_currency(question)
        context_currencies = self._detect_context_currencies(retrieval_results)
        check = {
            "question_currency": question_currency,
            "context_currencies": context_currencies,
            "currency_mismatch": False
        }
        if not question_currency:
            return answer, check
        if not context_currencies:
            return answer, check
        if question_currency not in context_currencies and len(context_currencies) > 0:
            check["currency_mismatch"] = True
            return "N/A", check
        return answer, check

    def _calculate_answer_similarity(self, answer_text: str, retrieval_results: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
        if not self.enable_similarity_check or not answer_text or not retrieval_results:
            return None
        try:
            context_text = "\n\n".join(item.get("text", "") for item in retrieval_results[:5] if item.get("text"))
            if not context_text:
                return None
            context_similarity = VectorRetriever.get_strings_cosine_similarity(answer_text, context_text)

            chunk_scores = []
            for item in retrieval_results[:5]:
                text = item.get("text", "")
                if text:
                    score = VectorRetriever.get_strings_cosine_similarity(answer_text, text)
                    chunk_scores.append(score)

            best_chunk_similarity = max(chunk_scores) if chunk_scores else context_similarity
            return {
                "context_similarity": float(context_similarity),
                "best_chunk_similarity": float(best_chunk_similarity)
            }
        except Exception:
            return None

    def get_answer_for_company(
        self,
        company_name: str,
        question: str,
        schema: str,
        conversation_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        store_conversation_turn: bool = True
    ) -> dict:

        if self.llm_reranking:
            retriever = HybridRetriever(
                vector_db_dir=self.vector_db_dir,
                bm25_db_dir=self.bm25_db_dir,
                documents_dir=self.documents_dir,
                use_bge=self.use_bge,
                embedding_provider=self.embedding_provider,
                embedding_model=self.embedding_model,
                reranker_type=self.reranker_type,
                domain=self.domain
            )
        else:
            retriever = VectorRetriever(
                vector_db_dir=self.vector_db_dir,
                documents_dir=self.documents_dir,
                use_bge=self.use_bge,
                embedding_provider=self.embedding_provider,
                embedding_model=self.embedding_model
            )

        effective_history = self._get_conversation_history(conversation_id, conversation_history)
        retrieval_query = self._rewrite_query_for_retrieval(
            question=question,
            company_name=company_name,
            conversation_history=effective_history
        )

        if self.full_context:
            retrieval_results = retriever.retrieve_all(company_name)
            #full_context：开关，True 时返回公司所有文档（适合简单问题），False 时只返回和问题相关的 Top-N 片段（避免上下文过长，提升 LLM 回答效率）
        else:           
            retrieval_results = retriever.retrieve_by_company_name(
                company_name=company_name,
                query=retrieval_query,
                llm_reranking_sample_size=self.llm_reranking_sample_size,
                top_n=self.top_n_retrieval,
                return_parent_pages=self.return_parent_pages
            )
        
        if not retrieval_results:
            raise ValueError("No relevant context found")
        
        # 使用 AnswerGeneratorOptimized 生成答案
        answer_result = self.answer_generator.generate_answer(
            query=question,
            context=retrieval_results,
            schema=schema
        )
        similarity_scores = self._calculate_answer_similarity(
            answer_text=answer_result.answer,
            retrieval_results=retrieval_results
        )
        
        # 构建答案字典
        answer_dict = {
            "final_answer": answer_result.answer,
            "step_by_step_analysis": answer_result.reasoning,
            "reasoning_summary": answer_result.reasoning,
            "relevant_pages": [citation['page'] for citation in answer_result.citations if citation['is_valid']],
            "confidence": answer_result.confidence,
            "validation_passed": answer_result.validation_passed,
            "retrieval_query": retrieval_query,
            "answer_similarity": similarity_scores
        }
        answer_dict["final_answer"] = self._normalize_finance_numeric_answer(
            answer=answer_dict["final_answer"],
            schema=schema
        )
        checked_answer, finance_checks = self._apply_finance_currency_consistency(
            answer=answer_dict["final_answer"],
            question=question,
            retrieval_results=retrieval_results,
            schema=schema
        )
        answer_dict["final_answer"] = checked_answer
        answer_dict["finance_checks"] = finance_checks
        
        # 保存响应数据
        self.response_data = {"model": "gpt-4o-2024-08-06", "input_tokens": 0, "output_tokens": 0}  # 简化处理
        
        if self.new_challenge_pipeline:
            pages = answer_dict.get("relevant_pages", [])
            validated_pages = self._validate_page_references(pages, retrieval_results)
            answer_dict["relevant_pages"] = validated_pages
            answer_dict["references"] = self._extract_references(validated_pages, company_name)
        if store_conversation_turn:
            self._store_conversation_turn(
                conversation_id=conversation_id,
                question=question,
                answer=answer_dict["final_answer"]
            )
        return answer_dict

    def _extract_companies_from_subset(self, question_text: str) -> list[str]:
        """Extract company names from a question by matching against companies in the subset file."""
        if not hasattr(self, 'companies_df'):
            #hasattr() 是 Python 内置的一个核心函数，专门用来检查一个对象是否拥有指定名称的属性或方法
            if self.subset_path is None:
                raise ValueError("subset_path must be provided to use subset extraction")
            self.companies_df = pd.read_csv(self.subset_path)
        
        found_companies = []
        company_names = sorted(self.companies_df['company_name'].unique(), key=len, reverse=True)
        
        for company in company_names:
            escaped_company = re.escape(company)
            #re.escape() 的作用:把公司名中的所有正则特殊字符转义成普通字符（在特殊字符前加 \），让正则把它们当成「普通文字」处理
            
            pattern = rf'{escaped_company}(?:\W|$)'
            '''
            \W：匹配「非单词字符」(除字母，下划线，数字外)
            $：匹配「字符串结尾」
            非捕获组（?: 表示不单独捕获这个组）：只匹配，不捕获
            '''
            
            if re.search(pattern, question_text, re.IGNORECASE):#忽略大小写，在问题文本中搜索该公司名
                found_companies.append(company)
                question_text = re.sub(pattern, '', question_text, flags=re.IGNORECASE)
                #从问题文本中移除已匹配的公司名（避免重复匹配，比如问题里多次提「阿里巴巴」只算一次）
                #代码里「按公司名长度降序遍历」的逻辑，从根源上避免了「短公司名匹配长公司名」的情况
        
        return found_companies

    def process_question(
        self,
        question: str,
        schema: str,
        conversation_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ):
        if self.new_challenge_pipeline:
            extracted_companies = self._extract_companies_from_subset(question)
        else:
            extracted_companies = re.findall(r'"([^"]*)"', question)
            #从问题里，把所有被 双引号 "" 包裹起来的内容 提取出来！
        
        if len(extracted_companies) == 0:
            raise ValueError("No company name found in the question.")
        
        if len(extracted_companies) == 1:
            company_name = extracted_companies[0]
            answer_dict = self.get_answer_for_company(
                company_name=company_name,
                question=question,
                schema=schema,
                conversation_id=conversation_id,
                conversation_history=conversation_history
            )
            return answer_dict
        else:
            return self.process_comparative_question(
                question=question,
                companies=extracted_companies,
                schema=schema,
                conversation_id=conversation_id,
                conversation_history=conversation_history
            )
    
    def _create_answer_detail_ref(self, answer_dict: dict, question_index: int) -> str:
        """Create a reference ID for answer details and store the details"""
        ref_id = f"#/answer_details/{question_index}"
        with self._lock:
            self.answer_details[question_index] = {
                "step_by_step_analysis": answer_dict.get('step_by_step_analysis'),
                "reasoning_summary": answer_dict.get('reasoning_summary'),
                "relevant_pages": answer_dict.get('relevant_pages'),
                "retrieval_query": answer_dict.get("retrieval_query"),
                "answer_similarity": answer_dict.get("answer_similarity"),
                "finance_checks": answer_dict.get("finance_checks"),
                "response_data": self.response_data,
                "self": ref_id
            }
        return ref_id

    def _calculate_statistics(self, processed_questions: List[dict], print_stats: bool = False) -> dict:#print_stats=False（默认）→ 不打印
        """Calculate statistics about processed questions."""
        total_questions = len(processed_questions)
        error_count = sum(1 for q in processed_questions if "error" in q)
        na_count = sum(
            1 for q in processed_questions
            if self._is_na_like(q.get("value") if "value" in q else q.get("answer"))
        )
        #没有答案、无法回答、信息不足、找不到相关内容
        #==“N/A”对应的是if和else两个条件
        success_count = total_questions - error_count - na_count
        if print_stats:
            print(f"\nFinal Processing Statistics:")
            print(f"Total questions: {total_questions}")
            print(f"Errors: {error_count} ({(error_count/total_questions)*100:.1f}%)")
            print(f"N/A answers: {na_count} ({(na_count/total_questions)*100:.1f}%)")
            print(f"Successfully answered: {success_count} ({(success_count/total_questions)*100:.1f}%)\n")
        
        return {
            "total_questions": total_questions,
            "error_count": error_count,
            "na_count": na_count,
            "success_count": success_count
        }

    def process_questions_list(self, questions_list: List[dict], output_path: str = None, submission_file: bool = False, team_email: str = "", submission_name: str = "", pipeline_details: str = "") -> dict:
        #submission_file: bool = False：可选：是否生成提交格式
        total_questions = len(questions_list)
        # Add index to each question so we know where to write the answer details
        questions_with_index = [{**q, "_question_index": i} for i, q in enumerate(questions_list)]
        #**q：字典解包：把字典 q 里的所有键值对，全部展开、原封不动搬过去
        #拼接一个新字典，加入index
        self.answer_details = [None] * total_questions  # Preallocate list for answer details
        processed_questions = []
        parallel_threads = self.parallel_requests

        if parallel_threads <= 1:
            for question_data in tqdm(questions_with_index, desc="Processing questions"):
                processed_question = self._process_single_question(question_data)
                processed_questions.append(processed_question)
                if output_path:
                    self._save_progress(processed_questions, output_path, submission_file=submission_file, team_email=team_email, submission_name=submission_name, pipeline_details=pipeline_details)
        else:
            with tqdm(total=total_questions, desc="Processing questions") as pbar:
                for i in range(0, total_questions, parallel_threads):
                    batch = questions_with_index[i : i + parallel_threads]
                    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_threads) as executor:
                        # executor.map will return results in the same order as the input list.
                        batch_results = list(executor.map(self._process_single_question, batch))
                    processed_questions.extend(batch_results)
                    
                    if output_path:
                        self._save_progress(processed_questions, output_path, submission_file=submission_file, team_email=team_email, submission_name=submission_name, pipeline_details=pipeline_details)
                    pbar.update(len(batch_results))
        
        statistics = self._calculate_statistics(processed_questions, print_stats = True)
        
        return {
            "questions": processed_questions,
            "answer_details": self.answer_details,
            "statistics": statistics
        }

    def _process_single_question(self, question_data: dict) -> dict:
        question_index = question_data.get("_question_index", 0)
        conversation_id = question_data.get("conversation_id")
        conversation_history = question_data.get("conversation_history") or question_data.get("history")
        
        if self.new_challenge_pipeline:
            question_text = question_data.get("text")
            schema = question_data.get("kind")
        else:
            question_text = question_data.get("question")
            schema = question_data.get("schema")
        try:
            answer_dict = self.process_question(
                question=question_text,
                schema=schema,
                conversation_id=conversation_id,
                conversation_history=conversation_history
            )
            
            if "error" in answer_dict:
                detail_ref = self._create_answer_detail_ref({
                    "step_by_step_analysis": None,
                    "reasoning_summary": None,
                    "relevant_pages": None
                }, question_index)
                if self.new_challenge_pipeline:
                    return {
                        "question_text": question_text,
                        "kind": schema,
                        "value": None,
                        "references": [],
                        "error": answer_dict["error"],
                        "answer_details": {"$ref": detail_ref}
                    }
                else:
                    return {
                        "question": question_text,
                        "schema": schema,
                        "answer": None,
                        "error": answer_dict["error"],
                        "answer_details": {"$ref": detail_ref},
                    }
            detail_ref = self._create_answer_detail_ref(answer_dict, question_index)
            if self.new_challenge_pipeline:
                result = {
                    "question_text": question_text,
                    "kind": schema,
                    "value": answer_dict.get("final_answer"),
                    "references": answer_dict.get("references", []),
                    "answer_details": {"$ref": detail_ref}
                }
                if conversation_id:
                    result["conversation_id"] = conversation_id
                return result
            else:
                result = {
                    "question": question_text,
                    "schema": schema,
                    "answer": answer_dict.get("final_answer"),
                    "answer_details": {"$ref": detail_ref},
                }
                if conversation_id:
                    result["conversation_id"] = conversation_id
                return result
        except Exception as err:
            return self._handle_processing_error(question_text, schema, err, question_index)

    def _handle_processing_error(self, question_text: str, schema: str, err: Exception, question_index: int) -> dict:
        """
        Handle errors during question processing.
        Log error details and return a dictionary containing error information.
        """
        import traceback#Python 自带的打印详细报错堆栈工具
        error_message = str(err)
        tb = traceback.format_exc()#拿到完整的错误调用链
        error_ref = f"#/answer_details/{question_index}"
        error_detail = {
            "error_traceback": tb,
            "self": error_ref
        }
        
        with self._lock:
            self.answer_details[question_index] = error_detail#把错误存到 answer_details
        
        print(f"Error encountered processing question: {question_text}")
        print(f"Error type: {type(err).__name__}")
        print(f"Error message: {error_message}")
        print(f"Full traceback:\n{tb}\n")
        
        if self.new_challenge_pipeline:
            return {
                "question_text": question_text,
                "kind": schema,
                "value": None,
                "references": [],
                "error": f"{type(err).__name__}: {error_message}",
                "answer_details": {"$ref": error_ref}
            }
        else:
            return {
                "question": question_text,
                "schema": schema,
                "answer": None,
                "error": f"{type(err).__name__}: {error_message}",
                "answer_details": {"$ref": error_ref},
            }

    def _post_process_submission_answers(self, processed_questions: List[dict]) -> List[dict]:
        """
        Post-process answers for submission format:
        1. Convert page indices from one-based to zero-based
        2. Clear references for N/A answers
        3. Format answers according to submission schema
        4. Include step_by_step_analysis from answer details
        """
        submission_answers = []
        
        for q in processed_questions:
            question_text = q.get("question_text") or q.get("question")
            kind = q.get("kind") or q.get("schema")
            value = "N/A" if "error" in q else (q.get("value") if "value" in q else q.get("answer"))
            references = q.get("references", [])
            
            answer_details_ref = q.get("answer_details", {}).get("$ref", "")#接着从上面拿到的值里，再取 "$ref" 这个 key
            step_by_step_analysis = None
            if answer_details_ref and answer_details_ref.startswith("#/answer_details/"):#必须是以 #/answer_details/ 开头
                try:
                    index = int(answer_details_ref.split("/")[-1])#以’/‘进行分割，并去最后一个值，在这个场景中最后一个值应该是数字
                    if 0 <= index < len(self.answer_details) and self.answer_details[index]:
                        step_by_step_analysis = self.answer_details[index].get("step_by_step_analysis")
                except (ValueError, IndexError):
                    pass
            
            # Clear references if value is N/A
            if self._is_na_like(value):
                value = "N/A"
                references = []
            else:
                # Convert page indices from one-based to zero-based (competition requires 0-based page indices, but for debugging it is easier to use 1-based)
                references = [
                    {
                        "pdf_sha1": ref["pdf_sha1"],
                        "page_index": ref["page_index"] - 1
                    }
                    for ref in references
                ]
            
            submission_answer = {
                "question_text": question_text,
                "kind": kind,
                "value": value,
                "references": references,
            }
            
            if step_by_step_analysis:
                submission_answer["reasoning_process"] = step_by_step_analysis
            
            submission_answers.append(submission_answer)
        
        return submission_answers

    def _save_progress(self, processed_questions: List[dict], output_path: Optional[str], submission_file: bool = False, team_email: str = "", submission_name: str = "", pipeline_details: str = ""):
        if output_path:
            statistics = self._calculate_statistics(processed_questions)
            
            # Prepare debug content
            result = {
                "questions": processed_questions,
                "answer_details": self.answer_details,
                "statistics": statistics
            }
            output_file = Path(output_path)
            debug_file = output_file.with_name(output_file.stem + "_debug" + output_file.suffix)
            with open(debug_file, 'w', encoding='utf-8') as file:
                json.dump(result, file, ensure_ascii=False, indent=2)
            
            if submission_file:
                # Post-process answers for submission
                submission_answers = self._post_process_submission_answers(processed_questions)
                submission = {
                    "answers": submission_answers,
                    "team_email": team_email,
                    "submission_name": submission_name,
                    "details": pipeline_details
                }
                with open(output_file, 'w', encoding='utf-8') as file:
                    json.dump(submission, file, ensure_ascii=False, indent=2)

    def process_all_questions(self, output_path: str = 'questions_with_answers.json', team_email: str = "79250515615@yandex.com", submission_name: str = "Ilia_Ris SO CoT + Parent Document Retrieval", submission_file: bool = False, pipeline_details: str = ""):
        result = self.process_questions_list(
            self.questions,
            output_path,
            submission_file=submission_file,
            team_email=team_email,
            submission_name=submission_name,
            pipeline_details=pipeline_details
        )
        return result

    def process_comparative_question(
        self,
        question: str,
        companies: List[str],
        schema: str,
        conversation_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> dict:
        """
        Process a question involving multiple companies in parallel:
        1. Rephrase the comparative question into individual questions
        2. Process each individual question using parallel threads
        3. Combine results into final comparative answer
        """
        # Step 1: Rephrase the comparative question
        rephrased_questions = self.openai_processor.get_rephrased_questions(
            original_question=question,
            companies=companies
        )
        
        individual_answers = {}#创建空字典：用来存每个公司的单独答案
        aggregated_references = []#创建空列表：用来汇总所有公司的引用页码
        
        # Step 2: Process each individual question in parallel
        def process_company_question(company: str) -> tuple[str, dict]:
            """Helper function to process one company's question and return (company, answer)"""
            sub_question = rephrased_questions.get(company)
            if not sub_question:
                raise ValueError(f"Could not generate sub-question for company: {company}")

            best_answer: Optional[dict] = None
            retry_queries = [sub_question] + self._build_comparative_company_fallback_queries(question, company)
            for candidate_query in retry_queries:
                try:
                    answer_dict = self.get_answer_for_company(
                        company_name=company,
                        question=candidate_query,
                        schema="number",
                        conversation_id=conversation_id,
                        conversation_history=conversation_history,
                        store_conversation_turn=False
                    )
                    best_answer = self._choose_better_numeric_company_answer(answer_dict, best_answer)
                    if best_answer and not self._is_na_like(best_answer.get("final_answer")):
                        break
                except Exception:
                    continue

            if best_answer is None:
                best_answer = {
                    "final_answer": "N/A",
                    "step_by_step_analysis": f"No report found for company '{company}'.",
                    "reasoning_summary": f"No report found for company '{company}'.",
                    "references": [],
                    "relevant_pages": [],
                }
            return company, best_answer

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_company = {
                executor.submit(process_company_question, company): company 
                for company in companies
            }
            
            for future in concurrent.futures.as_completed(future_to_company):
                #future_to_company：表示的是{任务名：公司}
                #as_completed(...)：完成一个任务就返回一个
                try:
                    company, answer_dict = future.result()
                    individual_answers[company] = answer_dict
                    
                    company_references = answer_dict.get("references", [])
                    aggregated_references.extend(company_references)
                except Exception as e:
                    company = future_to_company[future]
                    print(f"Error processing company {company}: {str(e)}")
                    individual_answers[company] = {
                        "final_answer": "N/A",
                        "step_by_step_analysis": f"Error processing company {company}: {str(e)}",
                        "reasoning_summary": f"Error processing company {company}: {str(e)}",
                        "references": [],
                        "relevant_pages": [],
                    }
        
        # Remove duplicate references
        unique_refs = {}
        for ref in aggregated_references:
            key = (ref.get("pdf_sha1"), ref.get("page_index"))
            unique_refs[key] = ref
        aggregated_references = list(unique_refs.values())
        
        # Step 3: Get the comparative answer using all individual answers
        # 构建比较性问题的上下文
        comparative_context = []
        for company, answer in individual_answers.items():
            company_context = {
                "text": f"Company: {company}\nAnswer: {answer.get('final_answer')}\nReasoning: {answer.get('step_by_step_analysis')}",
                "page": 1  # 虚拟页码，实际不会使用
            }
            comparative_context.append(company_context)
        
        # 使用 AnswerGeneratorOptimized 生成比较性答案
        comparative_result = self.answer_generator.generate_answer(
            query=question,
            context=comparative_context,
            schema="comparative"
        )

        # Use deterministic numeric ranking as the primary decision for name comparisons.
        # This avoids model drift when one company is missing or cross-currency language appears.
        comparative_answer_value = comparative_result.answer
        if schema == "name":
            fallback_company = self._fallback_comparative_name_answer(question, individual_answers)
            if fallback_company:
                comparative_answer_value = fallback_company
            else:
                valid_companies = [
                    company for company, answer in individual_answers.items()
                    if not self._is_na_like(answer.get("final_answer"))
                ]
                if comparative_answer_value not in valid_companies:
                    if len(valid_companies) == 1:
                        comparative_answer_value = valid_companies[0]
                    elif len(valid_companies) == 0:
                        comparative_answer_value = "N/A"
        
        # 构建比较性答案字典
        comparative_answer = {
            "final_answer": comparative_answer_value,
            "step_by_step_analysis": comparative_result.reasoning,
            "reasoning_summary": comparative_result.reasoning,
            "relevant_pages": [citation['page'] for citation in comparative_result.citations if citation['is_valid']],
            "confidence": comparative_result.confidence,
            "validation_passed": comparative_result.validation_passed,
            "retrieval_query": question,
            "answer_similarity": None
        }
        
        # 保存响应数据
        self.response_data = {"model": "gpt-4o-2024-08-06", "input_tokens": 0, "output_tokens": 0}  # 简化处理
        
        comparative_answer["references"] = aggregated_references
        self._store_conversation_turn(
            conversation_id=conversation_id,
            question=question,
            answer=comparative_answer["final_answer"]
        )
        return comparative_answer
    