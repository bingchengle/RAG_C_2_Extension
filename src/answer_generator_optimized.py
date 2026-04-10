"""
Optimized answer generation with multi-round verification and citation validation
"""
import json
from typing import Any, List, Dict, Optional, Tuple
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv
import os
import re

from src.route_config import RouteUsageStats


@dataclass
class AnswerValidationResult:
    """See implementation."""
    is_valid: bool
    issues: List[str]
    confidence: float
    suggestions: List[str]


@dataclass
class GeneratedAnswer:
    """See implementation."""
    answer: str
    reasoning: str
    citations: List[Dict]
    confidence: float
    validation_passed: bool


def _normalize_answer_text_for_verify(answer_text) -> str:
    """Structured outputs may return dict/list for answer; citation check needs a string."""
    if answer_text is None:
        return ""
    if isinstance(answer_text, str):
        return answer_text
    if isinstance(answer_text, (int, float, bool)):
        return str(answer_text)
    if isinstance(answer_text, dict):
        return json.dumps(answer_text, ensure_ascii=False)
    if isinstance(answer_text, list):
        return json.dumps(answer_text, ensure_ascii=False)
    return str(answer_text)


class AnswerGeneratorOptimized:
    """See implementation."""

    def __init__(
        self,
        model: str = "gpt-4o-2024-08-06",
        verification_model: str = "gpt-4o-mini-2024-07-18",
        temperature: float = 0.3,
        max_verification_rounds: int = 2,
        domain: str = "general",
        usage_stats: Optional[RouteUsageStats] = None,
    ):
        """See implementation."""
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)
        self.model = model
        self.verification_model = verification_model
        self.temperature = temperature
        self.max_verification_rounds = max_verification_rounds
        self.domain = (domain or "general").lower()
        self.usage_stats = usage_stats

    def _domain_instruction(self) -> str:
        if self.domain != "finance":
            return ""
        return (
            "\n金融问答额外要求：\n"
            "- 严格区分指标名称，避免将相近指标当作同一指标。\n"
            "- 明确时间口径（财年/季度/期末）和单位（元/千元/百万元/%）。\n"
            "- 若问题明确指定了币种，且上下文仅出现其他币种，返回\"信息不足\"。\n"
            "- 若问题未指定币种，可在同一币种内比较并明确说明币种差异风险。\n"
            "- 若上下文没有直接给出目标指标，返回\"信息不足\"。\n"
            "- 不做推导计算，不用外部知识补全。"
        )

    def _schema_instruction(self, schema: str) -> str:
        """See implementation."""
        s = (schema or "text").lower()
        if s == "boolean":
            return (
                "\n【本题类型：判断题】\n"
                "- 最终答案 answer 必须是 JSON 布尔：true 或 false。\n"
                "- 对「是否提及 / 是否报告 / 是否披露 / 并购或收购 / 资本结构变化」等："
                "若上下文存在与问题实质相关的披露（含正文叙述、交易或债务/股权安排描述、脚注），应回答 true；"
                "仅当上下文确实不包含此类信息、或问题所指的窄定义在上下文中明确不成立时再答 false。\n"
                "- 不要仅因没有与问题用词完全相同的章节标题就答 false。\n"
            )
        if s == "number":
            return (
                "\n【本题类型：数值题】\n"
                "- answer 优先给出单一数字；若只能区间或近似，在 reasoning 说明并在 answer 给出最接近的可解析数字或\"信息不足\"。\n"
            )
        if s in ("name", "names"):
            return (
                "\n【本题类型：名称题】\n"
                "- answer 使用与公司名或题干要求一致的简短实体，避免冗长解释。\n"
            )
        return ""

    def generate_answer(
        self,
        query: str,
        context: List[Dict],
        schema: str = "text"
    ) -> GeneratedAnswer:
        """See implementation."""

        initial_answer = self._generate_initial_answer(query, context, schema)


        current_answer = initial_answer
        for round_num in range(self.max_verification_rounds):

            validation = self._validate_answer(
                query, current_answer, context
            )

            if validation.is_valid:
                break


            if round_num < self.max_verification_rounds - 1:
                current_answer = self._rewrite_answer(
                    query, current_answer, context, validation, schema
                )


        validated_citations = self._validate_citations(
            current_answer, context
        )

        return GeneratedAnswer(
            answer=_normalize_answer_text_for_verify(current_answer.get("answer")),
            reasoning=current_answer['reasoning'],
            citations=validated_citations,
            confidence=validation.confidence if 'validation' in locals() else 0.8,
            validation_passed=validation.is_valid if 'validation' in locals() else True
        )

    def _generate_initial_answer(
        self,
        query: str,
        context: List[Dict],
        schema: str
    ) -> Dict:
        """See implementation."""

        context_text = self._format_context(context)

        system_prompt = """你是一个专业的问答助手。基于提供的上下文回答问题。
要求：
1. 答案必须基于上下文，不要引入外部知识
2. 提供清晰的推理过程
3. 标注信息来源（页码）
4. 如果上下文不足以回答问题，明确说明"信息不足"
5. 对于数字类问题，确保数值准确""" + self._domain_instruction() + self._schema_instruction(schema)

        user_prompt = f"""上下文：
{context_text}

问题：{query}

请以JSON格式回答：
{{
    "reasoning": "逐步推理过程",
    "answer": "最终答案",
    "citations": ["页码1", "页码2"]
}}"""

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        if self.usage_stats:
            self.usage_stats.record_answer_generation()

        result = json.loads(response.choices[0].message.content)
        return result

    def _validate_answer(
        self,
        query: str,
        answer: Dict,
        context: List[Dict]
    ) -> AnswerValidationResult:
        """See implementation."""
        system_prompt = """你是一个答案质量评估专家。评估答案是否符合以下标准：
1. 准确性：答案是否与上下文一致
2. 完整性：是否回答了问题的所有部分
3. 引用准确性：引用的页码是否包含相关信息
4. 逻辑性：推理过程是否合理

输出JSON格式：
{
    "is_valid": true/false,
    "issues": ["问题1", "问题2"],
    "confidence": 0.0-1.0,
    "suggestions": ["改进建议1", "改进建议2"]
}""" + self._domain_instruction()

        user_prompt = f"""问题：{query}

答案：{_normalize_answer_text_for_verify(answer.get("answer"))}

推理过程：{answer['reasoning']}

引用页码：{answer.get('citations', [])}

上下文摘要：
{self._format_context_summary(context)}

请评估答案质量。"""

        response = self.client.chat.completions.create(
            model=self.verification_model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        if self.usage_stats:
            self.usage_stats.record_verification()

        result = json.loads(response.choices[0].message.content)
        return AnswerValidationResult(
            is_valid=result.get('is_valid', True),
            issues=result.get('issues', []),
            confidence=result.get('confidence', 0.8),
            suggestions=result.get('suggestions', [])
        )

    def _rewrite_answer(
        self,
        query: str,
        current_answer: Dict,
        context: List[Dict],
        validation: AnswerValidationResult,
        schema: str = "text",
    ) -> Dict:
        """See implementation."""
        system_prompt = """你是一个答案重写专家。根据验证反馈改进答案。
要求：
1. 解决所有指出的问题
2. 确保答案基于上下文
3. 保持推理过程的清晰性
4. 准确标注引用来源""" + self._domain_instruction() + self._schema_instruction(schema)

        user_prompt = f"""问题：{query}

当前答案：{_normalize_answer_text_for_verify(current_answer.get("answer"))}

验证反馈：
- 问题：{validation.issues}
- 建议：{validation.suggestions}

上下文：
{self._format_context(context)}

请重写答案，解决上述问题。以JSON格式输出：
{{
    "reasoning": "改进后的推理过程",
    "answer": "改进后的答案",
    "citations": ["页码1", "页码2"]
}}"""

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        if self.usage_stats:
            self.usage_stats.record_answer_rewrite()

        return json.loads(response.choices[0].message.content)

    def _validate_citations(
        self,
        answer: Dict,
        context: List[Dict]
    ) -> List[Dict]:
        """See implementation."""
        citations = answer.get('citations', [])
        validated_citations = []

        for citation in citations:

            page_num = self._extract_page_number(citation)
            if page_num is None:
                continue


            page_context = self._find_page_context(page_num, context)
            if page_context is None:
                continue


            is_valid = self._verify_citation_content(
                answer.get("answer"), page_context
            )

            validated_citations.append({
                'page': page_num,
                'is_valid': is_valid,
                'context_snippet': page_context[:200] if page_context else ""
            })

        return validated_citations

    def _extract_page_number(self, citation) -> Optional[int]:
        """See implementation."""
        if isinstance(citation, int):
            return citation
        if isinstance(citation, str):

            numbers = re.findall(r'\d+', citation)
            if numbers:
                return int(numbers[0])
        return None

    def _find_page_context(self, page_num: int, context: List[Dict]) -> Optional[str]:
        """See implementation."""
        for doc in context:
            if doc.get('page') == page_num:
                return doc.get('text', '')
        return None

    def _verify_citation_content(
        self,
        answer_text: Any,
        page_context: str
    ) -> bool:
        """See implementation."""
        answer_text = _normalize_answer_text_for_verify(answer_text)

        key_phrases = self._extract_key_phrases(answer_text)


        for phrase in key_phrases:
            if phrase.lower() in page_context.lower():
                return True

        return False

    def _extract_key_phrases(self, text: str) -> List[str]:
        """See implementation."""


        phrases = []


        numbers = re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', text)
        phrases.extend(numbers)


        quotes = re.findall(r'"([^"]+)"', text)
        phrases.extend(quotes)

        return phrases

    def _format_context(self, context: List[Dict]) -> str:
        """See implementation."""
        formatted = []
        for i, doc in enumerate(context, 1):
            formatted.append(f"【文档{i}】页码：{doc.get('page', 'N/A')}\n{doc.get('text', '')}")
        return "\n\n".join(formatted)

    def _format_context_summary(self, context: List[Dict]) -> str:
        """See implementation."""
        summaries = []
        for doc in context:
            text = doc.get('text', '')
            summary = text[:200] + "..." if len(text) > 200 else text
            summaries.append(f"页码{doc.get('page', 'N/A')}：{summary}")
        return "\n".join(summaries)


class SelfConsistencyChecker:
    """See implementation."""

    def __init__(self, generator: AnswerGeneratorOptimized, num_samples: int = 3):
        """See implementation."""
        self.generator = generator
        self.num_samples = num_samples

    def generate_with_self_consistency(
        self,
        query: str,
        context: List[Dict]
    ) -> GeneratedAnswer:
        """See implementation."""

        answers = []
        for _ in range(self.num_samples):
            answer = self.generator.generate_answer(query, context)
            answers.append(answer)


        best_answer = self._select_most_consistent(answers)

        return best_answer

    def _select_most_consistent(
        self,
        answers: List[GeneratedAnswer]
    ) -> GeneratedAnswer:
        """See implementation."""


        return max(answers, key=lambda x: x.confidence)



def generate_answer_optimized(
    query: str,
    context: List[Dict],
    use_verification: bool = True,
    use_self_consistency: bool = False
) -> Dict:
    """See implementation."""
    generator = AnswerGeneratorOptimized()

    if use_self_consistency:
        checker = SelfConsistencyChecker(generator)
        result = checker.generate_with_self_consistency(query, context)
    else:
        result = generator.generate_answer(query, context)

    return {
        'answer': result.answer,
        'reasoning': result.reasoning,
        'citations': result.citations,
        'confidence': result.confidence,
        'validation_passed': result.validation_passed
    }
