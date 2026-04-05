"""
Optimized answer generation with multi-round verification and citation validation
"""
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv
import os
import re


@dataclass
class AnswerValidationResult:
    """答案验证结果"""
    is_valid: bool
    issues: List[str]
    confidence: float
    suggestions: List[str]


@dataclass
class GeneratedAnswer:
    """生成的答案"""
    answer: str
    reasoning: str
    citations: List[Dict]
    confidence: float
    validation_passed: bool


class AnswerGeneratorOptimized:
    """优化的答案生成器，支持多轮验证和引用验证"""
    
    def __init__(
        self,
        model: str = "gpt-4o-2024-08-06",
        verification_model: str = "gpt-4o-mini-2024-07-18",
        temperature: float = 0.3,
        max_verification_rounds: int = 2
    ):
        """
        初始化答案生成器
        
        Args:
            model: 主生成模型
            verification_model: 验证模型
            temperature: 生成温度
            max_verification_rounds: 最大验证轮数
        """
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
    
    def generate_answer(
        self,
        query: str,
        context: List[Dict],
        schema: str = "text"
    ) -> GeneratedAnswer:
        """
        生成答案（带多轮验证）
        
        Args:
            query: 用户问题
            context: 检索到的上下文
            schema: 答案格式类型
            
        Returns:
            生成的答案对象
        """
        # 第一轮：生成初始答案
        initial_answer = self._generate_initial_answer(query, context, schema)
        
        # 多轮验证和重写
        current_answer = initial_answer
        for round_num in range(self.max_verification_rounds):
            # 验证答案
            validation = self._validate_answer(
                query, current_answer, context
            )
            
            if validation.is_valid:
                break
            
            # 如果验证不通过，重写答案
            if round_num < self.max_verification_rounds - 1:
                current_answer = self._rewrite_answer(
                    query, current_answer, context, validation
                )
        
        # 最终验证引用
        validated_citations = self._validate_citations(
            current_answer, context
        )
        
        return GeneratedAnswer(
            answer=current_answer['answer'],
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
        """生成初始答案"""
        # 构建提示
        context_text = self._format_context(context)
        
        system_prompt = """你是一个专业的问答助手。基于提供的上下文回答问题。
要求：
1. 答案必须基于上下文，不要引入外部知识
2. 提供清晰的推理过程
3. 标注信息来源（页码）
4. 如果上下文不足以回答问题，明确说明"信息不足"
5. 对于数字类问题，确保数值准确"""

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
        
        result = json.loads(response.choices[0].message.content)
        return result
    
    def _validate_answer(
        self,
        query: str,
        answer: Dict,
        context: List[Dict]
    ) -> AnswerValidationResult:
        """
        验证答案质量
        
        Args:
            query: 原始问题
            answer: 生成的答案
            context: 上下文
            
        Returns:
            验证结果
        """
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
}"""

        user_prompt = f"""问题：{query}

答案：{answer['answer']}

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
        validation: AnswerValidationResult
    ) -> Dict:
        """根据验证结果重写答案"""
        system_prompt = """你是一个答案重写专家。根据验证反馈改进答案。
要求：
1. 解决所有指出的问题
2. 确保答案基于上下文
3. 保持推理过程的清晰性
4. 准确标注引用来源"""

        user_prompt = f"""问题：{query}

当前答案：{current_answer['answer']}

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
        
        return json.loads(response.choices[0].message.content)
    
    def _validate_citations(
        self,
        answer: Dict,
        context: List[Dict]
    ) -> List[Dict]:
        """
        验证引用的准确性
        
        Args:
            answer: 答案数据
            context: 上下文
            
        Returns:
            验证后的引用列表
        """
        citations = answer.get('citations', [])
        validated_citations = []
        
        for citation in citations:
            # 提取页码
            page_num = self._extract_page_number(citation)
            if page_num is None:
                continue
            
            # 查找对应页面
            page_context = self._find_page_context(page_num, context)
            if page_context is None:
                continue
            
            # 验证引用内容是否确实在页面中
            is_valid = self._verify_citation_content(
                answer['answer'], page_context
            )
            
            validated_citations.append({
                'page': page_num,
                'is_valid': is_valid,
                'context_snippet': page_context[:200] if page_context else ""
            })
        
        return validated_citations
    
    def _extract_page_number(self, citation) -> Optional[int]:
        """从引用中提取页码"""
        if isinstance(citation, int):
            return citation
        if isinstance(citation, str):
            # 尝试提取数字
            numbers = re.findall(r'\d+', citation)
            if numbers:
                return int(numbers[0])
        return None
    
    def _find_page_context(self, page_num: int, context: List[Dict]) -> Optional[str]:
        """查找指定页面的上下文"""
        for doc in context:
            if doc.get('page') == page_num:
                return doc.get('text', '')
        return None
    
    def _verify_citation_content(
        self,
        answer_text: str,
        page_context: str
    ) -> bool:
        """验证答案内容是否在页面上下文中"""
        # 提取答案中的关键信息
        key_phrases = self._extract_key_phrases(answer_text)
        
        # 检查关键信息是否在页面中
        for phrase in key_phrases:
            if phrase.lower() in page_context.lower():
                return True
        
        return False
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """提取文本中的关键短语"""
        # 简单的实现：提取数字、专有名词等
        # 实际应用中可以使用NLP工具
        phrases = []
        
        # 提取数字
        numbers = re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', text)
        phrases.extend(numbers)
        
        # 提取引号中的内容
        quotes = re.findall(r'"([^"]+)"', text)
        phrases.extend(quotes)
        
        return phrases
    
    def _format_context(self, context: List[Dict]) -> str:
        """格式化上下文"""
        formatted = []
        for i, doc in enumerate(context, 1):
            formatted.append(f"【文档{i}】页码：{doc.get('page', 'N/A')}\n{doc.get('text', '')}")
        return "\n\n".join(formatted)
    
    def _format_context_summary(self, context: List[Dict]) -> str:
        """格式化上下文摘要"""
        summaries = []
        for doc in context:
            text = doc.get('text', '')
            summary = text[:200] + "..." if len(text) > 200 else text
            summaries.append(f"页码{doc.get('page', 'N/A')}：{summary}")
        return "\n".join(summaries)


class SelfConsistencyChecker:
    """自一致性检查器"""
    
    def __init__(self, generator: AnswerGeneratorOptimized, num_samples: int = 3):
        """
        初始化自一致性检查器
        
        Args:
            generator: 答案生成器
            num_samples: 采样次数
        """
        self.generator = generator
        self.num_samples = num_samples
    
    def generate_with_self_consistency(
        self,
        query: str,
        context: List[Dict]
    ) -> GeneratedAnswer:
        """
        使用自一致性生成答案
        
        Args:
            query: 用户问题
            context: 上下文
            
        Returns:
            最一致的答案
        """
        # 生成多个答案
        answers = []
        for _ in range(self.num_samples):
            answer = self.generator.generate_answer(query, context)
            answers.append(answer)
        
        # 选择最一致的答案
        best_answer = self._select_most_consistent(answers)
        
        return best_answer
    
    def _select_most_consistent(
        self,
        answers: List[GeneratedAnswer]
    ) -> GeneratedAnswer:
        """选择最一致的答案"""
        # 简单的实现：选择置信度最高的
        # 实际应用中可以使用更复杂的投票机制
        return max(answers, key=lambda x: x.confidence)


# 便捷函数
def generate_answer_optimized(
    query: str,
    context: List[Dict],
    use_verification: bool = True,
    use_self_consistency: bool = False
) -> Dict:
    """
    生成优化后的答案（便捷函数）
    
    Args:
        query: 用户问题
        context: 上下文
        use_verification: 是否使用验证
        use_self_consistency: 是否使用自一致性
        
    Returns:
        答案字典
    """
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
