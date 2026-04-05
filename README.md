# RAG Challenge 2（扩展版）

基于原始获奖项目 [IlyaRice/RAG-Challenge-2](https://github.com/IlyaRice/RAG-Challenge-2) 的中文扩展版本，面向公司年报问答场景。

## 核心增强

- 查询改写（Query Rewrite）
- 多轮会话记忆
- 答案与上下文相似度校验
- 多嵌入模型支持（`openai` / `bge_api`）
- BGE 重排支持
- `enhanced / legacy` 运行模式切换

## 快速开始

```bash
python -m venv venv
venv\Scripts\Activate.ps1
pip install -e . -r requirements.txt
```

将 `.env.example` 复制为 `.env`，至少配置：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`（代理通常要带 `/v1`）
- `BGE_API_KEY`（使用 BGE 时）

运行（推荐增强模式）：

```bash
cd data/test_set
python ..\..\main.py process-questions --config base --profile enhanced
```

## 运行模式

- `enhanced`（默认）：改造后结构，默认 BGE 优先（`bge_api + bge`）
- `legacy`：原项目风格，默认 `openai + llm reranker`

示例：

```bash
python main.py process-questions --config base --profile enhanced
python main.py process-questions --config base --profile legacy
python main.py process-questions --config base --profile enhanced --embedding-provider openai --reranker-type llm
```

## 金融垂直模式

```bash
cd data/test_set
python ..\..\main.py process-questions --config finance_vertical --profile enhanced
```

评测：

```bash
python scripts/finance_eval.py --pred data/test_set/answers_finance_vertical.json
python scripts/finance_eval.py --pred data/test_set/answers_finance_vertical.json --gold data/test_set/answers_max_nst_o3m.json
```

### 设计构思

金融问答的核心难点不是“能否回答”，而是“口径是否正确”。  
同一个问题中，指标名称、时间范围、币种、单位只要有一个不一致，就可能产生看似合理但实际错误的答案。

因此金融垂直模式的设计目标是：

1. 优先保证口径一致性（宁缺毋滥）
2. 减少“相关指标替代目标指标”的幻觉
3. 让检索与生成都围绕金融语义约束工作

### 核心机制

- **检索前约束改写**：保留公司名、财年、单位、币种，不做语义漂移
- **BGE 优先检索链路**：在金融数据上提升相关片段召回稳定性
- **金融规则后处理**：对数值答案做单位规范化、币种一致性检查
- **缺失值保守策略**：上下文不足或口径冲突时返回 `N/A` / 信息不足

### 典型应用场景（给业务方/团队直接使用）

- **投研与研报辅助**：从多家公司年报中快速抽取关键财务指标（营收、净利、现金流、资产负债等）并给出处页码
- **财务尽调与并购分析**：批量问答目标公司历史财务口径，减少人工翻阅 PDF 的时间成本
- **审计与内控支持**：对指标口径、币种、单位做一致性校验，降低“数字看起来对但口径错”的风险
- **IR/董秘与管理层问答支持**：把高频财务问题转成可追溯问答，快速定位原文证据
- **金融知识库检索中台**：作为企业内部财报问答引擎，为 BI、风控、客服机器人提供结构化问答能力

### 业务意义

- 在财务分析、投研支持、审计辅助等场景中，降低“数字正确但口径错误”的风险
- 提升结果可解释性：回答更容易追溯到对应页码与原文
- 更适合做自动化批量问答，因为错误类型更可控、可监控

### 适用边界

- 该模式更偏“保守准确”，在信息缺失时会更倾向拒答
- 如果你的场景更看重召回覆盖率，可配合 `legacy` 或放宽后处理策略

## 常用实验脚本

- `benchmark_prepare_round1.py`：准备 benchmark 数据
- `benchmark_compare_round1.py`：跑对比评测
- `scripts/ab_score_openai_bge.py`：OpenAI vs BGE 打分
- `scripts/triple_eval_repeats.py`：三组重复实验
- `scripts/triple_recompute_from_existing.py`：重算修正版均值

## 当前结果（benchmark_round1_soft）

- 改造后 OpenAI vs 改造后 BGE（10题）：`90.00%` vs `100.00%`
- 三组 3 次重复平均：
  - 原项目：`93.33%`
  - 改造后 OpenAI：`83.33%`
  - 改造后 BGE：`100.00%`

结果文件：

- `data/benchmark_round1_soft/ab_report_openai_vs_bge.json`
- `data/benchmark_round1_soft/triple_compare_repeats_report_fixed.json`

## 测试

```bash
$env:RUN_INTEGRATION_TESTS="1"
python -m pytest -q
```

## 许可证

MIT
