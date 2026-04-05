# RAG Challenge 2（扩展版分支）

这是一个面向年报长文档问答的 RAG 系统实现，基于比赛获奖方案扩展，新增了多模型嵌入、多轮会话与可靠性校验能力。

上游参考仓库：  
- [bingchengle/RAG-Challenge-2](https://github.com/bingchengle/RAG-Challenge-2)

## 本分支新增能力

- 检索前查询改写（Query Rewrite）
- 多轮对话管理（`conversation_id` + 历史窗口）
- 答案与上下文相似度校验
- 多模型嵌入支持（`openai` / `bge_api`）
- 可选 BGE API 重排（`reranker_type="bge"`）
- 更可控的集成测试（默认不触发外部 API）

## 项目结构

```text
.
├── main.py                         # Click CLI 入口
├── src/
│   ├── pipeline.py                 # 全流程 Pipeline + 运行配置
│   ├── questions_processing.py     # 问答主流程（改写/检索/会话）
│   ├── retrieval.py                # 向量/BM25/混合检索 + 重排
│   ├── ingestion.py                # 向量库与 BM25 索引构建
│   ├── embedding_clients.py        # OpenAI/BGE 嵌入与 BGE 重排客户端
│   ├── answer_generator_optimized.py
│   ├── api_requests.py
│   └── ...
├── data/
│   ├── test_set/                   # 小规模可运行数据
│   └── erc2_set/                   # 全量竞赛元数据与问题集
├── test_*.py                       # 冒烟/集成测试
├── requirements.txt
└── .env.example
```

## 环境安装

```bash
python -m venv venv
venv\Scripts\Activate.ps1
pip install -e . -r requirements.txt
```

将 `.env.example` 复制为 `.env`，并填写密钥。

常用必填项：
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`（如果你使用代理，通常是 `.../v1`）
- `BGE_API_KEY`（当使用 `bge_api` 嵌入或 BGE 重排时）

## 数据集说明

`data/test_set/` 目录包含：
- `questions.json`、`subset.csv`
- `databases.zip`（预处理后的检索数据）
- `pdf_reports/`（原始 PDF）
- 示例答案文件

快速使用预处理数据：
1. 解压 `data/test_set/databases.zip`
2. 确认目录结构如下：
   - `data/test_set/databases/chunked_reports`
   - `data/test_set/databases/vector_dbs`

> `databases/` 体积较大，通常不建议重复生成后提交。

## 快速运行

在测试数据目录执行：

```bash
cd data/test_set
python ..\..\main.py process-questions --config base
```

运行后会在 `data/test_set/` 下生成 `answers_*.json`。

## 嵌入与重排配置

在 `src/pipeline.py` 的 `RunConfig` 中配置：

- `embedding_provider`: `"openai"` 或 `"bge_api"`
- `embedding_model`: 可选，手动覆盖模型名
- `reranker_type`: `"llm"` 或 `"bge"`

示例：

```python
RunConfig(
    embedding_provider="bge_api",
    embedding_model="BAAI/bge-large-zh-v1.5",
    llm_reranking=True,
    reranker_type="bge",
)
```

## 多轮对话与查询改写

`QuestionsProcessor` 支持：
- `enable_query_rewrite=True`
- `enable_multi_turn=True`
- `conversation_max_turns=<N>`

每条问题可选字段：
- `conversation_id`
- `conversation_history`（或 `history`）

若未传入外部历史，系统会按 `conversation_id` 从内部会话存储中读取历史。

## 测试说明

- `test_final.py`：导入冒烟测试
- `test_api_answer_generator.py`
- `test_embedding_comparison.py`
- `test_fallback.py`
- `test_answer_generator_optimized.py`

后四项为集成测试，默认跳过；开启方式：

```bash
$env:RUN_INTEGRATION_TESTS="1"
python -m pytest -q
```

## 许可证

MIT