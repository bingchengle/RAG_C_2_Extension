# Enterprise RAG Challenge — Round 2 数据（本地整理）

本目录存放 [trustbit/enterprise-rag-challenge](https://github.com/trustbit/enterprise-rag-challenge) 仓库 **Round 2** 的公开材料，便于在本项目中扩展实验。

## 来源与许可

- 上游仓库：<https://github.com/trustbit/enterprise-rag-challenge>（`round2/`）
- PDF 列表：<https://github.com/trustbit/enterprise-rag-challenge/tree/main/round2/pdfs>
- 许可见同目录 `LICENSE`（与上游一致）。

## 目录说明

| 路径 | 说明 |
|------|------|
| `pdf_reports/` | 100 份年报 PDF，文件名为 `sha1.pdf`，与 `subset.csv` 中 `sha1` 对应 |
| `subset.csv` / `subset.json` | 公司元数据（含 `sha1`、`company_name`、币种与行业标签等） |
| `questions.json` | Round 2 题目（字段为 `text`、`kind`） |
| `answers.json` | 官方参考答案（用于评测对齐） |
| `dataset.json` | 赛方数据集描述 |
| `ranking.csv` | 赛方排行相关 |

## 与当前 RAG 管线的关系

- 将 `Pipeline` / `main.py` 的工作目录指到本文件夹（或复制为 `questions.json` + `subset.csv` + `pdf_reports/` 结构）即可按现有流程做 **parse → merge → chunk → 建库 → 答题**。
- Round 2 的 `kind` 包含如 `boolean`、`names` 等，与本项目原赛题（多为 `number` / `name` / `comparative`）可能不一致，**答题与 prompt 侧可能需要按需扩展**（下一步可一起做）。

## 更新 PDF 的方式

若需重新拉取上游文件，可在项目外执行：

```bash
git clone --depth 1 https://github.com/trustbit/enterprise-rag-challenge.git
# 复制 enterprise-rag-challenge/round2/pdfs/*.pdf 到本目录 pdf_reports/
```

或使用 Git LFS（若上游对大文件启用 LFS）后再复制。
