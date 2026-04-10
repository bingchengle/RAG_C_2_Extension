# 数据集目录（统一入口）

本目录集中存放评测语料与赛题数据，路径在代码中由 [`src/data_paths.py`](../../src/data_paths.py) 定义，避免散落硬编码。

## 子目录

| 目录 | 说明 |
|------|------|
| `test_set/` | 日常开发与 README 示例用的子集（含 `questions.json`、`subset.csv` 等） |
| `benchmark_round1_mini/`、`benchmark_round1_soft/`、`benchmark_round1_full/` | Round1 不同规模的 benchmark 与实验输出 |
| `enterprise_rag_round2/` | [Enterprise RAG Challenge](https://github.com/trustbit/enterprise-rag-challenge) Round2 的 PDF 与元数据 |
| `erc2_set/` | 原 ERC2 / RAG Challenge 参考题目与 subset（若存在） |

## 重复数据说明

- **已删除**：`data/_erc_round2_clone/`（与 `enterprise_rag_round2` 内容重复，仅为临时 git 克隆）。
- **跨目录相同 `sha1` 的 PDF**：不同 benchmark 子集可能包含同一公司的年报副本，这是**各子集可独立复现实验**所必需，**未做跨目录去重**（去重会破坏自包含目录）。
- **单目录内**：`subset.csv` 中 `sha1`/`company_name` 唯一，`questions.json` 题干无重复，`pdf_reports` 下文件名（sha1）唯一。可用 `python scripts/validate_datasets.py` 复检。

## 移动路径后的影响范围（若你仍有旧脚本）

以下已随迁移更新：

- 根目录 [`README.md`](../../README.md) 中的 `cd` 与评测示例路径  
- 根目录 [`benchmark_compare_round1.py`](../../benchmark_compare_round1.py)、[`benchmark_prepare_round1.py`](../../benchmark_prepare_round1.py)（薄入口，实现见 [`src/benchmark/compare_round1.py`](../../src/benchmark/compare_round1.py)、[`prepare_round1.py`](../../src/benchmark/prepare_round1.py)）的默认 `--bench-dir` / `--out-dir`  
- [`scripts/ab_score_openai_bge.py`](../../scripts/ab_score_openai_bge.py)、[`scripts/triple_eval_repeats.py`](../../scripts/triple_eval_repeats.py)、[`scripts/triple_recompute_from_existing.py`](../../scripts/triple_recompute_from_existing.py)  
- [`src/pipeline.py`](../../src/pipeline.py) 中 `__main__` 的默认 `root_path`  
- [`.gitignore`](../../.gitignore) 中与大文件相关的规则  
- `benchmark_round1_soft` 内部分报告 JSON 里记录的**绝对路径**字符串（已改为含 `datasets` 的新路径）

**你可能仍需自行修改的**：个人笔记、外部 CI、未入库的脚本里写死的 `data/test_set` 或 `data/benchmark_*`；以及历史上导出到别处的绝对路径引用。
