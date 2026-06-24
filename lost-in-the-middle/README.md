# Lost in the Middle — 2026 复刻实验

复现并扩展 Liu et al. 2023 的经典论文「Lost in the Middle: How Language Models Use Long Contexts」（TACL 2023），针对 2026 年的现代模型测试上下文位置敏感性。

## 论文核心

- **任务**：多文档问答（Multi-document QA, NQ-Open）+ 键值检索（KV Retrieval）。
- **核心结论**：当相关信息位于上下文的**开头**或**末尾**时模型准确率最高，位于**中部**时显著下降，呈现 U 形曲线。即使是显式声称支持长上下文的模型也未能完全克服。
- **被测模型（2023）**：GPT-3.5-Turbo (4k/16k)、Claude 1.3 (8k/100k)、LongChat-13B (16k)、MPT-30B-Instruct、Llama-2、Vicuna。

本仓库的目标：在 **2026 年**用同样的方法测试新一代模型（默认通过 `.env` 配置 Volces ARK 上的 `ark-code-latest`），看 U 形是否还在。

## 测试设计

| # | 实验 | 描述 | 直接对应论文 |
|---|------|------|--------------|
| A | 多文档 QA | 使用论文原始 NQ-open 数据，固定 10/20 个文档，将黄金文档放到不同位置 | §2 |
| B | 闭卷基线 | 不提供文档直接问，衡量模型自身知识 | §2.2 |
| C | KV 检索 | 合成 UUID 键值对，将目标键放到不同位置 | §4 |

## 项目结构

```
lost-in-the-middle/
├── .env                    # 模型 API 配置（已 gitignore）
├── README.md               # 本文件
├── litm_runner.py          # 主测试脚本
├── src/
│   └── prompting.py        # 与原论文一致的 prompt 模板
├── data/qa/                # NQ-open 多文档 QA 数据（从原论文 GitHub 下载）
└── results/                # 实验输出
    ├── qa_{N}doc_pos{P}.json    # 每个位置的准确率
    ├── qa_{N}doc_pos{P}_raw.json # 样例回答（调试用）
    ├── kv_{N}keys.json           # KV 实验结果
    ├── closed_book.json          # 闭卷基线
    └── report.md                 # 最终 markdown 报告
```

## 使用方法

```bash
# 1. 安装依赖
python3 -m venv .venv
.venv/bin/pip install openai python-dotenv

# 2. 配置 .env（已就位）
# LITM_BASE_URL=...
# LITM_API_KEY=...
# LITM_MODEL=...

# 3. 数据下载（已就位）
# 详见 scripts/download_qa_data.sh

# 4. 运行
.venv/bin/python litm_runner.py
```

## 复现要点

1. **Prompt 模板**：与论文 `src/lost_in_the_middle/prompts/qa.prompt`、`kv_retrieval.prompt` 逐字一致。
2. **文档格式**：`Document [{idx+1}](Title: {title}) {text}`，与论文一致。
3. **评测指标**：QA 任务沿用论文的 *substring match* —— 任一黄金答案（normalized）作为子串出现即算正确，比 EM 更宽容，与论文可比。
4. **数据集**：直接复用论文 GitHub 仓库提供的预构建 `nq-open-{10,20}_total_documents_gold_at_{0,4,9,...}.jsonl.gz`，避免重做 Contriever 检索引入差异。

## 引用

```bibtex
@article{liu-etal-2024-lost,
    title = "Lost in the Middle: How Language Models Use Long Contexts",
    author = "Liu, Nelson F. and Lin, Kevin and Hewitt, John and Paranjape, Ashwin and Bevilacqua, Michele and Petroni, Fabio and Liang, Percy",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "12",
    year = "2024",
    url = "https://aclanthology.org/2024.tacl-1.9",
    pages = "157--173",
}
```
