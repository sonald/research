# mini-gepa

这是一个**教学版 GEPA 实现**，用于把论文 **GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning**（arXiv:2507.19457v2）和 `gepa-ai/gepa` 官方代码中的核心思想，压缩成可快速阅读的最小骨架。

> 目标：保留主干，去掉大量工程细节（边界处理、异常恢复、复杂策略组合、日志/追踪系统、缓存恢复等）。

---

## 我用“子代理式”两阶段流程做了什么

### 子代理 A：调研与抽象
从论文摘要与仓库 `src/gepa` 的 API / 核心流程中抽象出以下主干：
1. 维护候选池（candidate pool）。
2. 基于 **Pareto frontier** 选择父代。
3. 在 minibatch 上收集可反思的 side information（ASI 思想）。
4. 通过“反思 -> 变异”产出子代。
5. 在验证集评估并按简单接受策略入池。
6. 迭代直到预算耗尽。

### 子代理 B：实现审查与修复
- 审查后发现：收敛后会反复加入重复候选，导致教学日志噪声较大。
- 已修复：新增 `seen_prompts` 去重。
- 详见 `review_subagent_report.md`。

---

## 文件说明

- `mini_gepa.py`：主实现（含大量教学注释）。
- `review_subagent_report.md`：第二阶段审查与修复记录。

---

## 运行方式

```bash
python mini-gepa/mini_gepa.py
```

输出会展示每一步：
- 父代均分
- 子代均分
- 候选池大小

以及最终最佳 prompt。

---

## 与官方 GEPA 的差异（刻意简化）

1. 只优化单文本组件（prompt），而非多组件系统。
2. 反思变异不是 LLM 生成，而是规则化拼接（便于教学观察）。
3. 接受策略、候选选择、评估策略都使用最小版本。
4. 不含缓存、恢复、并行、实验跟踪、回调等工程模块。

如果你希望，我可以在 `mini-gepa` 里继续加一个 `v2`：
- 支持多组件 candidate（如 `{"system_prompt": ..., "tool_desc": ...}`）
- 支持“伪多目标”Pareto（多个 objective）
- 用可插拔 proposal 函数模拟官方 adapter/proposer 分层。
