# Optimize Anything 版本 AutoResearch（原型）

本目录包含：

1. `REPORT.md`：AutoResearch 与 Optimize Anything 的深入对比和可行性分析。
2. `optimized_anything_autoresearch.py`：一个可运行的 Optimize Anything 风格 AutoResearch 原型。

## 快速开始

```bash
python optimized_anything_autoresearch.py \
  --topic "如何构建高可靠性的代码代理系统" \
  --max-iters 8 \
  --outdir outputs
```

运行后会生成：

- `outputs/report.md`
- `outputs/trace.json`

## 说明

当前实现默认使用 `ToolAdapter` 的 mock 能力（便于离线验证）。
如需接入真实能力，可替换下列方法：

- `plan_subquestions`
- `search`
- `summarize`
- `critique`

