# mini-act-checkpointing

一个从零实现 activation checkpointing 的 learn-by-doing 小项目。

它包含两部分：

- `src/mini_act_checkpointing/checkpoint.py`: 一个教学但完整的 PyTorch reentrant checkpoint 实现，支持嵌套 `args/kwargs`、参数梯度、CPU/MPS RNG 保存恢复。
- `site/`: 中文交互式教学网站，用 SVG 动画解释“为什么省显存、为什么要重算、为什么 dropout 需要 RNG 回放、生产版 non-reentrant 又多了什么”。

## 运行

```bash
uv run --extra dev pytest
uv run mini-act-checkpointing-demo
python3 -m http.server 5173 -d site
```

然后打开 `http://127.0.0.1:5173`。

## 目录

```text
mini-act-checkpointing/
  docs/research.md
  site/index.html
  site/styles.css
  site/app.js
  src/mini_act_checkpointing/checkpoint.py
  src/mini_act_checkpointing/demo.py
  src/mini_act_checkpointing/memory.py
  tests/test_checkpoint.py
```

## 和 PyTorch 生产版的关系

这个项目实现的是最适合教学的 reentrant 核心：前向阶段不记录 checkpoint 区域内部 autograd graph，只保存输入和 RNG 状态；反向阶段恢复 RNG、重新运行 forward，用重算出来的激活构造局部 graph 并求梯度。

PyTorch 当前推荐 `use_reentrant=False`，生产版还会用 saved tensor hooks、metadata check、early stop、debug traces 等机制。站点和 `docs/research.md` 会把这部分作为“生产版桥段”拆开讲，但代码实现保持在一个适合学习和验证的小核心内。
