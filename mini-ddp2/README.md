# mini-ddp2

`mini-ddp2` 是一个从 0 实现 PyTorch `DistributedDataParallel` 核心语义的教学项目。它不追求 NCCL/Gloo 的生产性能，而是保留 DDP 最值得学习的机制：rank、replica、rank 0 broadcast、autograd hook、Reducer bucket、all-reduce mean、`param.grad` writeback、local optimizer step、`no_sync` 和 unused parameter 错误。

所有代码都能在单进程内运行，因此可以在 CPU 上稳定测试，也能在 Apple MPS 可用时跑 smoke test。

## 目录

```text
mini-ddp2/
  src/mini_ddp2/
    core.py       # MiniDDP, MiniReducer, InMemoryProcessGroup, Bucket
    demo.py       # CPU/MPS runnable demo
  tests/
    test_mini_ddp2.py
  docs/
    research.md   # PyTorch DDP 原理调研和实现映射
  site/
    index.html    # 细节完整的中文教学网站
    styles.css
    app.js
    assets/generated/*.png
```

## 快速验证

```bash
cd /Users/siancao/work/ai/research/mini-ddp2
uv run --extra dev pytest
uv run mini-ddp2-demo --device cpu
uv run mini-ddp2-demo --device auto
```

`--device auto` 会优先选择 MPS，MPS 不可用时回退到 CPU。

教学网站可以直接打开：

```text
/Users/siancao/work/ai/research/mini-ddp2/site/index.html
```

## 建议学习顺序

1. 打开 `site/index.html`，先按章节理解每个前提。
2. 阅读 `docs/research.md`，把 PyTorch 官方 DDP 概念和本项目代码对应起来。
3. 阅读 `src/mini_ddp2/core.py`，从 `InMemoryProcessGroup` 到 `MiniReducer.synchronize()`。
4. 运行 `uv run mini-ddp2-demo --device cpu`，观察 reducer trace。
5. 阅读 `tests/test_mini_ddp2.py`，用测试理解每个不变量。

## 这个实现保留了什么

- 每个 rank 拥有完整模型副本。
- 初始化阶段从 rank 0 广播参数和 buffer。
- 每个 trainable parameter 注册 autograd hook。
- Reducer 按 bucket 组织参数梯度。
- bucket ready 后执行 all-reduce mean。
- 平均梯度写回每个 replica 的 `param.grad`。
- 每个 rank 使用本地 optimizer，但由于参数和梯度一致，更新后仍保持同步。
- `no_sync()` 允许先积累 rank-local 梯度，再在下一次同步 backward 时平均。
- unused parameter 和 rank-divergent branch 会给出清晰错误。

## 这个实现没有复刻什么

- 没有真实多进程 rendezvous。
- 没有 NCCL/Gloo backend。
- 没有通信和 backward 计算重叠。
- 没有 optimizer state sharding。
- 没有 fault tolerance、timeout、join algorithm。
- 没有 mixed precision reducer 和 communication hook。

这些属于生产 DDP 的性能和工程可靠性层。`mini-ddp2` 的目标是让核心语义可读、可跑、可测。
