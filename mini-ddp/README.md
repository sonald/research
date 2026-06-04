# mini-ddp

`mini-ddp` 是一个教学版 PyTorch DDP：它把生产版 `DistributedDataParallel` 的核心控制流缩小到一个容易读、容易测试、能在 CPU 或 Apple MPS 上运行的实现。

它不是 NCCL/Gloo 的替代品，也不追求吞吐；它保留的是 DDP 最值得学习的结构：

1. 每个 rank 有一份完整模型副本
2. 初始化时从 rank 0 广播参数和 buffer
3. 每个参数注册 autograd hook
4. Reducer 把参数梯度组织成 bucket
5. backward 后对每个 bucket 做 all-reduce mean
6. 每个 optimizer 只更新本地 replica，但因为梯度相同，所以 replica 保持同步

## 目录

```text
mini-ddp/
  src/mini_ddp/
    core.py       # MiniDDP、MiniReducer、bucket、shard_batch
    demo.py       # CPU/MPS smoke demo
  tests/
    test_mini_ddp.py
  docs/
    research.md    # Torch DDP 实现原理调研
  site/
    index.html    # 第一性原理教学网站
```

## 快速开始

```bash
cd /Users/siancao/work/ai/research/mini-ddp
uv run --extra dev pytest
uv run mini-ddp-demo --device cpu
uv run mini-ddp-demo --device auto
```

`--device auto` 会在 MPS 可用时使用 MPS，否则回退到 CPU。

## 和生产版 PyTorch DDP 的关系

这个实现来自对 PyTorch 官方 DDP 文档、设计笔记和源码入口的抽象：

- 生产版 DDP 依赖 `ProcessGroup` 做跨进程通信
- DDP 构造阶段会广播 rank 0 的模型状态
- `Reducer` 在构造阶段给每个参数的 gradient accumulator 注册 hook
- 参数按大小放进 bucket，梯度 ready 后 bucket 才能 reduce
- bucket 的 all-reduce 结果会写回 `param.grad`
- optimizer 看起来只是在更新本地模型，但所有 rank 的参数会保持一致

更完整的调研笔记在 `docs/research.md`。

`mini-ddp` 为了能在 CPU/MPS 上稳定教学，把“跨进程通信”改成了一个进程内的多 replica 平均；把“异步通信和反向计算重叠”改成了 hook 记录 ready 状态、backward 结束后同步。也就是说，它牺牲性能路径，保留可解释的语义路径。

## 建议阅读顺序

1. 先看 `site/index.html` 的第一性原理图解
2. 再看 `src/mini_ddp/core.py` 里的 `MiniDDP.__init__`
3. 接着看 `MiniReducer._register_autograd_hooks`
4. 最后看 `MiniReducer.synchronize`

## 已知边界

1. 不支持 unused parameters；每个参数都必须参与当前 backward
2. 不做跨进程 rendezvous、容错、join、timeout
3. 不做通信和反向计算重叠，只保留 bucket ready 机制
4. 不做 mixed precision、gradient_as_bucket_view、communication hook
5. MPS 路径用于验证语义，不代表生产分布式训练方式
