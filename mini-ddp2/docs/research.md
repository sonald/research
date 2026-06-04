# Torch DDP 实现原理调研

这份笔记服务于 `mini-ddp2`：我们先调研 PyTorch `DistributedDataParallel` 的真实控制流，再把它压缩成一个可以在 CPU/MPS 上验证的教学实现。

## 主要资料

- PyTorch DDP design note: https://docs.pytorch.org/docs/stable/notes/ddp.html
- PyTorch DDP tutorial: https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html
- PyTorch DDP API: https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
- PyTorch `distributed.py`: https://github.com/pytorch/pytorch/blob/main/torch/nn/parallel/distributed.py
- PyTorch `reducer.cpp`: https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/reducer.cpp
- PyTorch Distributed 论文: https://arxiv.org/abs/2006.15704

## 1. DDP 要解决的问题

单模型训练时，一步更新大致是：

```text
batch -> model.forward -> loss -> loss.backward -> param.grad -> optimizer.step
```

如果 batch 很大，或者模型很大，单设备训练会慢。数据并行的想法是：复制多份完整模型，每份模型处理 batch 的一部分。关键问题不在 forward，而在 backward：每个 replica 只能看到本地 shard，因此每个 replica 产生的是本地梯度。如果直接让每个 rank 用本地梯度更新参数，模型副本很快会分叉。

DDP 的核心答案是：每轮 backward 后，把同一个参数在所有 rank 上的梯度做 all-reduce mean，然后把平均梯度写回每个 rank 的 `param.grad`。这样每个 optimizer 看到的是同一份梯度，参数继续保持同步。

## 2. ProcessGroup 是通信抽象

生产版 DDP 不直接关心 NCCL 或 Gloo 的细节，它通过 c10d `ProcessGroup` 调用 collective communication。DDP 主要依赖两类 collective：

- broadcast：初始化或 forward 前，把 rank 0 的状态复制给其他 rank。
- all-reduce：backward 中，把所有 rank 的梯度汇总并分发回每个 rank。

`mini-ddp2` 用 `InMemoryProcessGroup` 保留这个接口形状：

```python
process_group.broadcast_state_dict(replicas, source_rank=0)
process_group.all_reduce_mean(flat_grads_by_rank, bucket_index=bucket.index)
```

这里没有 socket、rendezvous、进程间通信。多个 rank 只是一个 Python 进程中的多个模型副本。这样牺牲了生产环境的通信路径，但保留了数学语义。

## 3. 构造阶段为什么要 broadcast

DDP 正确性的第一个前提是所有 replica 从同一组参数开始。如果 rank 0 和 rank 1 的初始参数不同，即便之后每步梯度相同，optimizer step 也会把不同参数更新成不同参数。

所以构造阶段必须同步 state：

```text
rank 0 state_dict
  -> rank 1 state_dict
  -> rank 2 state_dict
```

`state_dict` 包含 parameter，也包含 buffer。BatchNorm 的 running statistics 就是典型 buffer。`mini-ddp2` 的 `broadcast_parameters_and_buffers()` 直接把 rank 0 的 `state_dict()` 加载到其他 replica。

## 4. Reducer 为什么注册 autograd hook

用户调用的仍然是普通 PyTorch autograd：

```python
loss.backward()
```

DDP 没有要求用户手动告诉它每个参数的梯度什么时候产生。它在构造时给每个 trainable parameter 的 gradient accumulator 注册 hook。反向传播计算到某个参数的梯度时，hook 会触发，Reducer 就知道：

```text
rank r 的 parameter p 已经 ready
```

这个机制很重要，因为真实 DDP 想在 backward 还没完全结束时就尽早发起通信。后层参数通常先产生梯度，先 ready 的 bucket 可以先 all-reduce，从而和前层 backward 计算重叠。

`mini-ddp2` 为了可读性没有实现异步重叠，但仍然保留 hook 和 ready 记录：

```python
param.register_hook(self._make_hook(rank, param_index))
```

## 5. bucket 不是数学必需，而是性能结构

如果模型有很多小参数，每个参数单独 all-reduce 会产生大量通信调用。DDP 把参数分成 bucket：一个 bucket 包含多个参数的梯度。bucket 中的所有参数都 ready 后，Reducer 把这些梯度 flatten 成一段连续 tensor，再做一次 all-reduce。

数学上：

```text
reduce(g1), reduce(g2), reduce(g3)
```

和：

```text
reduce(flatten(g1, g2, g3))
```

结果等价。bucket 改变的是通信粒度和调度时机，不改变平均梯度的定义。

`mini-ddp2` 的 bucket 由 `bucket_cap_mb` 控制。为了贴近 backward ready 顺序，它从参数列表尾部开始打包，因为后层参数通常先在 backward 中 ready。

## 6. all-reduce mean 如何等价于大 batch

假设每个 rank 的 shard 大小相同，loss 使用 mean reduction。rank 0 梯度是本地 shard 的平均梯度，rank 1 也是本地 shard 的平均梯度。对这些 rank-local 梯度再取平均：

```text
grad_ddp = mean(grad_rank_0, grad_rank_1, ..., grad_rank_n)
```

这等价于单模型直接在完整 batch 上计算 mean loss 的梯度。

这个等价性有前提：

- 每个 shard 大小相同，或者用正确权重处理不等 shard。
- 每个 rank 使用同一份初始参数。
- 每个 rank 走同样的参数使用图。
- 每个 rank 的 optimizer 配置和状态同步。

`mini-ddp2` 的 `shard_batch()` 有意拒绝不等 shard，因为 uneven input 是生产 DDP 的另一个主题。

## 7. optimizer 为什么仍然是本地的

DDP 不需要一个全局 optimizer。每个 rank 都有一个普通 optimizer：

```python
optimizer = torch.optim.SGD(model.parameters(), lr=...)
```

只要满足：

```text
same parameters before step
same averaged gradients
same optimizer state
same optimizer hyperparameters
```

那么每个 rank 的本地 optimizer step 会产生同样的新参数。`mini-ddp2` 的 `assert_replicas_equal()` 就是用来验证这个不变量。

## 8. no_sync 做了什么

`no_sync()` 的目的通常是梯度累积。假设想把两个 microbatch 当成一个更大的有效 batch，可以先在 `no_sync()` 中执行 backward，让每个 rank 保留本地累计梯度，不做 all-reduce。下一次正常 backward 时，Reducer 再把累计后的本地梯度一起平均。

`mini-ddp2` 的行为是：

```text
microbatch 1 backward -> hooks fire -> skip synchronize
microbatch 2 backward -> hooks fire -> all-reduce accumulated grad
```

注意：`no_sync()` 不是“不计算梯度”。它只是“不通信”。rank-local `.grad` 仍然会被 autograd 写入和累加。

## 9. unused parameter 和 rank-divergent branch

DDP 最容易让人困惑的一类错误是某些参数没有产生梯度。

如果一个参数在所有 rank 上都没有被使用，那么它可以被当作 unused parameter 处理。生产版 DDP 有 `find_unused_parameters` 路径，`mini-ddp2` 用 `allow_unused_parameters=True` 表示教学版跳过这些参数。

更危险的是 rank-divergent branch：

```text
rank 0 使用 optional layer
rank 1 没有使用 optional layer
```

这时同一个 bucket 中有些 rank 有梯度，有些 rank 没梯度。Reducer 无法对齐 all-reduce。真实训练中这类分叉可能导致 hang 或错误。`mini-ddp2` 会直接抛出包含 rank 和 parameter name 的错误。

## 10. 和生产版 PyTorch DDP 的边界

`mini-ddp2` 实现的是 DDP 语义骨架，不是生产替代品。

保留：

- rank 和 world size 概念。
- replica 初始化同步。
- buffer broadcast。
- autograd hook。
- bucket ready state。
- flatten bucket gradient。
- all-reduce mean。
- writeback to `param.grad`。
- local optimizer invariant。
- `no_sync` 梯度累积语义。
- unused parameter 的教学错误路径。

没有实现：

- 多进程启动和 rendezvous。
- Gloo/NCCL backend。
- 异步 all-reduce 和 backward overlap。
- timeout、join、fault tolerance。
- communication hook。
- mixed precision reducer。
- optimizer state sharding。
- uneven input join。

这些生产特性值得继续研究，但它们应该建立在本项目覆盖的语义前提之上。
