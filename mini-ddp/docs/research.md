# Torch DDP 实现原理调研

这份调研把 PyTorch `DistributedDataParallel` 的生产实现拆成教学版 `mini-ddp` 能复现的语义骨架。目标不是复刻 NCCL/Gloo 性能路径，而是保留“为什么多份模型副本能像一份大 batch 模型一样训练”的关键机制。

## 主要资料

1. PyTorch DDP design note: https://docs.pytorch.org/docs/2.12/notes/ddp.html
2. PyTorch DDP tutorial: https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html
3. PyTorch DDP API: https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
4. PyTorch `distributed.py` 入口: https://github.com/pytorch/pytorch/blob/main/torch/nn/parallel/distributed.py
5. PyTorch `reducer.cpp` 核心 reducer: https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/reducer.cpp
6. PyTorch Distributed 论文: https://arxiv.org/abs/2006.15704

## PyTorch DDP 的一轮训练

### 1. ProcessGroup 是通信基座

生产版 DDP 依赖 c10d `ProcessGroup`。用户先初始化进程组，DDP 后续用它做两类集体通信：

1. 初始化/前向阶段的 broadcast
2. 反向阶段的 all-reduce

`mini-ddp` 不创建真实进程组，而是在一个 Python 进程里维护多个 replica，用 tensor 平均模拟 all-reduce mean。

### 2. 构造阶段同步初始状态

DDP 构造时会把 rank 0 的 `state_dict()` 广播给其他 rank，保证所有模型副本从同一参数状态出发。这个步骤是数据并行正确性的第一个不变量：如果初始参数不同，即便后面同步梯度，也不再等价于单模型大 batch 训练。

`mini-ddp` 对应实现：

- `MiniDDP.__init__` 深拷贝出 `world_size` 个 replica
- `broadcast_parameters_and_buffers()` 把 rank 0 的 state 加载到其他 replica

### 3. Reducer 建 bucket 并注册 autograd hook

生产版 DDP 的 Reducer 会把参数梯度按 bucket 组织起来。bucket 的目的不是改变数学结果，而是让通信能在反向传播过程中尽早开始，并和后续反向计算重叠。DDP 还会给每个参数的 gradient accumulator 注册 hook；当某个参数梯度 ready，hook 就标记它所在 bucket 的 ready 状态。

`mini-ddp` 对应实现：

- `MiniReducer._build_buckets()` 根据 `bucket_cap_mb` 建 bucket
- `MiniReducer._register_autograd_hooks()` 给每个 replica 的每个参数注册 hook
- `_ready` 记录 `(rank, parameter_index)` 是否已经产生梯度

### 4. backward 触发梯度同步

用户调用的是普通 `loss.backward()`，不是 `ddp.backward()` 这种生产 API。DDP 的关键在于：构造阶段已经把 hook 埋进 autograd，所以反向传播产生梯度时，Reducer 能被动收到 ready 事件。生产版在 bucket ready 后发起异步 all-reduce；当 backward 返回时，参数的 `.grad` 已经是同步后的平均梯度。

`mini-ddp` 为了保持 CPU/MPS 可测，把这个异步路径收缩成同步路径：

- `MiniDDP.backward(losses)` 依次对每个 replica 的 loss 调 `backward()`
- `MiniReducer.synchronize()` 检查每个 bucket 是否完整 ready
- 对同一参数在所有 replica 上的梯度取 mean
- 把平均梯度写回每个 replica 的 `param.grad`

### 5. optimizer 仍然是本地 optimizer

DDP 不需要一个“全局 optimizer”。每个 rank 的 optimizer 看起来只更新本地模型。因为所有 replica 初始参数一样，且每轮 backward 后梯度一样，所以同样的 optimizer step 会让所有参数继续保持一致。

`mini-ddp` 对应实现：

- `MiniDDP.optimizers()` 为每个 replica 创建一个本地 optimizer
- `assert_replicas_equal()` 用于验证 optimizer step 后 replica 没有分叉

## 语义等价性

对常见 mean reduction loss 来说，如果每个 rank 处理同样大小的 batch shard，那么：

```text
grad_large_batch = mean(grad_rank_0, grad_rank_1, ..., grad_rank_n)
```

这就是 `mini-ddp` 测试里要证明的事情：

1. 单模型直接训练完整 batch 一步
2. `MiniDDP` 把 batch 切成两个 shard，各 replica 各自 backward
3. Reducer 平均梯度并写回
4. 两条路径的参数更新结果一致

## 没有复刻的生产特性

`mini-ddp` 有意不实现以下特性：

1. 真实跨进程 rendezvous、timeout、fault tolerance
2. NCCL/Gloo backend
3. 异步通信和反向计算重叠
4. `find_unused_parameters`
5. `gradient_as_bucket_view`
6. DDP communication hook
7. uneven input join
8. mixed precision reducer 路径

这些特性很重要，但它们属于“让 DDP 在生产环境快、稳、可扩展”的工程层。这个项目优先保留数学语义和控制流骨架。
