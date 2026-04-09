# mini-megatrain

`mini-megatrain` 是一个“教学版 MegaTrain”实现：它不追求通吃 HuggingFace 全家桶，也不追求论文级吞吐，而是把最核心的系统思想收缩成一套你可以读懂、改动、验证的最小闭环。

这个版本保留了 MegaTrain 最值得学习的 4 个主干：

1. CPU 是参数和优化器状态的权威存储
2. GPU 只是瞬时计算引擎，按层把权重搬上去执行
3. 反向传播通过分组重算来避免整网激活常驻显存
4. 前向和重算阶段都带一个教学型的双缓冲预取器

同时它刻意简化了论文和原仓库里最重的工程部分：

1. 只支持 text-only、dense decoder-only Transformer
2. 不做 HuggingFace 通用结构发现
3. 不做 MoE、混合注意力、VLM、C++/CUDA extension
4. 不追求真正的 pinned slab 池与生产级三流调度

为了避免得出错误结论，这个版本还显式约束了两件事：

1. 训练时 `dropout` 必须保持为 `0.0`，因为当前教学版还没有实现重算所需的 RNG 状态保存/恢复
2. 默认把组输入 checkpoint 保留在设备侧，而不是额外搬回 CPU

换句话说，这个目录更像“把论文的系统思想翻译成能读懂的 PyTorch 教材”，而不是原始 MegaTrain 仓库的精简拷贝。

## 目录

```text
mini-megatrain/
  configs/
    demo_1b.yaml
    tiny_debug.yaml
  src/mini_megatrain/
    config.py
    dataset.py
    model.py
    engine.py
    train.py
  tests/
```

## 核心设计

### 1. CPU master model

完整模型始终保存在 CPU 的 FP32 参数里，优化器也直接更新这份 CPU 参数。

### 2. Working copies

每个 step 都只把下面几类对象的“工作副本”搬到计算设备上：

1. 输入 embedding
2. 当前正在执行的一组 Transformer blocks
3. 最终 norm 和 lm head

这些副本用完就释放；下一次 step 会从 CPU master 重新 materialize。

### 3. 分组重算

教学版把 `checkpoint_group_size` 个 block 视为一个“重算组”：

1. 前向阶段：该组在 `torch.no_grad()` 下执行，只保存组输入
2. 反向阶段：从保存的组输入重新跑这组 block
3. 立刻用 `torch.autograd.grad()` 把参数梯度和输入梯度算出来
4. 把参数梯度累加回 CPU master 参数的 `.grad`

这就是 MegaTrain 论文里“block-wise recompute”的教学化实现。

### 4. 双缓冲预取

原论文强调 double buffering 是吞吐主干。这里的实现是一个“教学型近似”：

1. 当前 layer 在默认 stream 上计算
2. 下一层的工作副本会被提前放到一个 prefetch stream 上构造
3. 真正使用下一层前，再让默认 stream 等待 prefetch stream

由于 Python `deepcopy()` 和模块构造本身也在 host 侧运行，这个版本的重叠效果不可能接近论文实现，但代码结构已经保留了那条关键控制流。它仍然不是论文里的 stateless template / flat-buffer pipeline。

## 快速开始

先进入目录：

```bash
cd /Users/siancao/work/ai/research/mini-megatrain
```

跑一个 CPU 上的超小 smoke test：

```bash
PYTHONPATH=src python -m mini_megatrain.train --config configs/tiny_debug.yaml
```

如果你有 CUDA，想做 1B 级别机制验证：

```bash
PYTHONPATH=src python -m mini_megatrain.train --config configs/demo_1b.yaml
```

## 1B 配置说明

`configs/demo_1b.yaml` 使用的是一个大约 `1,008,281,088` 参数的自定义 dense decoder：

1. `hidden_size = 1536`
2. `num_layers = 24`
3. `num_heads = 12`
4. `mlp_hidden_size = 6144`
5. `vocab_size = 32768`
6. `tie_word_embeddings = false`

它的目标不是语言质量，而是让你验证下面这些系统层问题：

1. 显存峰值是否主要跟“当前工作组”和输入长度有关
2. 分组重算是否能把训练 step 跑通
3. CPU master + GPU working copies 的参数同步逻辑是否正确
4. 双缓冲控制流是否工作正常

## 你应该怎么读这份代码

建议顺序：

1. `config.py`：先看有哪些可调旋钮
2. `model.py`：理解自定义 dense decoder 的数学结构
3. `engine.py`：这是整个教学版最核心的文件
4. `train.py`：看训练入口如何把数据、模型、引擎串起来
5. `tests/test_streaming_equivalence.py`：看它如何证明“流式重算版”和普通 dense 训练在小模型上是一致的

## 已知边界

1. 这是教学实现，不是性能实现
2. 默认数据是随机 token 数据集，只用于系统验证
3. 1B 配置是否能跑通，仍取决于你的 GPU 显存、PCIe/NVLink 带宽和 CPU 内存
4. 这个版本没有做 checkpoint save/resume
5. 这个版本没有做生产级 pinned slab pool、flat tensor packing 和 stateless CUDA templates
6. 这个版本没有实现 `gradient_accumulation_steps`，每个 `train_step` 都会更新一次参数
7. 这个版本不是原仓库的 SFT/chat-template 数据路径，而是离线随机 token 系统验证路径

## 为什么它仍然有价值

原始论文和仓库最难读的地方，不是 Transformer 本身，而是“参数生命周期”和“反向阶段的数据流”。这个教学版把那两件事拆得非常直白：

1. CPU 参数什么时候是权威副本
2. GPU 工作副本什么时候创建、什么时候释放
3. 组输入为什么要保存
4. 为什么 backward 里要重新 forward 一次
5. 梯度是怎么回到 CPU master 的

如果你准备基于论文思想继续做自己的 1B 验证，这个目录应该能当一个比原仓库更容易下手的起点。
