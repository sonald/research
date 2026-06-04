# Activation Checkpointing 调研笔记

## 一句话

Activation checkpointing 用“反向时多算一次 forward”换“前向时少存中间激活”。默认 eager autograd 会保存 backward 需要的张量；checkpoint 区域里，前向不保存内部激活，只保留区域输入，到了 backward 再重算内部激活。

## PyTorch 当前行为

PyTorch 文档把 checkpoint 描述成“在 backward 期间重新运行 forward 片段”的技术；如果 checkpoint 区域里用了 dropout 之类的随机算子，PyTorch 默认会保存并恢复 RNG 状态，让 checkpoint 与非 checkpoint 的输出/梯度保持一致。

当前 PyTorch 有两类实现：

- `use_reentrant=True`: 经典版本。前向在 `torch.no_grad()` 下运行，不记录区域内部 autograd graph；反向时重跑整个 function，再调用 autograd 得到梯度。
- `use_reentrant=False`: 当前推荐版本。前向会记录 autograd graph，并用 saved tensor hooks 把“本来要保存的内部张量”替换成占位 holder；backward 解包 holder 时触发局部重算，并且支持 early stop，只重算到拿到需要的 tensor 为止。

## 生产版源码里值得抓住的点

经典 reentrant 的形状很直：

1. `forward` 里把 tensor inputs 记录下来。
2. `with torch.no_grad(): outputs = run_function(*args)`。
3. `backward` 里恢复输入、RNG/autocast 状态。
4. `with torch.enable_grad(): outputs = run_function(*detached_inputs)`。
5. 对重算 outputs 调用 `torch.autograd.backward`，再返回输入梯度。

non-reentrant 的核心不是一个简单的 `autograd.Function.backward`，而是一套 saved tensor hook 协议：

1. 前向运行真实 function，但 hook 拦截每次 “save for backward”。
2. hook 不保留大激活，只留下 holder/metadata。
3. backward 需要某个 saved tensor 时，unpack holder。
4. unpack 触发 recompute function。
5. recompute 期间另一个 hook 收集被重新 save 的 tensors。
6. 拿到目标 tensor 后返回给原 backward；early stop 可以避免重跑整个 function。

## 本项目实现什么

本项目实现 `use_reentrant=True` 的教学核心，并额外补齐工程上会马上遇到的细节：

- 支持嵌套 `args/kwargs`，例如 `checkpoint(fn, {"x": x}, bias=bias)`。
- 保存/恢复 CPU RNG；如果输入张量在 MPS 上且当前 PyTorch 支持 `torch.mps.get_rng_state`，也保存/恢复 MPS RNG。
- 允许 function 闭包里有 `nn.Module` 参数，反向重算会自然给参数累积梯度。
- 用 `saved_tensors_hooks` 统计普通 forward 与 checkpoint forward 保存的 tensor 数量和字节数。

## 为什么先实现 reentrant

因为它把概念闭环压缩到一个可读文件里：不存激活、保存输入、重算 forward、接回 autograd。non-reentrant 更接近 PyTorch 当前推荐路径，但要讲清 holder、weak ref、metadata check、early stop、嵌套 checkpoint 和 debug traces，适合在理解核心之后再看。

## 主要参考

- PyTorch checkpoint 文档: https://docs.pytorch.org/docs/2.12/checkpoint.html
- PyTorch checkpoint 源码: https://github.com/pytorch/pytorch/blob/main/torch/utils/checkpoint.py
- PyTorch 官方博客: https://pytorch.org/blog/activation-checkpointing-techniques/
