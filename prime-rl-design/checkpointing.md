# prime-rl 检查点（Checkpointing）机制设计文档

## 1. 高层设计概述

在长时间的深度学习训练过程中，检查点（Checkpointing）机制至关重要。它能够定期保存模型的训练状态，从而防止因意外中断（如断电、程序崩溃）导致数小时甚至数天的训练成果丢失。一个设计良好的检查点系统应该能够保存和恢复训练所需的所有关键信息，并能轻松集成到现有的训练流程中。

对于 `prime-rl` 这样的强化学习框架，一个完整的检查点不仅应包含模型权重，还应包括优化器状态、训练迭代次数、学习率调度器状态以及其他重要的训练指标（如累计奖励、胜率等）。

为此，我们提出一个灵活且可扩展的检查点管理器 `CheckpointManager`，其核心职责是：

*   **状态聚合**: 从训练流程中收集所有需要持久化的状态信息，包括模型（`model`）、优化器（`optimizer`）、以及一个包含任意训练元数据（如 `epoch`, `steps`, `reward`）的字典。
*   **定期保存**: 根据预设的策略（如每 N 步或每 K 小时）自动触发保存操作。
*   **版本管理**: 在保存目录中维护多个检查点版本，并能识别出“最新”的一个，方便恢复。
*   **状态恢复**: 能够从指定的检查点文件中加载所有状态，并将它们准确地恢复到训练流程的各个组件中。

## 2. 检查点保存与加载流程图

下面是检查点机制的工作流程图，使用 Mermaid 语法绘制：

### 保存流程 (`save`)
```mermaid
graph TD
    A[训练循环] -->|达到保存条件| B(CheckpointManager.save);
    B --> C{1. 创建状态字典};
    C -->|包含模型权重| D[model.state_dict()];
    C -->|包含优化器状态| E[optimizer.state_dict()];
    C -->|包含训练元数据| F[{"epoch": ..., "steps": ...}];
    subgraph "状态聚合"
        D
        E
        F
    end
    C --> G{2. 构造文件名};
    G -->|如: ckpt_epoch_10_steps_5000.pt| H(torch.save);
    H -->|写入磁盘| I[检查点文件.pt];
    B -->|返回路径| A;
```

### 加载流程 (`load`)
```mermaid
graph TD
    J[训练脚本启动] -->|指定恢复路径或自动查找| K(CheckpointManager.load);
    K --> L{1. torch.load};
    L -->|从磁盘读取| M[检查点文件.pt];
    L --> N{2. 恢复状态};
    N -->|加载模型权重| O[model.load_state_dict()];
    N -->|加载优化器状态| P[optimizer.load_state_dict()];
    N -->|恢复训练元数据| Q[返回 metadata 字典];
    subgraph "状态分发"
        O
        P
        Q
    end
    K -->|返回 metadata| J;
    J -->|继续训练| R[训练循环];
```

## 3. 核心代码设计与分析

我们将设计一个 `CheckpointManager` 类来封装检查点逻辑。

```python
import torch
import os
import glob
from typing import Dict, Any, Optional

class CheckpointManager:
    """
    一个负责处理模型训练检查点保存和加载的管理器。
    """
    def __init__(self, save_dir: str):
        """
        初始化 CheckpointManager。

        Args:
            save_dir (str): 保存检查点的目录。
        """
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def save(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, metadata: Dict[str, Any]) -> str:
        """
        保存一个检查点。

        Args:
            model (torch.nn.Module): 要保存的模型。
            optimizer (torch.optim.Optimizer): 要保存的优化器。
            metadata (Dict[str, Any]): 需要保存的训练元数据，例如 {'epoch': 1, 'steps': 1000}。

        Returns:
            str: 保存的检查点文件的路径。
        """
        epoch = metadata.get('epoch', 0)
        steps = metadata.get('steps', 0)

        # 1. 聚合状态
        state = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metadata': metadata
        }

        # 2. 构造文件名并保存
        filename = f"checkpoint_epoch_{epoch}_steps_{steps}.pt"
        save_path = os.path.join(self.save_dir, filename)
        torch.save(state, save_path)

        print(f"检查点已保存至: {save_path}")
        return save_path

    def load(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """
        从一个检查点加载状态。

        Args:
            model (torch.nn.Module): 需要恢复状态的模型。
            optimizer (torch.optim.Optimizer): 需要恢复状态的优化器。
            checkpoint_path (Optional[str]): 检查点文件的具体路径。如果为 None，则自动加载最新的检查点。

        Returns:
            Dict[str, Any]: 恢复的训练元数据。
        """
        if checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint()

        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            print("未找到检查点，将从头开始训练。")
            return {}

        # 1. 从磁盘加载
        state = torch.load(checkpoint_path)

        # 2. 恢复状态
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])

        print(f"已从 {checkpoint_path} 成功加载检查点。")
        return state.get('metadata', {})

    def get_latest_checkpoint(self) -> Optional[str]:
        """
        在保存目录中查找最新的检查点文件。
        查找策略是基于文件名中的 'steps'。
        """
        checkpoints = glob.glob(os.path.join(self.save_dir, "checkpoint_*.pt"))
        if not checkpoints:
            return None

        # 通过解析文件名中的 steps 来找到最新的检查点
        latest_ckpt = max(checkpoints, key=lambda p: int(p.split('_')[-1].split('.')[0]))
        return latest_ckpt
```

## 4. 代码示例

下面的示例展示了如何在典型的训练循环中使用 `CheckpointManager`。

```python
import torch
import torch.nn as nn
import torch.optim as optim
import os

# --- 1. 定义模型和优化器 ---
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)

# --- 2. 训练设置 ---
model = SimpleModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
checkpoint_dir = "./training_checkpoints"

# 初始化检查点管理器
checkpoint_manager = CheckpointManager(save_dir=checkpoint_dir)

# --- 3. 尝试从检查点恢复 ---
# 如果存在检查点，这将加载模型和优化器的状态，并返回元数据
metadata = checkpoint_manager.load(model, optimizer)

start_epoch = metadata.get('epoch', 0)
total_steps = metadata.get('steps', 0)

print(f"将从 Epoch {start_epoch}, Step {total_steps} 开始训练...")

# --- 4. 模拟训练循环 ---
num_epochs = 5
for epoch in range(start_epoch, num_epochs):
    for step in range(100): # 模拟每个 epoch 100步
        # 模拟训练步骤
        inputs = torch.randn(32, 10)
        labels = torch.randint(0, 2, (32,))

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()

        total_steps += 1

        # --- 5. 定期保存检查点 ---
        if total_steps % 200 == 0: # 每 200 步保存一次
            current_metadata = {
                'epoch': epoch + 1,
                'steps': total_steps,
                'current_loss': loss.item()
            }
            checkpoint_manager.save(model, optimizer, current_metadata)

print("训练完成！")

# 验证：检查检查点目录中是否已创建文件
if os.path.exists(checkpoint_dir) and os.listdir(checkpoint_dir):
    print(f"\n检查点文件已成功保存在 '{checkpoint_dir}' 目录中。")
    print("目录内容:", os.listdir(checkpoint_dir))
else:
    print("检查点目录为空或不存在。")
```
**如何运行此示例:**
1.  将 `CheckpointManager` 类和示例代码保存在同一个 Python 文件中。
2.  首次运行脚本时，它会从头开始训练，并在达到指定步数时创建检查点文件。
3.  再次运行同一个脚本时，它会自动检测到最新的检查点，加载状态，并从中断的地方继续训练，而不会从 Epoch 0 重新开始。
