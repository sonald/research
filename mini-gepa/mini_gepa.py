"""
mini_gepa.py

一个“教学版”的 GEPA（Genetic-Pareto）实现。

设计目标：
1) 只保留论文与官方实现中的主干思想：
   - Pareto 前沿候选选择
   - 反思式变异（reflection -> mutation）
   - 基于 minibatch 的迭代优化
   - 候选池持续演化
2) 刻意省略大量工程细节：
   - 复杂边界处理
   - 异常恢复
   - 多组件系统/并行执行/缓存/日志系统等

这份代码适合教学与阅读，不适合直接用于生产。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Callable, Iterable


# ----------------------------
# 一、基础数据结构
# ----------------------------


@dataclass
class Task:
    """一个最小任务定义。

    在真实 GEPA 中，一个 task 通常是一个样本（或多目标评估中的一个目标切片），
    这里我们把它抽象成：
      - name: 任务名
      - requirement: 这个任务希望 prompt 包含的“规则片段”

    教学简化：
      如果 candidate 的 prompt 包含 requirement，则记为成功，否则失败。
    """

    name: str
    requirement: str


@dataclass
class Candidate:
    """候选解（教学版只优化一个文本组件：prompt）。"""

    prompt: str
    # 每个任务上的分数（0 或 1）。
    per_task_scores: dict[str, float] = field(default_factory=dict)

    @property
    def mean_score(self) -> float:
        """平均分，方便展示与最终选择。"""

        if not self.per_task_scores:
            return 0.0
        return sum(self.per_task_scores.values()) / len(self.per_task_scores)


# ----------------------------
# 二、评估 + Pareto 前沿
# ----------------------------


class MiniEvaluator:
    """教学版 evaluator。

    对每个任务进行“是否命中 requirement”的打分，同时输出可反思的 side information。
    """

    def evaluate(self, candidate: Candidate, tasks: Iterable[Task]) -> tuple[dict[str, float], list[str]]:
        """返回：
        - per_task_scores: 每个任务分数
        - side_info: 供反思阶段使用的可读反馈（ASI 的极简替身）
        """

        scores: dict[str, float] = {}
        side_info: list[str] = []

        for task in tasks:
            hit = task.requirement.lower() in candidate.prompt.lower()
            score = 1.0 if hit else 0.0
            scores[task.name] = score

            if hit:
                side_info.append(
                    f"[PASS] task={task.name}: prompt 已覆盖 requirement='{task.requirement}'"
                )
            else:
                side_info.append(
                    f"[FAIL] task={task.name}: 缺少 requirement='{task.requirement}'"
                )

        return scores, side_info


def dominates(a: Candidate, b: Candidate) -> bool:
    """Pareto domination: a 支配 b 当且仅当：
    - 对所有任务，a 分数 >= b
    - 且至少一个任务上严格更高
    """

    keys = a.per_task_scores.keys()
    ge_all = all(a.per_task_scores[k] >= b.per_task_scores[k] for k in keys)
    gt_any = any(a.per_task_scores[k] > b.per_task_scores[k] for k in keys)
    return ge_all and gt_any


def pareto_frontier(candidates: list[Candidate]) -> list[Candidate]:
    """计算候选池的 Pareto 前沿。"""

    frontier: list[Candidate] = []
    for c in candidates:
        dominated_by_someone = False
        for other in candidates:
            if other is c:
                continue
            if dominates(other, c):
                dominated_by_someone = True
                break

        if not dominated_by_someone:
            frontier.append(c)

    return frontier


# ----------------------------
# 三、反思式变异（核心教学点）
# ----------------------------


def reflective_mutation(parent: Candidate, side_info: list[str]) -> Candidate:
    """根据 side information 对 parent 做“目标化”修改。

    真实 GEPA 里这一步由反思模型（LLM）完成：
      - 读 trajectory / 工具输出 / 错误日志
      - 归纳失败原因
      - 生成新的文本组件

    教学简化规则：
      - 解析失败反馈中的 requirement
      - 把缺失 requirement 逐条拼到 prompt 末尾
    """

    prompt = parent.prompt

    for line in side_info:
        if "[FAIL]" in line and "requirement='" in line:
            # 从日志里提取 requirement（简化解析，不做鲁棒处理）
            missing = line.split("requirement='")[1].split("'")[0]
            patch = f"\n- Rule: {missing}."
            if patch not in prompt:
                prompt += patch

    # 返回新候选，分数稍后由 evaluator 填充。
    return Candidate(prompt=prompt)


# ----------------------------
# 四、最小 GEPA 优化循环
# ----------------------------


@dataclass
class MiniGEPAConfig:
    iterations: int = 12
    minibatch_size: int = 3
    seed: int = 7


@dataclass
class MiniGEPAResult:
    best_candidate: Candidate
    pool: list[Candidate]


class MiniGEPA:
    """教学版 GEPA 主循环。"""

    def __init__(self, config: MiniGEPAConfig):
        self.config = config
        self.rng = random.Random(config.seed)
        self.evaluator = MiniEvaluator()

    def optimize(self, seed_prompt: str, train_tasks: list[Task], val_tasks: list[Task]) -> MiniGEPAResult:
        # 1) 初始化种子候选并做一次全量验证评分
        seed = Candidate(prompt=seed_prompt)
        seed.per_task_scores, _ = self.evaluator.evaluate(seed, val_tasks)

        # 候选池（对应官方实现中的 candidate pool + frontier tracking）
        pool: list[Candidate] = [seed]
        seen_prompts = {seed.prompt}

        for step in range(self.config.iterations):
            # 2) 从当前 Pareto 前沿选 parent（简化：随机选一个）
            frontier = pareto_frontier(pool)
            parent = self.rng.choice(frontier)

            # 3) 取一个 minibatch 做“反思数据集”
            minibatch = self.rng.sample(train_tasks, k=self.config.minibatch_size)
            _, side_info = self.evaluator.evaluate(parent, minibatch)

            # 4) 反思式变异，生成 child
            child = reflective_mutation(parent, side_info)

            # 5) 在验证集上全量评估 child
            child.per_task_scores, _ = self.evaluator.evaluate(child, val_tasks)

            # 6) 接受策略（教学简化：只要不比 parent 差就收录）
            if child.mean_score >= parent.mean_score and child.prompt not in seen_prompts:
                pool.append(child)
                seen_prompts.add(child.prompt)

            # 教学输出：便于观察迭代过程
            print(
                f"step={step:02d} | parent={parent.mean_score:.3f} | "
                f"child={child.mean_score:.3f} | pool={len(pool)}"
            )

        # 7) 返回平均分最高者（同分时保留先出现）
        best = max(pool, key=lambda c: c.mean_score)
        return MiniGEPAResult(best_candidate=best, pool=pool)


# ----------------------------
# 五、可直接运行的教学 demo
# ----------------------------


def build_demo_tasks() -> tuple[list[Task], list[Task]]:
    """构造一个可复现的玩具数据集。

    我们故意把任务 requirement 写成“规则短语”，
    让反思变异阶段能把它们拼进 prompt，从而看到分数逐步提升。
    """

    train = [
        Task("math_format", "final answer must be a single integer"),
        Task("show_work", "show concise reasoning before the answer"),
        Task("self_check", "verify the final number once before output"),
        Task("units", "include units when the question asks for physical quantity"),
        Task("bounds", "if uncertain, provide a bounded estimate"),
    ]

    val = list(train)
    return train, val


def main() -> None:
    train_tasks, val_tasks = build_demo_tasks()

    seed_prompt = "You are a helpful assistant for problem solving."

    optimizer = MiniGEPA(MiniGEPAConfig(iterations=10, minibatch_size=3, seed=42))
    result = optimizer.optimize(seed_prompt=seed_prompt, train_tasks=train_tasks, val_tasks=val_tasks)

    print("\n=== Best Candidate ===")
    print(result.best_candidate.prompt)
    print("mean_score=", f"{result.best_candidate.mean_score:.3f}")


if __name__ == "__main__":
    main()
