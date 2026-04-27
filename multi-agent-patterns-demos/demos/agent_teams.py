"""
最小标准库 demo，对应上游仓库：

- FoundationAgents/MetaGPT
  - Team / Role 协作模型
  - https://docs.deepwisdom.ai/main/en/guide/get_started/quickstart.html
  - https://docs.deepwisdom.ai/main/en/guide/tutorials/multi_agent_101.html

这不是 MetaGPT 源码复刻，而是把它的 team + persistent roles 思路
重写成零依赖版本：
- worker 持续存在
- worker 保留本地记忆
- coordinator 只做较粗粒度分派
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class TeamTask:
    title: str
    partition: str
    request: str


@dataclass
class Worker:
    name: str
    partition: str
    specialty: str
    memory: List[str] = field(default_factory=list)
    handled_count: int = 0

    def handle(self, task: TeamTask) -> Dict[str, str]:
        self.handled_count += 1
        prior = self.memory[-1] if self.memory else "首次接手这个分区"
        note = (
            f"{self.name} 处理 {task.title}；"
            f"专长={self.specialty}；"
            f"上一次该分区经验={prior}"
        )
        self.memory.append(task.title)
        return {
            "worker": self.name,
            "partition": self.partition,
            "task": task.title,
            "note": note,
        }


class TeamCoordinator:
    def __init__(self) -> None:
        self.workers = {
            "frontend": Worker("ui-lead", "frontend", "页面与交互"),
            "backend": Worker("api-lead", "backend", "服务接口与可靠性"),
            "design-system": Worker("design-owner", "design-system", "组件与设计令牌"),
        }

    def run_round(self, tasks: List[TeamTask]) -> Dict[str, List[Dict[str, str]]]:
        trace: List[str] = []
        results: List[Dict[str, str]] = []
        conflicts: List[Dict[str, str]] = []
        busy_partitions = set()

        for task in tasks:
            if task.partition in busy_partitions:
                conflicts.append(
                    {
                        "task": task.title,
                        "partition": task.partition,
                        "reason": "同一轮内共享分区冲突，需要分区或串行化。",
                    }
                )
                trace.append(f"coordinator: defer {task.title} because {task.partition} is already busy")
                continue

            busy_partitions.add(task.partition)
            worker = self.workers[task.partition]
            results.append(worker.handle(task))
            trace.append(f"coordinator: assign {task.title} -> {worker.name}")

        return {"trace": trace, "results": results, "conflicts": conflicts}

    def run(self) -> dict:
        round_one = [
            TeamTask("landing-refresh", "frontend", "刷新首页转化区"),
            TeamTask("auth-hardening", "backend", "加固登录接口"),
            TeamTask("token-cleanup", "design-system", "清理重复颜色令牌"),
        ]
        round_two = [
            TeamTask("checkout-copy-tuning", "frontend", "优化结算页文案"),
            TeamTask("billing-retry-policy", "backend", "更新重试策略"),
            TeamTask("checkout-hotfix", "frontend", "修复结算页样式抖动"),
        ]

        first = self.run_round(round_one)
        second = self.run_round(round_two)

        worker_state = {
            name: {"handled_count": worker.handled_count, "memory": list(worker.memory)}
            for name, worker in self.workers.items()
        }
        return {
            "pattern": "Agent Teams",
            "round_one": first,
            "round_two": second,
            "worker_state": worker_state,
        }


def run_demo() -> dict:
    return TeamCoordinator().run()


def main() -> None:
    result = run_demo()
    print("=== Agent Teams Demo ===")
    print("[round one]")
    for line in result["round_one"]["trace"]:
        print(line)
    for item in result["round_one"]["results"]:
        print(f"- {item['worker']} -> {item['note']}")

    print("\n[round two]")
    for line in result["round_two"]["trace"]:
        print(line)
    for item in result["round_two"]["results"]:
        print(f"- {item['worker']} -> {item['note']}")
    for item in result["round_two"]["conflicts"]:
        print(f"- conflict: {item['task']} / {item['reason']}")

    print("\nworker memory:")
    for worker, state in result["worker_state"].items():
        print(f"- {worker}: handled={state['handled_count']}, memory={state['memory']}")


if __name__ == "__main__":
    main()
