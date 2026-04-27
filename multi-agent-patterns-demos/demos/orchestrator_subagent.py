"""
最小标准库 demo，对应上游仓库：

- crewAIInc/crewAI
  - hierarchical process
  - https://docs.crewai.com/en/learn/hierarchical-process
  - https://docs.crewai.com/en/concepts/processes

这不是 CrewAI 源码复刻，而是把它的 manager -> delegated workers 结构
重写成零依赖版本：
- orchestrator 负责拆分任务
- subagent 负责短生命周期执行
- orchestrator 汇总结果
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import count
from typing import Dict, List


_CONTEXT_IDS = count(1)


@dataclass(frozen=True)
class SubTask:
    name: str
    goal: str


def run_subagent(subtask: SubTask) -> Dict[str, str]:
    context_id = f"subagent-{next(_CONTEXT_IDS)}"
    if subtask.name == "inventory":
        result = "发现 3 个可独立迁移的服务：billing、auth、search。"
    elif subtask.name == "risk":
        result = "主要风险是共享数据库 schema 与认证回调兼容性。"
    else:
        result = "建议迁移顺序：search -> billing -> auth，并在最后做集成回归。"

    return {
        "task": subtask.name,
        "goal": subtask.goal,
        "context_id": context_id,
        "result": result,
        "context_policy": "ephemeral",
    }


class Orchestrator:
    def plan(self) -> List[SubTask]:
        return [
            SubTask(name="inventory", goal="盘点可独立迁移的服务边界"),
            SubTask(name="risk", goal="找出跨服务耦合与回滚风险"),
            SubTask(name="rollout", goal="给出可执行的迁移顺序"),
        ]

    def run(self) -> dict:
        tasks = self.plan()
        trace = ["orchestrator: 拆分任务并派发 3 个边界清晰的子任务"]
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(run_subagent, tasks))

        trace.extend(
            f"orchestrator: 收到 {item['context_id']} 的结果，task={item['task']}, policy={item['context_policy']}"
            for item in results
        )

        summary = " ".join(item["result"] for item in results)
        return {
            "pattern": "Orchestrator-Subagent",
            "tasks": [task.__dict__ for task in tasks],
            "subagent_results": results,
            "summary": summary,
            "trace": trace,
        }


def run_demo() -> dict:
    return Orchestrator().run()


def main() -> None:
    result = run_demo()
    print("=== Orchestrator-Subagent Demo ===")
    for line in result["trace"]:
        print(line)
    print("\nsubtasks:")
    for item in result["tasks"]:
        print(f"- {item['name']}: {item['goal']}")
    print("\nresults:")
    for item in result["subagent_results"]:
        print(f"- {item['context_id']} -> {item['result']}")
    print("\nfinal synthesis:")
    print(result["summary"])


if __name__ == "__main__":
    main()
