"""
更贴近参考仓库形状的 demo，对应上游仓库：

- microsoft/autogen
  - python/packages/autogen-ext/src/autogen_ext/teams/magentic_one.py
  - user-guide/agentchat-user-guide/magentic-one.html

重写目标：
- 保留 lead Orchestrator + Task Ledger + Progress Ledger 的结构
- 用标准库模拟 WebSurfer / FileSurfer / Coder / ComputerTerminal
- 演示“外层重规划 + 内层推进进度”的双层循环
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class TaskLedger:
    facts: List[str] = field(default_factory=list)
    educated_guesses: List[str] = field(default_factory=list)
    plan: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class ProgressLedger:
    completed: List[str] = field(default_factory=list)
    stalled_steps: int = 0
    current_status: str = "in_progress"


class SpecialistAgent:
    def __init__(self, name: str) -> None:
        self.name = name

    def perform(self, subtask: str) -> Dict[str, str]:
        outputs = {
            "web-surfer": {
                "inspect release issue": "发现发布说明必须包含 breaking changes、migration guide、known issues。"
            },
            "file-surfer": {
                "scan changelog": "CHANGELOG 里有 2 条 feature 和 1 条 breaking change。"
            },
            "coder": {
                "draft release note": "生成了一版 release note，但漏掉 migration guide 链接。"
            },
            "computer-terminal": {
                "validate completeness": "校验失败：缺少 migration guide 链接。"
            },
        }
        result = outputs[self.name].get(subtask, "完成了子任务。")
        return {"agent": self.name, "subtask": subtask, "result": result}


class LeadOrchestrator:
    def __init__(self) -> None:
        self.specialists = {
            "web-surfer": SpecialistAgent("web-surfer"),
            "file-surfer": SpecialistAgent("file-surfer"),
            "coder": SpecialistAgent("coder"),
            "computer-terminal": SpecialistAgent("computer-terminal"),
        }

    def initial_ledger(self) -> TaskLedger:
        return TaskLedger(
            facts=["目标：生成一份可发布的 release note。"],
            educated_guesses=["可能需要查 issue、CHANGELOG 和 migration guide。"],
            plan=[
                {"agent": "web-surfer", "subtask": "inspect release issue"},
                {"agent": "file-surfer", "subtask": "scan changelog"},
                {"agent": "coder", "subtask": "draft release note"},
                {"agent": "computer-terminal", "subtask": "validate completeness"},
            ],
        )

    def replan(self, ledger: TaskLedger) -> None:
        ledger.facts.append("需要把 migration guide 链接补进 release note。")
        ledger.plan = [
            {"agent": "web-surfer", "subtask": "inspect migration guide"},
            {"agent": "coder", "subtask": "revise release note"},
            {"agent": "computer-terminal", "subtask": "revalidate release note"},
        ]

    def run(self, task: str) -> dict:
        task_ledger = self.initial_ledger()
        progress = ProgressLedger()
        trace = [f"orchestrator: start task={task}"]

        outer_loops = 0
        while progress.current_status != "done" and outer_loops < 2:
            outer_loops += 1
            trace.append(f"orchestrator: outer loop {outer_loops}, plan_steps={len(task_ledger.plan)}")

            for step in task_ledger.plan:
                agent = self.specialists.get(step["agent"])
                if agent is None:
                    progress.stalled_steps += 1
                    continue

                if step["subtask"] == "inspect migration guide":
                    result = {
                        "agent": "web-surfer",
                        "subtask": "inspect migration guide",
                        "result": "找到 migration guide 链接：/docs/migrate/v4。",
                    }
                elif step["subtask"] == "revise release note":
                    result = {
                        "agent": "coder",
                        "subtask": "revise release note",
                        "result": "修订 release note，已加入 migration guide 与 known issues。",
                    }
                elif step["subtask"] == "revalidate release note":
                    result = {
                        "agent": "computer-terminal",
                        "subtask": "revalidate release note",
                        "result": "校验通过：release note 结构完整。",
                    }
                else:
                    result = agent.perform(step["subtask"])

                progress.completed.append(f"{result['agent']}::{result['subtask']}")
                trace.append(f"{result['agent']}: {result['result']}")

                if step["subtask"] == "validate completeness":
                    progress.stalled_steps += 1
                    trace.append("orchestrator: progress stalled, update Task Ledger and replan")
                    self.replan(task_ledger)
                    break
                if step["subtask"] == "revalidate release note":
                    progress.current_status = "done"
                    break

        return {
            "pattern": "Orchestrator-Subagent",
            "style": "AutoGen Magentic-One",
            "task": task,
            "outer_loops": outer_loops,
            "task_ledger": {
                "facts": task_ledger.facts,
                "educated_guesses": task_ledger.educated_guesses,
                "remaining_plan": task_ledger.plan,
            },
            "progress_ledger": {
                "completed": progress.completed,
                "stalled_steps": progress.stalled_steps,
                "status": progress.current_status,
            },
            "trace": trace,
        }


def run_demo() -> dict:
    return LeadOrchestrator().run("Produce a release note for v4.0")


def main() -> None:
    result = run_demo()
    print("=== Magentic-One Style Orchestrator Demo ===")
    for line in result["trace"]:
        print(line)
    print("\nprogress ledger:")
    print(result["progress_ledger"])


if __name__ == "__main__":
    main()
