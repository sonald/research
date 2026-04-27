"""
更贴近参考仓库形状的 demo，对应上游仓库：

- camel-ai/camel
  - camel/societies/workforce/workforce.py
  - docs.camel-ai.org/key_modules/workforce

重写目标：
- 保留 Workforce / Coordinator / Task Planner / Dynamic Worker 的结构
- 用标准库模拟 persistent workers 与动态扩编
- 演示“团队长期存在，但必要时可以增补新 worker”
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List


@dataclass
class Task:
    name: str
    specialty: str
    retries: int = 0


@dataclass
class WorkerNode:
    name: str
    specialties: List[str]
    history: List[str] = field(default_factory=list)

    def can_handle(self, task: Task) -> bool:
        return task.specialty in self.specialties

    def execute(self, task: Task) -> Dict[str, str]:
        self.history.append(task.name)
        return {
            "worker": self.name,
            "task": task.name,
            "specialty": task.specialty,
            "result": f"{self.name} 完成了 {task.name}",
        }


class TaskPlannerAgent:
    def decompose(self, main_task: str) -> Deque[Task]:
        return deque(
            [
                Task("research-competitors", "research"),
                Task("draft-deck", "writing"),
                Task("design-one-pager", "design"),
            ]
        )


class CoordinatorAgent:
    def assign(self, task: Task, workers: List[WorkerNode]) -> WorkerNode | None:
        for worker in workers:
            if worker.can_handle(task):
                return worker
        return None


class Workforce:
    def __init__(self, description: str) -> None:
        self.description = description
        self.task_planner = TaskPlannerAgent()
        self.coordinator = CoordinatorAgent()
        self.children = [
            WorkerNode("researcher", ["research"]),
            WorkerNode("writer", ["writing"]),
        ]
        self.pending_tasks: Deque[Task] = deque()
        self.completed_tasks: List[Dict[str, str]] = []
        self.dynamic_workers: List[WorkerNode] = []
        self.trace: List[str] = []

    def _create_dynamic_worker(self, specialty: str) -> WorkerNode:
        worker = WorkerNode(name=f"dynamic-{specialty}-worker", specialties=[specialty])
        self.dynamic_workers.append(worker)
        self.children.append(worker)
        self.trace.append(f"workforce: create dynamic worker for specialty={specialty}")
        return worker

    def kickoff(self, main_task: str) -> dict:
        self.pending_tasks = self.task_planner.decompose(main_task)
        self.trace.append(f"task_planner: decomposed {main_task} into {len(self.pending_tasks)} tasks")

        while self.pending_tasks:
            task = self.pending_tasks.popleft()
            worker = self.coordinator.assign(task, self.children)
            if worker is None:
                worker = self._create_dynamic_worker(task.specialty)

            result = worker.execute(task)
            self.completed_tasks.append(result)
            self.trace.append(f"coordinator: assign {task.name} -> {worker.name}")

        return {
            "pattern": "Agent Teams",
            "style": "CAMEL Workforce",
            "description": self.description,
            "completed_tasks": self.completed_tasks,
            "dynamic_workers": [worker.name for worker in self.dynamic_workers],
            "worker_histories": {worker.name: list(worker.history) for worker in self.children},
            "trace": self.trace,
        }


def run_demo() -> dict:
    workforce = Workforce(description="A workforce system using specialized agents")
    return workforce.kickoff("Prepare an investor update package")


def main() -> None:
    result = run_demo()
    print("=== CAMEL Workforce Style Demo ===")
    for line in result["trace"]:
        print(line)
    print("\ndynamic workers:")
    print(result["dynamic_workers"])


if __name__ == "__main__":
    main()
