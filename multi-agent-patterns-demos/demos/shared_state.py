"""
最小标准库 demo，对应上游仓库：

- langchain-ai/langgraph
  - StateGraph / shared state
  - https://docs.langchain.com/oss/python/langgraph/graph-api
  - https://docs.langchain.com/oss/python/langgraph/use-graph-api

这不是 LangGraph 源码复刻，而是把 shared state coordination 的核心机制
重写成零依赖版本：
- 多 agent 读写共享 store
- version 递增
- termination condition 防止 reactive loop
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, List


@dataclass
class SharedStore:
    lock: Lock = field(default_factory=Lock)
    version: int = 0
    findings: List[Dict[str, str]] = field(default_factory=list)
    done: bool = False
    idle_rounds: int = 0

    def snapshot(self) -> Dict[str, object]:
        with self.lock:
            return {
                "version": self.version,
                "findings": list(self.findings),
                "done": self.done,
                "idle_rounds": self.idle_rounds,
            }

    def append_finding(self, agent: str, category: str, detail: str) -> bool:
        with self.lock:
            exists = any(
                item["category"] == category and item["detail"] == detail for item in self.findings
            )
            if exists:
                return False
            self.version += 1
            self.findings.append(
                {
                    "version": str(self.version),
                    "agent": agent,
                    "category": category,
                    "detail": detail,
                }
            )
            self.idle_rounds = 0
            return True

    def mark_idle_round(self) -> None:
        with self.lock:
            self.idle_rounds += 1

    def mark_done(self) -> None:
        with self.lock:
            self.done = True


class ResearchAgent:
    def __init__(self, name: str) -> None:
        self.name = name

    def act(self, store: SharedStore) -> bool:
        snapshot = store.snapshot()
        categories = {item["category"] for item in snapshot["findings"]}

        if self.name == "academic" and "academia" not in categories:
            return store.append_finding(self.name, "academia", "发现 Lab42 的 Rivera 教授是关键研究者")
        if self.name == "industry" and "academia" in categories and "industry" not in categories:
            return store.append_finding(self.name, "industry", "Rivera 参与孵化了 NovaForge 公司")
        if self.name == "patent" and "industry" in categories and "patent" not in categories:
            return store.append_finding(self.name, "patent", "检索到 NovaForge 关联专利 CN-2048-AGENT")
        if self.name == "news" and "industry" in categories and "news" not in categories:
            return store.append_finding(self.name, "news", "TechDaily 报道 NovaForge 获得新一轮融资")
        return False


def termination_monitor(store: SharedStore, quiet_round_limit: int = 2) -> str | None:
    snapshot = store.snapshot()
    categories = {item["category"] for item in snapshot["findings"]}
    if {"academia", "industry", "patent", "news"}.issubset(categories):
        store.mark_done()
        return "目标类别已经齐备，终止协作。"
    if snapshot["idle_rounds"] >= quiet_round_limit:
        store.mark_done()
        return "连续多个空转周期没有新增发现，终止协作。"
    return None


def run_demo(max_cycles: int = 8) -> dict:
    store = SharedStore()
    agents = [
        ResearchAgent("academic"),
        ResearchAgent("industry"),
        ResearchAgent("patent"),
        ResearchAgent("news"),
    ]
    trace: List[str] = []

    for cycle in range(1, max_cycles + 1):
        wrote = False
        trace.append(f"cycle {cycle}: version={store.snapshot()['version']}")
        for agent in agents:
            if agent.act(store):
                latest = store.snapshot()["findings"][-1]
                trace.append(
                    f"  {agent.name}: wrote version={latest['version']} category={latest['category']}"
                )
                wrote = True
            else:
                trace.append(f"  {agent.name}: no-op")

        if not wrote:
            store.mark_idle_round()
            trace.append(f"  monitor: idle_rounds={store.snapshot()['idle_rounds']}")

        stop_reason = termination_monitor(store)
        if stop_reason is not None:
            trace.append(f"  termination: {stop_reason}")
            break

    return {
        "pattern": "Shared State",
        "version": store.snapshot()["version"],
        "findings": store.snapshot()["findings"],
        "done": store.snapshot()["done"],
        "trace": trace,
    }


def main() -> None:
    result = run_demo()
    print("=== Shared State Demo ===")
    print(f"done={result['done']} version={result['version']}")
    for line in result["trace"]:
        print(line)
    print("\nfindings:")
    for item in result["findings"]:
        print(f"- v{item['version']} {item['category']}: {item['detail']} ({item['agent']})")


if __name__ == "__main__":
    main()
