"""
更贴近参考仓库形状的 demo，对应上游仓库：

- langchain-ai/langgraph
  - libs/langgraph/langgraph/graph/state.py
  - docs.langchain.com/oss/python/langgraph
  - docs.langchain.com/oss/python/langgraph/use-graph-api

重写目标：
- 保留 StateGraph / compile / invoke / MessagesState 的心智模型
- 用标准库模拟“节点读取 shared state，返回 partial state”
- 演示 messages 作为共享 state channel 的累计更新
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple


State = Dict[str, object]
NodeFn = Callable[[State], State]


@dataclass
class PseudoCompiledGraph:
    nodes: Dict[str, NodeFn]
    edges: Dict[str, str]
    trace: List[str] = field(default_factory=list)

    def _merge(self, state: State, update: State) -> State:
        merged = dict(state)
        for key, value in update.items():
            if key == "messages":
                merged[key] = list(merged.get(key, [])) + list(value)
            else:
                merged[key] = value
        return merged

    def invoke(self, initial_state: State) -> State:
        current = self.edges["START"]
        state = dict(initial_state)
        while current != "END":
            update = self.nodes[current](state)
            self.trace.append(f"node={current} update_keys={list(update.keys())}")
            state = self._merge(state, update)
            current = self.edges[current]
        return state


@dataclass
class PseudoStateGraph:
    state_schema: str
    nodes: Dict[str, NodeFn] = field(default_factory=dict)
    edges: Dict[str, str] = field(default_factory=dict)

    def add_node(self, name: str, fn: NodeFn) -> None:
        self.nodes[name] = fn

    def add_edge(self, source: str, target: str) -> None:
        self.edges[source] = target

    def compile(self) -> PseudoCompiledGraph:
        return PseudoCompiledGraph(nodes=self.nodes, edges=self.edges)


def planner_node(state: State) -> State:
    topic = state["messages"][0]["content"]
    return {
        "outline": f"研究 {topic} 的目标用户、痛点与差异化价值。",
        "messages": [{"role": "ai", "content": "planner: outline ready"}],
    }


def research_node(state: State) -> State:
    return {
        "evidence": [
            "目标用户是增长团队",
            "核心痛点是数据口径不一致",
            "差异化价值是可审计的指标定义",
        ],
        "messages": [{"role": "ai", "content": "research: evidence gathered"}],
    }


def answer_node(state: State) -> State:
    answer = (
        f"{state['outline']} "
        f"建议主叙事：{state['evidence'][2]}，并用 {state['evidence'][1]} 作为对比。"
    )
    return {
        "answer": answer,
        "messages": [{"role": "ai", "content": "answer: final draft ready"}],
    }


def run_demo() -> dict:
    builder = PseudoStateGraph("MessagesState")
    builder.add_node("planner_node", planner_node)
    builder.add_node("research_node", research_node)
    builder.add_node("answer_node", answer_node)
    builder.add_edge("START", "planner_node")
    builder.add_edge("planner_node", "research_node")
    builder.add_edge("research_node", "answer_node")
    builder.add_edge("answer_node", "END")

    graph = builder.compile()
    result = graph.invoke({"messages": [{"role": "user", "content": "analytics launch"}]})
    return {
        "pattern": "Shared State",
        "style": "LangGraph StateGraph / MessagesState",
        "final_state": result,
        "trace": graph.trace,
    }


def main() -> None:
    result = run_demo()
    print("=== LangGraph StateGraph Style Demo ===")
    for line in result["trace"]:
        print(line)
    print("\nfinal answer:")
    print(result["final_state"]["answer"])


if __name__ == "__main__":
    main()
