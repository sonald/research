from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


@dataclass
class Evidence:
    source: str
    snippet: str
    confidence: float
    subquestion: str


@dataclass
class ResearchState:
    topic: str
    subquestions: List[str] = field(default_factory=list)
    evidence_bank: List[Evidence] = field(default_factory=list)
    draft_sections: Dict[str, str] = field(default_factory=dict)
    action_history: List[str] = field(default_factory=list)


@dataclass
class ScoreBreakdown:
    coverage: float
    verifiability: float
    consistency: float
    novelty: float
    cost_penalty: float

    @property
    def total(self) -> float:
        return (
            0.30 * self.coverage
            + 0.30 * self.verifiability
            + 0.20 * self.consistency
            + 0.20 * self.novelty
            - 0.15 * self.cost_penalty
        )


class ToolAdapter:
    """Mock adapter. Replace with real search/LLM integrations in production."""

    def plan_subquestions(self, topic: str) -> List[str]:
        return [
            f"{topic} 的核心定义与边界是什么？",
            f"{topic} 的主流技术路线与代表方法有哪些？",
            f"{topic} 在工程落地中的瓶颈与评测指标是什么？",
            f"{topic} 的前沿趋势与未解决问题是什么？",
        ]

    def search(self, query: str, k: int = 3) -> List[Tuple[str, str, float]]:
        return [
            (f"mock://paper/{abs(hash(query + str(i))) % 99999}", f"关于 {query} 的证据片段 {i+1}", 0.55 + 0.1 * i)
            for i in range(k)
        ]

    def summarize(self, subquestion: str, evidences: Sequence[Evidence]) -> str:
        joined = "；".join(e.snippet for e in evidences[:3])
        return f"子问题：{subquestion}\n基于证据：{joined}。\n初步结论：该方向有效但仍需进一步实证。"

    def critique(self, draft_sections: Dict[str, str], evidence_bank: Sequence[Evidence]) -> str:
        if not evidence_bank:
            return "证据不足，需继续检索。"
        if len(draft_sections) < 2:
            return "章节覆盖不足，需补充更多子问题结论。"
        return "当前草稿结构较完整，建议补充反例与局限性分析。"


class ResearchOptimizer:
    def __init__(self, adapter: ToolAdapter, max_iters: int = 8, seed: int = 42):
        self.adapter = adapter
        self.max_iters = max_iters
        self.rng = random.Random(seed)

    def score(self, state: ResearchState) -> ScoreBreakdown:
        coverage = min(1.0, len(state.draft_sections) / max(1, len(state.subquestions)))
        verifiability = min(1.0, sum(e.confidence for e in state.evidence_bank) / max(1, len(state.evidence_bank)))
        consistency = 0.5 + 0.5 * (1.0 if len(state.evidence_bank) >= len(state.draft_sections) else 0.0)
        novelty = min(1.0, 0.3 + 0.1 * len({e.source for e in state.evidence_bank}))
        cost_penalty = min(1.0, len(state.action_history) / 20.0)
        return ScoreBreakdown(coverage, verifiability, consistency, novelty, cost_penalty)

    def candidate_actions(self, state: ResearchState) -> List[str]:
        actions = []
        if not state.subquestions:
            actions.append("plan_subquestions")
        else:
            unsolved = [sq for sq in state.subquestions if sq not in state.draft_sections]
            if unsolved:
                actions.append("search_and_summarize")
            actions.append("critique_and_refine")
        return actions

    def apply(self, action: str, state: ResearchState) -> ResearchState:
        state.action_history.append(action)

        if action == "plan_subquestions":
            state.subquestions = self.adapter.plan_subquestions(state.topic)
            return state

        if action == "search_and_summarize":
            unsolved = [sq for sq in state.subquestions if sq not in state.draft_sections]
            if not unsolved:
                return state
            sq = self.rng.choice(unsolved)
            raw_hits = self.adapter.search(sq, k=3)
            evidences = [Evidence(source=s, snippet=snip, confidence=conf, subquestion=sq) for s, snip, conf in raw_hits]
            state.evidence_bank.extend(evidences)
            section = self.adapter.summarize(sq, evidences)
            state.draft_sections[sq] = section
            return state

        if action == "critique_and_refine":
            comment = self.adapter.critique(state.draft_sections, state.evidence_bank)
            state.draft_sections["审稿与改进"] = comment
            return state

        return state

    def run(self, topic: str) -> Tuple[ResearchState, List[Dict[str, float]]]:
        state = ResearchState(topic=topic)
        trace = []

        for _ in range(self.max_iters):
            actions = self.candidate_actions(state)
            if not actions:
                break

            best_action = None
            best_score = float("-inf")

            for action in actions:
                sandbox = ResearchState(
                    topic=state.topic,
                    subquestions=list(state.subquestions),
                    evidence_bank=list(state.evidence_bank),
                    draft_sections=dict(state.draft_sections),
                    action_history=list(state.action_history),
                )
                sandbox = self.apply(action, sandbox)
                s = self.score(sandbox).total
                if s > best_score:
                    best_score = s
                    best_action = action

            state = self.apply(best_action, state)
            sb = self.score(state)
            trace.append(
                {
                    "action": best_action,
                    "total": round(sb.total, 4),
                    "coverage": round(sb.coverage, 4),
                    "verifiability": round(sb.verifiability, 4),
                    "consistency": round(sb.consistency, 4),
                    "novelty": round(sb.novelty, 4),
                    "cost_penalty": round(sb.cost_penalty, 4),
                }
            )

            if sb.total >= 0.82 and len(state.draft_sections) >= max(2, len(state.subquestions) // 2):
                break

        return state, trace


def render_markdown_report(state: ResearchState, trace: List[Dict[str, float]]) -> str:
    lines = [
        f"# AutoResearch (Optimize Anything 版本) 报告\n",
        f"## 研究主题\n{state.topic}\n",
        "## 子问题\n",
    ]
    for i, sq in enumerate(state.subquestions, 1):
        lines.append(f"{i}. {sq}")

    lines.append("\n## 研究内容\n")
    for sq, section in state.draft_sections.items():
        lines.append(f"### {sq}\n{section}\n")

    lines.append("## 优化轨迹\n")
    for i, step in enumerate(trace, 1):
        lines.append(f"{i}. action={step['action']} total={step['total']} coverage={step['coverage']} verifiability={step['verifiability']}")

    lines.append("\n## 证据清单\n")
    for ev in state.evidence_bank:
        lines.append(f"- [{ev.subquestion}] ({ev.source}) conf={ev.confidence:.2f} :: {ev.snippet}")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize Anything style AutoResearch prototype")
    parser.add_argument("--topic", required=True, help="Research topic")
    parser.add_argument("--max-iters", type=int, default=8)
    parser.add_argument("--outdir", default="outputs")
    args = parser.parse_args()

    optimizer = ResearchOptimizer(adapter=ToolAdapter(), max_iters=args.max_iters)
    state, trace = optimizer.run(args.topic)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    report_path = outdir / "report.md"
    trace_path = outdir / "trace.json"

    report_path.write_text(render_markdown_report(state, trace), encoding="utf-8")
    trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote report to: {report_path}")
    print(f"Wrote trace to: {trace_path}")


if __name__ == "__main__":
    main()
