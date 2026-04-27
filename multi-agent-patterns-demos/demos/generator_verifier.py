"""
最小标准库 demo，对应上游仓库：

- microsoft/autogen
  - Reflection design pattern
  - https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/reflection.html

这不是 AutoGen 源码复刻，而是把它的核心协调机制重写成零依赖版本：
- generator 先产出
- verifier 按 criteria 审核
- feedback 回流给 generator
- 用 max_iterations 防止无限循环
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


REQUIRED_CRITERIA = {
    "pricing": "明确说明套餐价格",
    "timeline": "明确说明回复时限",
    "owner": "明确说明后续负责人",
}


@dataclass
class Draft:
    subject: str
    sections: List[str] = field(default_factory=list)

    def covers(self, criterion: str) -> bool:
        return any(section.startswith(f"{criterion}:") for section in self.sections)


@dataclass
class ReviewResult:
    accepted: bool
    feedback: List[str]
    mode: str


class Generator:
    def initial_draft(self) -> Draft:
        return Draft(subject="客户支持邮件", sections=["greeting: 感谢你的来信，我们已经收到问题。"])

    def revise(self, draft: Draft, feedback: List[str]) -> Draft:
        revised = Draft(subject=draft.subject, sections=list(draft.sections))
        for criterion in REQUIRED_CRITERIA:
            if any(criterion in item for item in feedback) and not revised.covers(criterion):
                revised.sections.append(f"{criterion}: {REQUIRED_CRITERIA[criterion]}")
                break
        return revised


class Verifier:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    def review(self, draft: Draft) -> ReviewResult:
        if self.mode == "vague":
            accepted = bool(draft.sections) and "感谢" in "".join(draft.sections)
            feedback = [] if accepted else ["请写得更像一封完整邮件。"]
            return ReviewResult(accepted=accepted, feedback=feedback, mode=self.mode)

        missing = [
            f"missing:{criterion}:{description}"
            for criterion, description in REQUIRED_CRITERIA.items()
            if not draft.covers(criterion)
        ]
        return ReviewResult(accepted=not missing, feedback=missing, mode=self.mode)


def run_review_loop(verifier_mode: str, max_iterations: int) -> dict:
    generator = Generator()
    verifier = Verifier(verifier_mode)
    draft = generator.initial_draft()
    trace: List[str] = []

    for attempt in range(1, max_iterations + 1):
        review = verifier.review(draft)
        trace.append(
            f"attempt {attempt}: mode={verifier_mode}, sections={len(draft.sections)}, accepted={review.accepted}"
        )
        if review.accepted:
            return {
                "accepted": True,
                "attempts": attempt,
                "sections": list(draft.sections),
                "trace": trace,
            }
        draft = generator.revise(draft, review.feedback)

    return {
        "accepted": False,
        "attempts": max_iterations,
        "sections": list(draft.sections),
        "trace": trace,
    }


def run_demo() -> dict:
    vague = run_review_loop(verifier_mode="vague", max_iterations=2)
    strict = run_review_loop(verifier_mode="strict", max_iterations=4)
    capped = run_review_loop(verifier_mode="strict", max_iterations=2)
    return {
        "pattern": "Generator-Verifier",
        "criteria": REQUIRED_CRITERIA,
        "vague_verifier": vague,
        "strict_verifier": strict,
        "capped_strict_verifier": capped,
    }


def main() -> None:
    result = run_demo()
    print("=== Generator-Verifier Demo ===")
    print("显式 criteria:")
    for key, value in result["criteria"].items():
        print(f"- {key}: {value}")

    print("\n[1] criteria 模糊时，verifier 会过早放行")
    for line in result["vague_verifier"]["trace"]:
        print(line)
    print(f"final sections: {result['vague_verifier']['sections']}")

    print("\n[2] criteria 明确时，generator 会按反馈逐步补齐")
    for line in result["strict_verifier"]["trace"]:
        print(line)
    print(f"final sections: {result['strict_verifier']['sections']}")

    print("\n[3] 即使未收敛，也要靠最大迭代数终止")
    for line in result["capped_strict_verifier"]["trace"]:
        print(line)
    print(f"final sections: {result['capped_strict_verifier']['sections']}")


if __name__ == "__main__":
    main()
