"""
更贴近参考仓库形状的 demo，对应上游仓库：

- crewAIInc/crewAI-examples
  - flows/self_evaluation_loop_flow/README.md
  - flows/self_evaluation_loop_flow/src/self_evaluation_loop_flow/main.py

重写目标：
- 保留 Self Evaluation Loop Flow 的“两套 crew + feedback 重试”结构
- 用标准库模拟 `ShakespeareanXPostCrew` 与 `XPostReviewCrew`
- 保留 maximum retry limit
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


def contains_emoji(text: str) -> bool:
    emoji_markers = {"😀", "😂", "🔥", "✨", "🚀"}
    return any(marker in text for marker in emoji_markers)


@dataclass
class ReviewOutcome:
    valid: bool
    feedback: List[str]


class ShakespeareanXPostCrew:
    name = "ShakespeareanXPostCrew"

    def kickoff(self, topic: str, feedback: List[str], attempt: int) -> str:
        if attempt == 1:
            return f"O {topic}, thou shiny wonder of tomorrow! 😀 Behold this merry contraption that shall let every dreamer ride above the traffic with comic delight."

        post = f"O {topic}, thou art a wondrous carriage of the air; no emoji attends thee, and thy promise fits within one brisk proclamation."
        if any("shorter" in item for item in feedback):
            post = f"O {topic}, thou art a wondrous carriage of the air; no emoji attends thee."
        if any("include shakespearean voice" in item for item in feedback):
            post += " Verily, thou still soundest of the stage."
        return post


class XPostReviewCrew:
    name = "XPostReviewCrew"

    def kickoff(self, post: str) -> ReviewOutcome:
        feedback: List[str] = []
        if len(post) > 180:
            feedback.append("make it shorter")
        if contains_emoji(post):
            feedback.append("remove emoji")
        if "thou" not in post.lower() and "thee" not in post.lower():
            feedback.append("include shakespearean voice")
        return ReviewOutcome(valid=not feedback, feedback=feedback)


class SelfEvaluationLoopFlow:
    def __init__(self, max_retry_limit: int = 3) -> None:
        self.max_retry_limit = max_retry_limit
        self.generator = ShakespeareanXPostCrew()
        self.reviewer = XPostReviewCrew()

    def kickoff(self, topic: str) -> dict:
        trace: List[str] = []
        feedback: List[str] = []
        last_post = ""

        for attempt in range(1, self.max_retry_limit + 1):
            post = self.generator.kickoff(topic=topic, feedback=feedback, attempt=attempt)
            review = self.reviewer.kickoff(post)
            trace.append(
                f"attempt {attempt}: {self.generator.name} -> {self.reviewer.name}, "
                f"valid={review.valid}, feedback={review.feedback}"
            )
            last_post = post
            if review.valid:
                return {
                    "pattern": "Generator-Verifier",
                    "style": "CrewAI Self Evaluation Loop Flow",
                    "topic": topic,
                    "valid": True,
                    "attempts": attempt,
                    "post": post,
                    "trace": trace,
                }
            feedback = review.feedback

        return {
            "pattern": "Generator-Verifier",
            "style": "CrewAI Self Evaluation Loop Flow",
            "topic": topic,
            "valid": False,
            "attempts": self.max_retry_limit,
            "post": last_post,
            "trace": trace,
        }


def run_demo() -> dict:
    return SelfEvaluationLoopFlow(max_retry_limit=3).kickoff("Flying cars")


def main() -> None:
    result = run_demo()
    print("=== CrewAI Self Evaluation Loop Demo ===")
    for line in result["trace"]:
        print(line)
    print("\nfinal post:")
    print(result["post"])
    print(f"\nvalid={result['valid']} attempts={result['attempts']}")


if __name__ == "__main__":
    main()
