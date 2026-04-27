"""
更贴近参考仓库形状的 demo，对应上游仓库：

- microsoft/autogen
  - core design pattern: group chat
  - python/packages/autogen-core/src/autogen_core/_routed_agent.py
  - user-guide/core-user-guide/design-patterns/group-chat.html

重写目标：
- 保留“所有参与者订阅同一个 group topic”的总线风格
- 用标准库模拟 Group Chat Manager 和 RequestToSpeak
- 演示 publish / subscribe 在群聊模式下的路由方式
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, DefaultDict, Dict, List


Handler = Callable[["TopicBus", str, Dict[str, str]], None]


@dataclass
class TopicBus:
    subscribers: DefaultDict[str, List[Handler]]
    trace: List[str]

    def subscribe(self, topic: str, handler: Handler) -> None:
        self.subscribers[topic].append(handler)
        self.trace.append(f"subscribe: {handler.__name__} -> {topic}")

    def publish(self, topic: str, payload: Dict[str, str]) -> None:
        self.trace.append(f"publish: {topic} payload={payload}")
        for handler in list(self.subscribers.get(topic, [])):
            handler(self, topic, payload)


class GroupChatRuntime:
    def __init__(self) -> None:
        self.bus = TopicBus(subscribers=defaultdict(list), trace=[])
        self.group_topic = "group_chat"
        self.turn_order = ["planner", "writer", "editor", "user"]
        self.turn_index = 0
        self.final_transcript: List[str] = []

        self.bus.subscribe(self.group_topic, self.group_chat_manager)
        self.bus.subscribe("planner", self.planner_agent)
        self.bus.subscribe("writer", self.writer_agent)
        self.bus.subscribe("editor", self.editor_agent)
        self.bus.subscribe("user", self.user_agent)

    def group_chat_manager(self, bus: TopicBus, topic: str, payload: Dict[str, str]) -> None:
        if payload["kind"] == "task":
            bus.publish(self.turn_order[self.turn_index], {"kind": "request_to_speak", "task": payload["task"]})
            return

        if payload["kind"] == "group_message":
            self.final_transcript.append(f"{payload['speaker']}: {payload['content']}")
            if payload["speaker"] == "user":
                return
            self.turn_index += 1
            next_speaker = self.turn_order[self.turn_index]
            bus.publish(next_speaker, {"kind": "request_to_speak", "task": payload["task"]})

    def planner_agent(self, bus: TopicBus, topic: str, payload: Dict[str, str]) -> None:
        bus.publish(
            self.group_topic,
            {
                "kind": "group_message",
                "speaker": "planner",
                "task": payload["task"],
                "content": "先产出一个一句话标题，再由 editor 压缩措辞。",
            },
        )

    def writer_agent(self, bus: TopicBus, topic: str, payload: Dict[str, str]) -> None:
        bus.publish(
            self.group_topic,
            {
                "kind": "group_message",
                "speaker": "writer",
                "task": payload["task"],
                "content": "Ship analytics your whole team can trust.",
            },
        )

    def editor_agent(self, bus: TopicBus, topic: str, payload: Dict[str, str]) -> None:
        bus.publish(
            self.group_topic,
            {
                "kind": "group_message",
                "speaker": "editor",
                "task": payload["task"],
                "content": "Trustworthy analytics for the whole team.",
            },
        )

    def user_agent(self, bus: TopicBus, topic: str, payload: Dict[str, str]) -> None:
        bus.publish(
            self.group_topic,
            {
                "kind": "group_message",
                "speaker": "user",
                "task": payload["task"],
                "content": "Approved.",
            },
        )

    def run(self, task: str) -> dict:
        self.bus.publish(self.group_topic, {"kind": "task", "task": task})
        return {
            "pattern": "Message Bus",
            "style": "AutoGen Group Chat",
            "group_topic": self.group_topic,
            "participants": list(self.turn_order),
            "transcript": self.final_transcript,
            "trace": self.bus.trace,
        }


def run_demo() -> dict:
    return GroupChatRuntime().run("Create a landing-page headline")


def main() -> None:
    result = run_demo()
    print("=== AutoGen Group Chat Style Demo ===")
    for line in result["trace"]:
        print(line)
    print("\ntranscript:")
    for line in result["transcript"]:
        print(f"- {line}")


if __name__ == "__main__":
    main()
