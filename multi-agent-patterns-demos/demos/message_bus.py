"""
最小标准库 demo，对应上游仓库：

- microsoft/autogen
  - message passing / topic / subscription / broadcast
  - https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/framework/message-and-communication.html
  - https://microsoft.github.io/autogen/dev/user-guide/core-user-guide/core-concepts/topic-and-subscription.html

这不是 AutoGen Core 源码复刻，而是把 publish-subscribe 机制
重写成零依赖版本：
- topic router
- subscribe
- publish
- fire-and-forget broadcast
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, DefaultDict, Dict, List


Handler = Callable[["MessageBus", str, Dict[str, str]], None]


@dataclass
class MessageBus:
    subscribers: DefaultDict[str, List[Handler]]
    trace: List[str]

    def subscribe(self, topic: str, handler: Handler) -> None:
        self.subscribers[topic].append(handler)
        self.trace.append(f"subscribe: {handler.__name__} -> {topic}")

    def publish(self, topic: str, payload: Dict[str, str]) -> int:
        handlers = list(self.subscribers.get(topic, []))
        self.trace.append(f"publish: {topic} -> handlers={len(handlers)} payload={payload}")
        for handler in handlers:
            handler(self, topic, payload)
        return len(handlers)


def audit_logger(bus: MessageBus, topic: str, payload: Dict[str, str]) -> None:
    bus.trace.append(f"audit_logger: observed {topic} severity={payload.get('severity')}")


def triage_agent(bus: MessageBus, topic: str, payload: Dict[str, str]) -> None:
    routed_topic = "alerts.network" if payload["kind"] == "network" else "alerts.identity"
    bus.trace.append(f"triage_agent: route {payload['id']} -> {routed_topic}")
    bus.publish(routed_topic, payload)


def network_investigator(bus: MessageBus, topic: str, payload: Dict[str, str]) -> None:
    finding = {
        "alert_id": payload["id"],
        "finding": "发现异常出站流量，建议隔离主机",
    }
    bus.trace.append("network_investigator: publish findings.ready")
    bus.publish("findings.ready", finding)


def response_coordinator(bus: MessageBus, topic: str, payload: Dict[str, str]) -> None:
    bus.trace.append(
        f"response_coordinator: received finding for {payload['alert_id']} -> {payload['finding']}"
    )


def run_demo() -> dict:
    bus = MessageBus(subscribers=defaultdict(list), trace=[])
    bus.subscribe("alerts.received", audit_logger)
    bus.subscribe("alerts.received", triage_agent)
    bus.subscribe("alerts.network", network_investigator)
    bus.subscribe("findings.ready", response_coordinator)

    initial_alert = {"id": "alert-17", "kind": "network", "severity": "high"}
    deliveries = bus.publish("alerts.received", initial_alert)
    return {
        "pattern": "Message Bus",
        "deliveries_on_entry_topic": deliveries,
        "response_style": "fire-and-forget broadcast",
        "trace": bus.trace,
    }


def main() -> None:
    result = run_demo()
    print("=== Message Bus Demo ===")
    print(f"entry deliveries: {result['deliveries_on_entry_topic']}")
    print(f"response style: {result['response_style']}")
    for line in result["trace"]:
        print(line)


if __name__ == "__main__":
    main()
