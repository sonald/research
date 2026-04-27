# multi-agent-patterns-demos

这个目录把 Claude 博文 [Multi-agent coordination patterns: Five approaches and when to use them](https://claude.com/blog/multi-agent-coordination-patterns) 里的 5 种协作模式，映射到 GitHub 上流行的开源 Agent 系统，并给出不依赖外部模型或 API 的最小可运行 Python 示例。

## 目录结构

- `REPORT.md`
  - 中文报告。包含 5 种模式的定义、适用场景、对应开源系统、为什么能归类到该模式、关键参考链接，以及工程启发。
- `PROVENANCE.md`
  - 代码级溯源说明。直接说明每个 demo 对应哪个仓库、哪个文档页面、哪个核心抽象。
- `demos/`
  - 5 个最小标准库 demo、5 个更贴近参考仓库形状的 demo，以及一个 `run_all.py` 入口。
- `tests/`
  - 基于 `unittest` 的 smoke tests。

## 模式映射

| Claude 模式 | 参考开源系统 | 说明 |
| --- | --- | --- |
| `Generator-Verifier` | `crewAIInc/crewAI-examples` | 主参考是 Self Evaluation Loop Flow，补充参考是 AutoGen Reflection |
| `Orchestrator-Subagent` | `microsoft/autogen` | 主参考是 Magentic-One，补充参考是 CrewAI hierarchical process |
| `Agent Teams` | `FoundationAgents/MetaGPT` | 主参考是 MetaGPT team / role，补充参考是 CAMEL Workforce |
| `Message Bus` | `microsoft/autogen` | 主参考是 AutoGen Core 与 Group Chat |
| `Shared State` | `langchain-ai/langgraph` | 使用 LangGraph 的 `StateGraph` 与共享 state 抽象作为参考 |

## 补充示例

这次新增了一组“仓库形状 demo”，更直接对应你给出的入口：

| 本地 demo | 更贴近的上游入口 |
| --- | --- |
| `demos/generator_verifier_crewai_flow.py` | `crewAI-examples/flows/self_evaluation_loop_flow/README.md` 与 `.../main.py` |
| `demos/orchestrator_subagent_magentic_one.py` | `autogen_ext/teams/magentic_one.py` |
| `demos/agent_teams_camel_workforce.py` | `camel/societies/workforce/workforce.py` |
| `demos/message_bus_group_chat.py` | AutoGen Group Chat pattern 与 `_routed_agent.py` |
| `demos/shared_state_langgraph_api.py` | `langgraph/graph/state.py` 与 `StateGraph(MessagesState)` |

## 运行方式

运行所有 demo：

```bash
python3 /Users/siancao/work/ai/research/multi-agent-patterns-demos/demos/run_all.py
```

单独运行某个 demo：

```bash
python3 /Users/siancao/work/ai/research/multi-agent-patterns-demos/demos/generator_verifier.py
python3 /Users/siancao/work/ai/research/multi-agent-patterns-demos/demos/orchestrator_subagent.py
python3 /Users/siancao/work/ai/research/multi-agent-patterns-demos/demos/agent_teams.py
python3 /Users/siancao/work/ai/research/multi-agent-patterns-demos/demos/message_bus.py
python3 /Users/siancao/work/ai/research/multi-agent-patterns-demos/demos/shared_state.py
python3 /Users/siancao/work/ai/research/multi-agent-patterns-demos/demos/generator_verifier_crewai_flow.py
python3 /Users/siancao/work/ai/research/multi-agent-patterns-demos/demos/orchestrator_subagent_magentic_one.py
python3 /Users/siancao/work/ai/research/multi-agent-patterns-demos/demos/agent_teams_camel_workforce.py
python3 /Users/siancao/work/ai/research/multi-agent-patterns-demos/demos/message_bus_group_chat.py
python3 /Users/siancao/work/ai/research/multi-agent-patterns-demos/demos/shared_state_langgraph_api.py
```

运行测试：

```bash
python3 -m unittest discover -s /Users/siancao/work/ai/research/multi-agent-patterns-demos/tests -v
```

## 设计原则

- 所有示例只使用 Python 标准库。
- 示例代码演示的是“协调模式”，不是对上游框架源码的复刻。
- 每个 demo 文件头部都写明了上游参考仓库与对应抽象，例如 `Reflection`、`Process.hierarchical`、`Team`、`publish/subscribe`、`StateGraph`。
- 目录分成两层：
  - 最小 demo：优先把模式讲清楚。
  - 仓库形状 demo：优先让本地代码更明显地贴近参考仓库的类名、组织方式和心智模型。
- trace 会尽量把模式的关键行为打印清楚，例如：
  - verifier criteria 不清晰时的误验收
  - orchestrator 如何分解子任务
  - team worker 如何保留本地上下文
  - message bus 如何 fire-and-forget 广播
  - shared state 如何用 termination condition 避免循环
