# 代码级溯源说明

这份目录里的 demo 代码不是从上游仓库直接复制或裁剪出来的，而是为了“本地可运行、零依赖、突出协调模式”而用 Python 标准库重写的版本。

现在目录分成两层：

- 最小 demo：先把模式本身讲清楚。
- 仓库形状 demo：让本地代码更明显地贴近上游仓库的类名、组织方式和心智模型。

如果你要看“每个 demo 是从哪些开源系统的哪些抽象借来的”，请按下面这张表读：

| 本地 demo | 参考仓库 | 对应上游概念 | 代码里保留的核心机制 |
| --- | --- | --- | --- |
| `demos/generator_verifier.py` | `microsoft/autogen` | `Reflection` 设计模式里的 generator / reviewer 回路 | 初稿生成、显式 criteria、feedback 回流、最大迭代上限 |
| `demos/orchestrator_subagent.py` | `crewAIInc/crewAI` | `Process.hierarchical`、`manager_agent` / `manager_llm` | orchestrator 拆任务、subagent 短生命周期、结果回收与综合 |
| `demos/agent_teams.py` | `FoundationAgents/MetaGPT` | `Team`、`hire(...)`、长期 `Role` 协作 | 持久 worker、角色专长、跨轮次本地记忆、分区冲突 |
| `demos/message_bus.py` | `microsoft/autogen` | Core runtime 的 message passing、topic、subscription、broadcast | `publish`、`subscribe`、router 分发、无响应广播 |
| `demos/shared_state.py` | `langchain-ai/langgraph` | `StateGraph` 的 shared state / state updates | 多 agent 读写同一 store、version、termination condition |
| `demos/generator_verifier_crewai_flow.py` | `crewAIInc/crewAI-examples` | `Self Evaluation Loop Flow`、`ShakespeareanXPostCrew`、`XPostReviewCrew` | 两套 crew、反馈回流、最大重试数 |
| `demos/orchestrator_subagent_magentic_one.py` | `microsoft/autogen` | `Magentic-One`、lead `Orchestrator`、`Task Ledger`、`Progress Ledger` | 外层重规划、内层分派执行、专职 subagents |
| `demos/agent_teams_camel_workforce.py` | `camel-ai/camel` | `Workforce`、`Coordinator Agent`、`Task Planner Agent`、dynamic workers | 团队常驻、动态扩编、任务分发 |
| `demos/message_bus_group_chat.py` | `microsoft/autogen` | Group Chat、shared topic、`RequestToSpeak`、`RoutedAgent` | 同 topic publish/subscribe、manager 驱动轮次 |
| `demos/shared_state_langgraph_api.py` | `langchain-ai/langgraph` | `StateGraph`、`compile()`、`invoke()`、`MessagesState` | 节点读取 shared state、返回 partial state、messages channel 累积 |

## 逐文件说明

### `generator_verifier.py`

参考来源：

- AutoGen Reflection
  - <https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/reflection.html>

上游抽象：

- `coder agent` / `reviewer agent`
- critique / revise loop

本地对应：

- `Generator.initial_draft()` / `Generator.revise()`
- `Verifier.review()`
- `run_review_loop(...)`

### `orchestrator_subagent.py`

参考来源：

- CrewAI Hierarchical Process
  - <https://docs.crewai.com/en/learn/hierarchical-process>
- CrewAI Processes
  - <https://docs.crewai.com/en/concepts/processes>

上游抽象：

- `process=Process.hierarchical`
- `manager_agent` / `manager_llm`

本地对应：

- `Orchestrator.plan()` / `Orchestrator.run()`
- `run_subagent(...)`

### `agent_teams.py`

参考来源：

- MetaGPT Quickstart
  - <https://docs.deepwisdom.ai/main/en/guide/get_started/quickstart.html>
- MetaGPT MultiAgent 101
  - <https://docs.deepwisdom.ai/main/en/guide/tutorials/multi_agent_101.html>

上游抽象：

- `Team()`
- `team.hire([...])`
- 稳定角色长期协作

本地对应：

- `TeamCoordinator`
- `Worker`
- `worker.memory`

### `message_bus.py`

参考来源：

- AutoGen Message and Communication
  - <https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/framework/message-and-communication.html>
- AutoGen Topic and Subscription
  - <https://microsoft.github.io/autogen/dev/user-guide/core-user-guide/core-concepts/topic-and-subscription.html>

上游抽象：

- message passing
- topic
- subscription
- broadcast

本地对应：

- `MessageBus.subscribe(...)`
- `MessageBus.publish(...)`
- 基于 topic 的处理器分发

### `shared_state.py`

参考来源：

- LangGraph Graph API Overview
  - <https://docs.langchain.com/oss/python/langgraph/graph-api>
- LangGraph Use the Graph API
  - <https://docs.langchain.com/oss/python/langgraph/use-graph-api>

上游抽象：

- `State`
- nodes 对 state 的读写
- graph termination

本地对应：

- `SharedStore`
- `ResearchAgent.act(...)`
- `termination_monitor(...)`

### `generator_verifier_crewai_flow.py`

参考来源：

- crewAI Self Evaluation Loop Flow README
  - <https://github.com/crewAIInc/crewAI-examples/blob/main/flows/self_evaluation_loop_flow/README.md>
- crewAI Self Evaluation Loop Flow `main.py`
  - <https://github.com/crewAIInc/crewAI-examples/blob/main/flows/self_evaluation_loop_flow/src/self_evaluation_loop_flow/main.py>

上游抽象：

- `ShakespeareanXPostCrew`
- `XPostReviewCrew`
- self evaluation loop

本地对应：

- `ShakespeareanXPostCrew`
- `XPostReviewCrew`
- `SelfEvaluationLoopFlow`

### `orchestrator_subagent_magentic_one.py`

参考来源：

- AutoGen Magentic-One
  - <https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/magentic-one.html>
- AutoGen `magentic_one.py`
  - <https://github.com/microsoft/autogen/blob/main/python/packages/autogen-ext/src/autogen_ext/teams/magentic_one.py>

上游抽象：

- lead `Orchestrator`
- `Task Ledger`
- `Progress Ledger`
- `WebSurfer` / `FileSurfer` / `Coder` / `ComputerTerminal`

本地对应：

- `LeadOrchestrator`
- `TaskLedger`
- `ProgressLedger`
- `SpecialistAgent`

### `agent_teams_camel_workforce.py`

参考来源：

- CAMEL Workforce docs
  - <https://docs.camel-ai.org/key_modules/workforce>
- CAMEL `workforce.py`
  - <https://github.com/camel-ai/camel/blob/master/camel/societies/workforce/workforce.py>

上游抽象：

- `Workforce`
- `Coordinator Agent`
- `Task Planner Agent`
- dynamic workers

本地对应：

- `Workforce`
- `CoordinatorAgent`
- `TaskPlannerAgent`
- `_create_dynamic_worker(...)`

### `message_bus_group_chat.py`

参考来源：

- AutoGen Group Chat
  - <https://microsoft.github.io/autogen/dev/user-guide/core-user-guide/design-patterns/group-chat.html>
- AutoGen `RoutedAgent`
  - <https://github.com/microsoft/autogen/blob/main/python/packages/autogen-core/src/autogen_core/_routed_agent.py>

上游抽象：

- shared group topic
- `RequestToSpeak`
- group chat manager

本地对应：

- `TopicBus`
- `GroupChatRuntime.group_chat_manager(...)`
- participant topics `planner` / `writer` / `editor` / `user`

### `shared_state_langgraph_api.py`

参考来源：

- LangGraph Graph API
  - <https://docs.langchain.com/oss/python/langgraph/use-graph-api>
- LangGraph overview
  - <https://docs.langchain.com/oss/python/langgraph>
- LangGraph `state.py`
  - <https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/graph/state.py>

上游抽象：

- `StateGraph`
- `compile()`
- `invoke()`
- `MessagesState`

本地对应：

- `PseudoStateGraph`
- `PseudoCompiledGraph`
- `planner_node` / `research_node` / `answer_node`

## 这不是“源码复刻”

如果你的期待是“从这些热门开源仓库里摘出更贴近原项目 API 形状的示例”，现在这份目录已经补了第二层：

- 第一层：用官方文档确认模式归类，再用标准库重演核心协调机制。
- 第二层：在不引入外部依赖的前提下，把 demo 尽量写得更像参考仓库的类名和结构。
