# 多 Agent 协调模式报告

## 摘要

本文把 Claude 在 2026-04-10 发布的文章 [Multi-agent coordination patterns: Five approaches and when to use them](https://claude.com/blog/multi-agent-coordination-patterns) 中的 5 种 coordination pattern，与 GitHub 上流行的开源 Agent 系统做对照，并给出可运行的最小演示代码。

选择原则：

- 只使用一手来源：官方 GitHub 仓库、官方文档、官方示例或直接源码页面。
- 优先选择“能清楚看出协调机制”的仓库或子系统，而不是泛泛的 agent 框架。
- 允许一个热门仓库覆盖多个模式，但必须明确说明映射的是哪个子系统。

GitHub 热度快照日期：`2026-04-20`

| 模式 | 适用场景 | 主参考 | 补充参考 | 核心理由 |
| --- | --- | --- | --- | --- |
| `Generator-Verifier` | 质量敏感，且验收标准可显式定义 | [`crewAIInc/crewAI-examples`](https://github.com/crewAIInc/crewAI-examples) 约 `5.8k` stars | [`microsoft/autogen`](https://github.com/microsoft/autogen) 约 `57.2k` stars | CrewAI 的 Self Evaluation Loop Flow 几乎就是标准的生成-评审-反馈重试闭环 |
| `Orchestrator-Subagent` | 任务可拆分为边界清晰的子任务 | [`microsoft/autogen`](https://github.com/microsoft/autogen) 约 `57.2k` stars | [`crewAIInc/crewAI`](https://github.com/crewAIInc/crewAI) 约 `49.3k` stars | Magentic-One 明确采用 lead orchestrator 规划、分派、追踪进度 |
| `Agent Teams` | 并行、长期、相互独立的子任务 | [`FoundationAgents/MetaGPT`](https://github.com/FoundationAgents/MetaGPT) 约 `67.3k` stars | [`camel-ai/camel`](https://github.com/camel-ai/camel) 约 `16.7k` stars | MetaGPT 偏“公司式 team”，CAMEL Workforce 偏“持久 worker 团队” |
| `Message Bus` | 事件驱动流水线，agent 生态会持续扩张 | [`microsoft/autogen`](https://github.com/microsoft/autogen) 约 `57.2k` stars | AutoGen Group Chat / `RoutedAgent`，另可补看 `SPADE` | AutoGen Core 和 Group Chat 都把 topic / publish / subscribe 放在中心位置 |
| `Shared State` | 协同研究，agent 需要共享并增量构建发现 | [`langchain-ai/langgraph`](https://github.com/langchain-ai/langgraph) 约 `29.7k` stars | 无 | LangGraph 把 `StateGraph` 与显式 state 放在核心抽象层 |

## 本次新增的本地示例

为了吸收这轮反馈，目录现在不只有“最小 demo”，还增加了 5 个更贴近参考仓库形状的本地示例：

- `Generator-Verifier`
  - [`generator_verifier.py`](/Users/siancao/work/ai/research/multi-agent-patterns-demos/demos/generator_verifier.py)
  - [`generator_verifier_crewai_flow.py`](/Users/siancao/work/ai/research/multi-agent-patterns-demos/demos/generator_verifier_crewai_flow.py)
- `Orchestrator-Subagent`
  - [`orchestrator_subagent.py`](/Users/siancao/work/ai/research/multi-agent-patterns-demos/demos/orchestrator_subagent.py)
  - [`orchestrator_subagent_magentic_one.py`](/Users/siancao/work/ai/research/multi-agent-patterns-demos/demos/orchestrator_subagent_magentic_one.py)
- `Agent Teams`
  - [`agent_teams.py`](/Users/siancao/work/ai/research/multi-agent-patterns-demos/demos/agent_teams.py)
  - [`agent_teams_camel_workforce.py`](/Users/siancao/work/ai/research/multi-agent-patterns-demos/demos/agent_teams_camel_workforce.py)
- `Message Bus`
  - [`message_bus.py`](/Users/siancao/work/ai/research/multi-agent-patterns-demos/demos/message_bus.py)
  - [`message_bus_group_chat.py`](/Users/siancao/work/ai/research/multi-agent-patterns-demos/demos/message_bus_group_chat.py)
- `Shared State`
  - [`shared_state.py`](/Users/siancao/work/ai/research/multi-agent-patterns-demos/demos/shared_state.py)
  - [`shared_state_langgraph_api.py`](/Users/siancao/work/ai/research/multi-agent-patterns-demos/demos/shared_state_langgraph_api.py)

## 1. Generator-Verifier

### Claude 原文中的定义与适用条件

Claude 把这个模式定义为：一个 generator 先产出初稿，再交给 verifier 按显式标准审查；如果不通过，就把反馈回送给 generator 继续修正，直到通过或达到最大迭代次数。

它适合：

- 输出错误代价高
- 验收标准可以显式写清
- 生成与评估这两件事可以相对分离

它的典型失败方式也很清晰：

- verifier 标准过于模糊，导致“看起来像验证、实际上只是放行”
- generator 和 verifier 互相拉扯，但没有收敛
- 没有最大迭代数或 fallback，最后变成无限循环

### 选中的开源系统

主参考系统：`crewAIInc/crewAI-examples` 的 `Self Evaluation Loop Flow`。

补充参考系统：`microsoft/autogen` 的 `Reflection` 设计模式。

### 为什么它属于这个模式

这是一个“主参考完全贴脸，补充参考也高度匹配”的模式。

CrewAI examples 里这个 flow 的 README 直接把结构写成“两套 crew”：一套负责生成，一套负责评审；如果不满足条件，就带着 feedback 重试，直到通过或者达到最大重试次数。这和 Claude 原文里的 generator-verifier 基本同构。

AutoGen Reflection 则是一个很好的补充参照：它把类似的生成-评审-修正回路抽象成了更通用的设计模式。所以这里最适合的读法是：

- 如果你要看“最标准的 loop 例子”，先看 CrewAI examples。
- 如果你要看“框架化的模式抽象”，再看 AutoGen Reflection。

### 直接参考链接

- Claude 原文：<https://claude.com/blog/multi-agent-coordination-patterns>
- CrewAI Self Evaluation Loop README：<https://github.com/crewAIInc/crewAI-examples/blob/main/flows/self_evaluation_loop_flow/README.md>
- CrewAI Self Evaluation Loop `main.py`：<https://github.com/crewAIInc/crewAI-examples/blob/main/flows/self_evaluation_loop_flow/src/self_evaluation_loop_flow/main.py>
- AutoGen Reflection：<https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/reflection.html>
- AutoGen README 中的维护状态说明：<https://raw.githubusercontent.com/microsoft/autogen/main/README.md>

### 维护状态说明

AutoGen 当前处于 `maintenance mode`。这意味着它不再是微软主推的新功能承载框架，但它已经沉淀出的设计模式文档仍然非常适合作为研究材料，尤其适合拿来理解经典 multi-agent 协调方式。

### 落地启发

最值得借鉴的不是“多加一个 reviewer agent”，而是把 verifier 的判定标准写成可执行、可枚举的 checklist。真正的价值来自：

- 标准清晰
- 反馈具体
- 有明确的停止条件

本目录里现在有两份本地示例：

- [`generator_verifier.py`](/Users/siancao/work/ai/research/multi-agent-patterns-demos/demos/generator_verifier.py)：更抽象、更适合先理解模式。
- [`generator_verifier_crewai_flow.py`](/Users/siancao/work/ai/research/multi-agent-patterns-demos/demos/generator_verifier_crewai_flow.py)：更贴近 CrewAI Self Evaluation Loop 的组织方式。

## 2. Orchestrator-Subagent

### Claude 原文中的定义与适用条件

Claude 把这个模式定义为：一个 lead agent 负责规划、拆分、委派和汇总；subagent 负责具体的、边界明确的子任务，做完后返回结果。

它适合：

- 任务可以拆成若干边界清晰的子任务
- 子任务之间的耦合不高
- 让主 agent 长时间背负所有探索上下文，成本太高

### 选中的开源系统

主参考系统：`microsoft/autogen` 的 `Magentic-One`。

补充参考系统：`crewAIInc/crewAI` 的 `hierarchical process`。

### 为什么它属于这个模式

这是一个“主参考偏复杂生产系统，补充参考偏简洁层级委派”的组合。

AutoGen 官方对 Magentic-One 的描述非常直白：lead orchestrator 负责高层规划、指挥其他 agents、追踪进度；它会维护 `Task Ledger` 和 `Progress Ledger`，在进度不理想时重新规划。这几乎就是 Claude 文里 orchestrator-subagent 的加强版。

CrewAI 的 hierarchical process 则是更简洁的同类参照。它没有 Magentic-One 那么多专职 agent 与 ledger 抽象，但 manager -> delegated workers 的层级关系非常清楚。

它和 Claude 文中的 orchestrator-subagent 几乎一一对应：

- manager 负责判断怎么拆分和派发
- worker agent 负责完成被委派的任务
- manager 汇总并推进后续步骤

需要注意的是，CrewAI 本身也支持其他 process，因此“CrewAI 整体”不是等同于这个模式；准确地说，是 CrewAI 的 hierarchical process 子机制对应 orchestrator-subagent。

### 直接参考链接

- Claude 原文：<https://claude.com/blog/multi-agent-coordination-patterns>
- AutoGen Magentic-One：<https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/magentic-one.html>
- AutoGen `magentic_one.py`：<https://github.com/microsoft/autogen/blob/main/python/packages/autogen-ext/src/autogen_ext/teams/magentic_one.py>
- CrewAI Hierarchical Process：<https://docs.crewai.com/en/learn/hierarchical-process>
- CrewAI Processes：<https://docs.crewai.com/en/concepts/processes>

### 落地启发

这个模式最容易被误用的地方，是把 subagent 当成“共享一个大脑的并行线程”。更稳健的做法恰恰相反：

- 每个 subagent 只拿到完成当前子任务所需的最小上下文
- 子任务产出结构化结果
- orchestrator 负责合并、仲裁和继续规划

本目录里现在有两份本地示例：

- [`orchestrator_subagent.py`](/Users/siancao/work/ai/research/multi-agent-patterns-demos/demos/orchestrator_subagent.py)：最小版，突出短生命周期 subagent。
- [`orchestrator_subagent_magentic_one.py`](/Users/siancao/work/ai/research/multi-agent-patterns-demos/demos/orchestrator_subagent_magentic_one.py)：更贴近 Magentic-One 的 orchestrator + ledgers 结构。

## 3. Agent Teams

### Claude 原文中的定义与适用条件

Claude 把 agent teams 描述为：多个 agent 像“队友”一样并行处理长期、独立的任务。它们不是每次都由 orchestrator 做细粒度调度，而是各自长期保有自己的上下文和专长。

它适合：

- 子任务之间相对独立
- 每个子任务都需要持续积累上下文
- 任务持续时间不一致，且适合由稳定角色长期接管

它的关键风险则是：

- 共享资源冲突
- 相互影响却又无法及时同步
- 完成时间不一致带来的协调难题

### 选中的开源系统

主参考系统：`FoundationAgents/MetaGPT`。

补充参考系统：`camel-ai/camel` 的 `Workforce`。

### 为什么它属于这个模式

这是一个“高匹配度，但属于更大系统中的主导组织方式”。

MetaGPT 最有代表性的抽象就是 `Team` 与一组 `Role`。官方 quickstart 里直接展示了 `ProductManager`、`Architect`、`ProjectManager`、`Engineer` 被 `Team` 雇佣后协同推进项目；官方 MultiAgent 101 教程则进一步展示了如何自定义多角色、把它们放进同一个 team 中运行。

这非常接近 Claude 所说的 agent teams：

- 角色稳定存在
- 角色有长期职责和领域上下文
- 协作以“团队”而不是一次性子任务为中心

CAMEL Workforce 则更接近另一种 agent teams 心智模型：不是“软件公司角色剧本”，而是一个可扩展 workforce，核心类名就叫 `Workforce`，并且文档明确写出 coordinator、task planner、dynamic workers 这些内部角色。

所以这一类最适合的读法是：

- 想看“公司式长期角色协作”，先读 MetaGPT。
- 想看“持久 worker 团队 + 动态扩编”，再读 CAMEL Workforce。

### 直接参考链接

- Claude 原文：<https://claude.com/blog/multi-agent-coordination-patterns>
- MetaGPT 仓库：<https://github.com/FoundationAgents/MetaGPT>
- MetaGPT Quickstart：<https://docs.deepwisdom.ai/main/en/guide/get_started/quickstart.html>
- MetaGPT MultiAgent 101：<https://docs.deepwisdom.ai/main/en/guide/tutorials/multi_agent_101.html>
- CAMEL Workforce docs：<https://docs.camel-ai.org/key_modules/workforce>
- CAMEL `workforce.py`：<https://github.com/camel-ai/camel/blob/master/camel/societies/workforce/workforce.py>

### 落地启发

这个模式最值得借鉴的点，是“把稳定的专长和局部记忆放进持久 worker”，而不是每次都从零开始派发。代价是共享资源会更容易冲突，所以工程上要尽早做：

- 任务分区
- 资源所有权划分
- 冲突检测或串行化策略

本目录里现在有两份本地示例：

- [`agent_teams.py`](/Users/siancao/work/ai/research/multi-agent-patterns-demos/demos/agent_teams.py)：更贴近 MetaGPT 式稳定角色。
- [`agent_teams_camel_workforce.py`](/Users/siancao/work/ai/research/multi-agent-patterns-demos/demos/agent_teams_camel_workforce.py)：更贴近 CAMEL Workforce 的 persistent workers + dynamic worker 结构。

## 4. Message Bus

### Claude 原文中的定义与适用条件

Claude 把 message bus 定义为：agent 不再通过固定的点对点连接互动，而是通过共享通信层进行 `publish` / `subscribe`。router 根据 topic 把消息送到订阅者，新的 agent 可以在不改动旧连接关系的情况下加入生态。

它适合：

- 事件驱动流水线
- agent 数量会持续增长
- 希望弱耦合接入新能力

### 选中的开源系统

主参考系统：`microsoft/autogen` 的 Core runtime、`RoutedAgent`、topic / subscription 与 Group Chat 机制。

### 为什么它属于这个模式

这是一个“纯匹配”，而且是 AutoGen Core 的基础能力，不是上层示例的偶然写法。

AutoGen 官方文档把 message 定义为 agent 间通信的唯一手段，并专门区分 direct messaging 和 broadcast；topic 与 subscription 文档则进一步把 broadcast 描述成运行时层面的 publish-subscribe 模型。

换句话说，这不是“某个 AutoGen demo 恰好用了总线风格”，而是 AutoGen Core 从 runtime 层就把消息总线、topic 和订阅抽象成了中心机制。

### 直接参考链接

- Claude 原文：<https://claude.com/blog/multi-agent-coordination-patterns>
- AutoGen Message and Communication：<https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/framework/message-and-communication.html>
- AutoGen Topic and Subscription：<https://microsoft.github.io/autogen/dev/user-guide/core-user-guide/core-concepts/topic-and-subscription.html>
- AutoGen Group Chat：<https://microsoft.github.io/autogen/dev/user-guide/core-user-guide/design-patterns/group-chat.html>
- AutoGen `RoutedAgent`：<https://github.com/microsoft/autogen/blob/main/python/packages/autogen-core/src/autogen_core/_routed_agent.py>

### 落地启发

message bus 真正的价值，不在于“消息多发几次”，而在于把系统从硬编码的调用关系里解耦出来。这样做的好处是：

- 新 agent 可以旁路接入
- 旧 agent 不必知道下游具体是谁
- 事件流可以同时被多个能力模块消费

本目录里现在有两份本地示例：

- [`message_bus.py`](/Users/siancao/work/ai/research/multi-agent-patterns-demos/demos/message_bus.py)：更抽象的 topic router / fire-and-forget broadcast。
- [`message_bus_group_chat.py`](/Users/siancao/work/ai/research/multi-agent-patterns-demos/demos/message_bus_group_chat.py)：更贴近 AutoGen Group Chat 里“共享 topic + request to speak”的模式。

## 5. Shared State

### Claude 原文中的定义与适用条件

Claude 把 shared-state 描述为：多个 agent 通过共享存储直接协作，而不是靠中心 coordinator 中转。agent 可以直接读写共享知识库，彼此的发现会立刻成为其他 agent 的下一轮输入。

它适合：

- 协同研究或协同发现
- agent 需要相互建立在对方的结果之上
- 希望减少中心节点带来的单点故障

它的难点则包括：

- 重复劳动
- 并发写入
- reactive loop

因此 termination condition 必须是一等公民，而不是事后补丁。

### 选中的开源系统

参考系统：`langchain-ai/langgraph` 的 `StateGraph`。

### 为什么它属于这个模式

这是一个“框架核心抽象与模式高度一致”的匹配。

LangGraph 官方把 `State` 定义为图执行时的共享数据结构，把 `Nodes` 定义为读入当前 state、再返回 state 更新的函数；同时文档还明确指出，默认情况下所有节点都通过同一个 schema 通信，也就是共享同一组 state channels。

这与 Claude 文里的 shared-state 模式几乎同构：

- 没有必须存在的中央 orchestrator
- 多个节点通过共享状态协作
- 节点的输出会更新公共状态，成为其他节点的输入

严格来说，LangGraph 不是只做 shared-state multi-agent，它是通用的 stateful orchestration framework；但它的核心建模方式，正好就是 shared-state coordination。

### 直接参考链接

- Claude 原文：<https://claude.com/blog/multi-agent-coordination-patterns>
- LangGraph README：<https://github.com/langchain-ai/langgraph>
- LangGraph Graph API Overview：<https://docs.langchain.com/oss/python/langgraph/graph-api>
- LangGraph Use the Graph API：<https://docs.langchain.com/oss/python/langgraph/use-graph-api>
- LangGraph `state.py`：<https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/graph/state.py>

### 落地启发

shared state 最大的工程价值，是让“发现”本身变成可持续积累的协作介质；最大的工程风险，则是系统没有自然终点。实践里至少要补上：

- versioning
- locking 或 reducer 语义
- 明确的 termination condition

本目录里现在有两份本地示例：

- [`shared_state.py`](/Users/siancao/work/ai/research/multi-agent-patterns-demos/demos/shared_state.py)：更强调 shared store、version、termination monitor。
- [`shared_state_langgraph_api.py`](/Users/siancao/work/ai/research/multi-agent-patterns-demos/demos/shared_state_langgraph_api.py)：更贴近 LangGraph 的 `StateGraph` / `compile()` / `invoke()` / `MessagesState` 心智模型。

## 总结

这 5 种模式并不是互斥的产品分类，更像是构建块：

- AutoGen 同时覆盖了 `Generator-Verifier` 和 `Message Bus`
- 一个系统可以先用 `Orchestrator-Subagent` 做总控，再在某个子问题里嵌入 `Shared State`
- `Agent Teams` 和 `Message Bus` 也可以组合成“事件驱动 + 持久 worker”结构

如果从工程起步角度看，Claude 在原文里的建议依然非常实用：先从最简单、最能解释当前问题的模式开始，再根据真实痛点演进，而不是一开始就追求“最像 swarm 的系统”。
