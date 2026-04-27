from __future__ import annotations

import generator_verifier
import generator_verifier_crewai_flow
import orchestrator_subagent
import orchestrator_subagent_magentic_one
import agent_teams
import agent_teams_camel_workforce
import message_bus
import message_bus_group_chat
import shared_state
import shared_state_langgraph_api


def main() -> None:
    minimal_demos = [
        ("Generator-Verifier", generator_verifier.run_demo),
        ("Orchestrator-Subagent", orchestrator_subagent.run_demo),
        ("Agent Teams", agent_teams.run_demo),
        ("Message Bus", message_bus.run_demo),
        ("Shared State", shared_state.run_demo),
    ]
    repo_shaped_demos = [
        ("CrewAI Self Evaluation Loop", generator_verifier_crewai_flow.run_demo),
        ("AutoGen Magentic-One", orchestrator_subagent_magentic_one.run_demo),
        ("CAMEL Workforce", agent_teams_camel_workforce.run_demo),
        ("AutoGen Group Chat", message_bus_group_chat.run_demo),
        ("LangGraph StateGraph", shared_state_langgraph_api.run_demo),
    ]

    print("=== Multi-Agent Coordination Patterns Demos ===")
    print("\n# Minimal demos")
    for name, runner in minimal_demos:
        result = runner()
        print(f"\n## {name}")
        if name == "Generator-Verifier":
            print(
                f"vague accepted={result['vague_verifier']['accepted']}, "
                f"strict accepted={result['strict_verifier']['accepted']}, "
                f"capped accepted={result['capped_strict_verifier']['accepted']}"
            )
        elif name == "Orchestrator-Subagent":
            print(result["summary"])
        elif name == "Agent Teams":
            print(
                f"frontend handled={result['worker_state']['frontend']['handled_count']}, "
                f"conflicts={len(result['round_two']['conflicts'])}"
            )
        elif name == "Message Bus":
            print(
                f"deliveries_on_entry_topic={result['deliveries_on_entry_topic']}, "
                f"response_style={result['response_style']}"
            )
        else:
            print(f"done={result['done']}, version={result['version']}, findings={len(result['findings'])}")

    print("\n# Repository-shaped demos")
    for name, runner in repo_shaped_demos:
        result = runner()
        print(f"\n## {name}")
        if name == "CrewAI Self Evaluation Loop":
            print(f"valid={result['valid']}, attempts={result['attempts']}, topic={result['topic']}")
        elif name == "AutoGen Magentic-One":
            print(
                f"outer_loops={result['outer_loops']}, "
                f"status={result['progress_ledger']['status']}, "
                f"stalled_steps={result['progress_ledger']['stalled_steps']}"
            )
        elif name == "CAMEL Workforce":
            print(
                f"completed_tasks={len(result['completed_tasks'])}, "
                f"dynamic_workers={len(result['dynamic_workers'])}"
            )
        elif name == "AutoGen Group Chat":
            print(
                f"group_topic={result['group_topic']}, "
                f"transcript_lines={len(result['transcript'])}"
            )
        else:
            print(
                f"messages={len(result['final_state']['messages'])}, "
                f"has_answer={'answer' in result['final_state']}"
            )


if __name__ == "__main__":
    main()
