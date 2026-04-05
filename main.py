"""
main.py — LangGraph Supervisor Orchestrator for GAIA Level 1.

The top-level graph:

    ┌────────────┐         ┌──────────────┐
    │  classify  │────────►│  researcher  │──┐
    │  (router)  │────────►│ mathematician│──┤
    │            │────────►│ file_analyst │──├──►  [finalizer] ---> final_answer
    │            │────────►│  generalist  │──┘
    └────────────┘         └──────────────┘

Usage:
    python main.py -q "What is the population of Tokyo?"
    python main.py --gaia-file tasks.jsonl
    python main.py --interactive

Requires:
    export HF_TOKEN=hf_...
    export TAVILY_API_KEY=tvly-...
"""

import os
import json
import argparse
import logging
from dotenv import load_dotenv
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from agents import (
    AgentState,
    build_qwen_llm,
    build_gemini_llm,
    build_llm,
    build_researcher,
    build_mathematician,
    build_file_analyst,
    build_generalist,
    build_finalizer
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Supervisor State  (extends AgentState with routing metadata)
# ============================================================================
def _merge_messages(existing: list[BaseMessage], new: list[BaseMessage]) -> list[BaseMessage]:
    return existing + new


class SupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage], _merge_messages]
    next_agent: str                   # set by the classifier
    task: str                         # original question
    file_path: str                    # optional attached file
    final_answer: str                 # populated at the end


# ============================================================================
# Node: Classifier / Router
# ============================================================================
def classify_node(state: SupervisorState) -> dict:
    """
    Use the LLM to classify the task into one of the sub-agent categories.
    Returns the routing key in `next_agent`.

    FIX: If a file is attached, ALWAYS route to FILE — the LLM classifier
    often misroutes audio/image/excel questions to RESEARCH or GENERAL.
    """
    file_path = state.get("file_path", "")

    # ── Hard rule: file attached → FILE agent ──
    if file_path:
        logger.info(f"File attached ({file_path}) — forcing route to FILE")
        return {"next_agent": "FILE"}

    # ── No file: use LLM to classify ──
    llm = build_qwen_llm()
    question = state["task"]

    prompt = (
        "Classify the following question into EXACTLY one category.\n"
        "Reply with ONLY the category name — nothing else.\n\n"
        "Categories:\n"
        "- RESEARCH : needs web search or factual lookup (people, events, dates, geography)\n"
        "- MATH     : primarily calculation, counting, or mathematical reasoning\n"
        "- FILE     : requires reading or analysing an attached file\n"
        "- GENERAL  : multi-step reasoning or doesn't fit one category\n\n"
        f"Question: {question}\n"
        "No file attached.\n\n"
        "Category:"
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip().upper()

    for cat in ("RESEARCH", "MATH", "FILE", "GENERAL"):
        if cat in raw:
            logger.info(f"Classified as: {cat}")
            return {"next_agent": cat}

    logger.info("Classification unclear — defaulting to GENERAL")
    return {"next_agent": "GENERAL"}


# ============================================================================
# Node factories for each sub-agent
# ============================================================================
# We lazily compile sub-graphs so we can share the same LLM instance.
_sub_agents: dict = {}


def _get_sub_agent(name: str):
    if name not in _sub_agents:
        gemini_llm = build_gemini_llm()
        builders = {
            "RESEARCH": lambda: build_researcher(gemini_llm),
            "MATH": lambda: build_mathematician(gemini_llm),
            "FILE": lambda: build_file_analyst(gemini_llm),
            "GENERAL": lambda: build_generalist(gemini_llm),
            "FINAL": lambda: build_finalizer(gemini_llm),
        }
        _sub_agents[name] = builders[name]()
    return _sub_agents[name]


def _make_sub_agent_node(agent_key: str):
    """Return a node function that invokes the given sub-agent graph."""

    def node_fn(state: SupervisorState) -> dict:
        question = state["task"]

        if agent_key == "FINAL":
            # FIX: Don't append `question` again — it's already inside `context`
            logger.info(f"Raw Answer: {state['final_answer']}")
            message_content = (
                f"Original Task: {state['task']}\n\n"
                f"Answer: {state['final_answer']}"
            )
        else:
            file_path = state.get("file_path", "")
            if file_path:
                # FIX: Use a clear, structured format so the LLM can extract the path
                message_content = (
                    f"FILE PATH: {file_path}\n\n"
                    f"QUESTION: {question}\n\n"
                    f"IMPORTANT: The file is located at exactly: {file_path}\n"
                    f"Use the appropriate tool to read/analyze this file."
                )
            else:
                message_content = question

        sub_graph = _get_sub_agent(agent_key)
        result = sub_graph.invoke(
            {"messages": [HumanMessage(content=message_content)]}
        )

        # Extract the final AI message as the answer
        last_ai = ""
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                last_ai = msg.content
                break

        return {
            "final_answer": last_ai,
            "messages": [AIMessage(content=last_ai)],
        }

    node_fn.__name__ = f"{agent_key.lower()}_node"
    return node_fn


# ============================================================================
# Routing edge
# ============================================================================
def route_to_agent(state: SupervisorState) -> str:
    """Conditional edge after classify — pick the sub-agent node."""
    return state["next_agent"]


# ============================================================================
# Build the Supervisor Graph
# ============================================================================
def build_supervisor() -> StateGraph:
    """
    Assemble and compile the top-level supervisor graph.
    """
    # Clear cached sub-agents so a fresh build gets fresh instances
    _sub_agents.clear()

    graph = StateGraph(SupervisorState)

    # -- Nodes ---------------------------------------------------------------
    graph.add_node("classify", classify_node)
    graph.add_node("RESEARCH", _make_sub_agent_node("RESEARCH"))
    graph.add_node("MATH", _make_sub_agent_node("MATH"))
    graph.add_node("FILE", _make_sub_agent_node("FILE"))
    graph.add_node("GENERAL", _make_sub_agent_node("GENERAL"))
    graph.add_node("FINAL", _make_sub_agent_node("FINAL"))

    # -- Edges ---------------------------------------------------------------
    graph.set_entry_point("classify")
    graph.add_conditional_edges(
        "classify",
        route_to_agent,
        {
            "RESEARCH": "RESEARCH",
            "MATH": "MATH",
            "FILE": "FILE",
            "GENERAL": "GENERAL",
        },
    )
    # Each sub-agent routes to FINAL then END
    for agent_name in ("RESEARCH", "MATH", "FILE", "GENERAL"):
        graph.add_edge(agent_name, "FINAL")

    graph.add_edge("FINAL", END)

    return graph.compile()


# ============================================================================
# Runner helpers
# ============================================================================
def solve(question: str, file_path: str = "", graph=None) -> str:
    """Run a single question through the supervisor graph."""
    if graph is None:
        graph = build_supervisor()

    initial_state: SupervisorState = {
        "messages": [],
        "next_agent": "",
        "task": question,
        "file_path": file_path,
        "final_answer": "",
    }

    result = graph.invoke(initial_state)
    return result.get("final_answer", "(no answer)")


# ============================================================================
# GAIA Benchmark Runner
# ============================================================================
def run_gaia_benchmark(filepath: str, output_path: str = "results.jsonl"):
    """Run the supervisor against a GAIA .jsonl file."""
    graph = build_supervisor()

    with open(filepath, "r") as f:
        tasks = [json.loads(line) for line in f if line.strip()]

    logger.info(f"Loaded {len(tasks)} tasks from {filepath}")

    for i, task in enumerate(tasks):
        question = task.get("Question", task.get("question", ""))
        task_id = task.get("task_id", f"task_{i}")
        file_name = task.get("file_name", "")
        file_path = os.path.join("gaia_files", file_name) if file_name else ""

        logger.info(f"\n{'=' * 60}\nTask {i + 1}/{len(tasks)}: {task_id}\n{'=' * 60}")

        try:
            answer = solve(question, file_path=file_path, graph=graph)
            entry = {"task_id": task_id, "model_answer": answer}
        except Exception as e:
            logger.error(f"Error on {task_id}: {e}", exc_info=True)
            entry = {"task_id": task_id, "model_answer": "", "error": str(e)}

        with open(output_path, "a") as out:
            out.write(json.dumps(entry) + "\n")

    logger.info(f"Results saved to {output_path}")


# ============================================================================
# Interactive REPL
# ============================================================================
def interactive_mode():
    """Run the supervisor interactively."""
    graph = build_supervisor()
    print("\n🤖 GAIA Multi-Agent System (LangGraph)")
    print("   Type 'quit' to exit.  Attach a file with  file:<path>\n")

    while True:
        try:
            user_input = input("❓ Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        file_path = ""
        if "file:" in user_input:
            parts = user_input.split("file:")
            user_input = parts[0].strip()
            file_path = parts[1].strip()

        answer = solve(user_input, file_path=file_path, graph=graph)

        print(f"\n💡 Answer: {answer}\n")


# ============================================================================
# CLI Entry Point
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="GAIA Level 1 — LangGraph Multi-Agent")
    parser.add_argument("--question", "-q", type=str, help="Single question")
    parser.add_argument("--file", "-f", type=str, default="", help="Attach a file")
    parser.add_argument("--gaia-file", type=str, help="GAIA benchmark .jsonl")
    parser.add_argument("--output", "-o", type=str, default="results.jsonl", help="Benchmark output")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive REPL")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.gaia_file:
        run_gaia_benchmark(args.gaia_file, output_path=args.output)
    elif args.question:
        answer = solve(args.question, file_path=args.file)
        print(f"\nAnswer: {answer}")
    elif args.interactive:
        interactive_mode()
    else:
        parser.print_help()


if __name__ == "__main__":
    load_dotenv()
    main()


# {"task_id": "f918266a-b3e0-4914-865d-4faa564f1aef","question": "What is the final numeric output from the attached Python code?","Level": "1","file_name": "f918266a-b3e0-4914-865d-4faa564f1aef.py"}
