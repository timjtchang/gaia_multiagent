"""
agents.py — LangGraph-based sub-agents for GAIA Level 1 multi-agent system.

Architecture:
    Each sub-agent is a compiled LangGraph sub-graph that runs a ReAct
    tool-calling loop (LLM node ⇄ Tool node) with scoped tools and
    a tailored system prompt.

    The Orchestrator in main.py wires these sub-graphs as nodes inside
    a top-level supervisor graph.
"""

import os
import logging
from typing import Annotated, TypedDict, Literal, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from tools import RESEARCH_TOOLS, MATH_TOOLS, FILE_TOOLS, ALL_TOOLS

logger = logging.getLogger(__name__)


# ============================================================================
# Shared State Schema
# ============================================================================
def _merge_messages(
    existing: list[BaseMessage], new: list[BaseMessage]
) -> list[BaseMessage]:
    """Reducer: append new messages to the existing list."""
    return existing + new


class AgentState(TypedDict):
    """State that flows through every node in a sub-agent graph."""
    messages: Annotated[list[BaseMessage], _merge_messages]


# ============================================================================
# LLM Factories
# ============================================================================
def build_qwen_llm(
    model_id: str = "Qwen/Qwen2.5-72B-Instruct",
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> ChatHuggingFace:
    """
    Build a ChatHuggingFace LLM backed by a HuggingFace Inference Endpoint.
    Used for routing/classification and lightweight sub-agents.
    Requires HF_TOKEN env var.
    """
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("Set the HF_TOKEN environment variable.")

    llm_endpoint = HuggingFaceEndpoint(
        repo_id=model_id,
        task="conversational",
        max_new_tokens=max_tokens,
        temperature=temperature,
        huggingfacehub_api_token=hf_token,
    )
    return ChatHuggingFace(llm=llm_endpoint)


def build_gemini_llm(
    model: str = "gemini-2.5-flash",
    temperature: float = 0.0,
    max_tokens: int = 4096,
    agent_name: str=""
) -> BaseChatModel:
    """
    Build a Gemini LLM via langchain-google-genai.
    Used for the Generalist agent where deep reasoning matters.
    Requires GOOGLE_API_KEY env var.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set the GOOGLE_API_KEY environment variable.")

    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )


# Backward-compatible alias
build_llm = build_qwen_llm


# ============================================================================
# Sub-Agent Graph Builder
# ============================================================================
def _should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """Conditional edge: route to tool node if the last message has tool calls."""
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tools"
    return "__end__"


def build_agent_graph(
    name: str,
    system_prompt: str,
    tools: list=[],
    llm: BaseChatModel | None = None,
) -> StateGraph:
    """
    Build a compiled ReAct sub-agent graph:

        ┌──────────┐     has tool calls     ┌───────────┐
        │  agent   │ ──────────────────────► │   tools   │
        │  (LLM)   │ ◄────────────────────── │  (exec)   │
        └──────────┘     tool results        └───────────┘
              │
              │  no tool calls
              ▼
            [FINAL] --------> [END]
    """
    if llm is None:
        llm = build_llm()

    logger.info("Use Model: "+llm.model)        
    
    # Bind tools so the LLM can emit tool_calls
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState) -> dict:
        """Invoke the LLM with the current message history."""
        # Prepend system prompt if not already present
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + messages

        response = llm_with_tools.with_config({"run_name": f"{name}_LLM"}).invoke(messages)

        # Handle Gemini's raw response format if langchain-google-genai doesn't parse it
        if isinstance(response.content, list):
            text = "".join(
                block.get("text", "") for block in response.content 
                if isinstance(block, dict) and block.get("type") == "text"
            )
            # Create a new AIMessage with cleaned content but preserve other attributes
            cleaned_response = AIMessage(
                content=text,
                tool_calls=response.tool_calls,
                additional_kwargs=response.additional_kwargs,
                response_metadata=response.response_metadata,
                id=response.id,
            )
            return {"messages": [cleaned_response]}

        return {"messages": [response]}

    # -- Assemble the graph --------------------------------------------------
    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", _should_continue, {"tools": "tools", "__end__": END})
    graph.add_edge("tools", "agent")      # loop back after tool execution

    return graph.compile()


# ============================================================================
# Pre-built Sub-Agents
# ============================================================================

def build_finalizer(llm: BaseChatModel | None = None):
    """Finalizer agent: to structure final answer."""
    return build_agent_graph(
        name="Finalizer",
        system_prompt=(
            "You are an output formatter for the GAIA benchmark.\n"
            "I will provide you with an original task and a raw answer.\n"
            "Your job is to extract the FINAL ANSWER and return ONLY the raw value.\n\n"
            "RULES:\n"
            "- Return ONLY the answer value, nothing else. No explanations, no prefixes.\n"
            "- If the raw answer contains reasoning/explanation, extract just the final value.\n"
            "- If the answer is a list, format as comma-separated values.\n"
            "- If the answer is a number, return just the number.\n"
            "- If the answer is a name, return just the name.\n"
            "- Do NOT add 'FINAL ANSWER:' or any other prefix.\n"
            "- Do NOT add periods at the end unless they are part of the answer.\n"
            "- If the raw answer is an error message or says 'I cannot', return an empty string."
        ),
        llm=llm,
    )

def build_researcher(llm: BaseChatModel | None = None):
    """Research agent: Tavily web search + Wikipedia."""
    return build_agent_graph(
        name="Researcher",
        system_prompt=(
            "You are a research specialist. Find accurate, factual information.\n"
            "Strategy:\n"
            "1. Use wiki_search for historical events, Olympic records, demographics, biography facts, country data, or any well-established factual question.\n"
            "2. Use tavily_search for recent/live events, niche topics, or when wiki_search returns insufficient results. If the prompt mentions a specific date, include that exact date in your query.\n"
            "3. Use arxiv_search for academic papers, preprints, or scholarly authors in STEM.\n"
            "Always cross-check sources. Give precise, exact answers."
        ),
        tools=RESEARCH_TOOLS,
        llm=llm,
    )


def build_mathematician(llm: BaseChatModel | None = None):
    """Math agent: calculator + Python executor."""

    return build_agent_graph(
        name="Mathematician",
        system_prompt=(
            "You are a math and computation specialist.\n"
            "Solve problems step by step. Use the calculator for arithmetic\n"
            "and run_python for complex computations.\n"
            "Always double-check your work. Give exact numerical answers."
        ),
        tools=MATH_TOOLS,
        llm=llm,
    )


def build_file_analyst(llm: BaseChatModel | None = None):
    """File analysis agent: read/write files + Python executor."""
    return build_agent_graph(
        name="FileAnalyst",
        system_prompt=(
            "You are an expert file analysis and data processing agent.\n"
            "Your goal is to extract information, process data, and execute code accurately.\n\n"
            "IMPORTANT: The user message will contain 'FILE PATH: <path>' at the top.\n"
            "You MUST use that exact path when calling tools. Do NOT guess or modify the path.\n\n"
            "TOOL ROUTING RULES (by file extension):\n"
            "1. .mp3 / .wav / .ogg audio → Use `transcribe_audio` with the exact file path.\n"
            "2. .png / .jpg / .jpeg / .gif / .webp image → Use `analyze_image` with the exact file path and a detailed prompt.\n"
            "3. .xlsx / .xls spreadsheet → Use `run_python` to write a pandas script. Example:\n"
            "   import pandas as pd\n"
            "   df = pd.read_excel('/tmp/file.xlsx')\n"
            "   print(df.head())\n"
            "   You MUST use print() to output results.\n"
            "4. .py Python file → Use `execute_python` with the exact file path to run it and get its output.\n"
            "5. .txt / .md / .json / .csv text → Use `read_file` for small files. For large CSVs, use `run_python` with pandas.\n"
            "6. .pdf → Use `run_python` with PyPDF2 or pdfplumber to extract text.\n\n"
            "CRITICAL RULES:\n"
            "- Always check the file extension FIRST to pick the right tool.\n"
            "- For `run_python`, you MUST use print() to output results.\n"
            "- Never say 'I cannot access the file' — the file IS available at the given path.\n"
            "- Think step-by-step, but keep your final answers concise."
        ),
        tools=FILE_TOOLS,
        llm=llm,
    )


def build_generalist(llm: BaseChatModel | None = None):
    """
    General-purpose agent with access to all tools.
    Defaults to Gemini 2.5 flash for stronger reasoning.
    Falls back to Qwen if GOOGLE_API_KEY is not set.
    """

    return build_agent_graph(
        name="Generalist",
        system_prompt=(
            "You are a capable general-purpose assistant with strong reasoning.\n"
            "Analyze the question carefully, decide which tools to use,\n"
            "and find the precise answer. Think step by step.\n"
            "For GAIA tasks, answers must be exact — no extra commentary."
        ),
        tools=ALL_TOOLS,
        llm=llm,
    )