from __future__ import annotations

import json
from datetime import date as dt_date
from typing import Dict, List, Literal, TypedDict

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langsmith import traceable


from rag import answer_policy_question
from mcp_client import (
    mcp_get_leave_balance,
    mcp_list_my_leave_requests,
    mcp_apply_leave,   # <-- make sure this is defined in mcp_client
)


# ---------------------------------------------------------------------
# Agent state definition
# ---------------------------------------------------------------------


class AgentState(TypedDict):
    """
    Shared state passed between LangGraph nodes.

    messages: full conversation as a list of {"role": "user"|"assistant", "content": str}
    intent:   last classified intent label
    token:    auth token for MCP calls (per logged-in user)
    """

    messages: List[Dict[str, str]]
    intent: str
    token: str


# ---------------------------------------------------------------------
# Helper: get the latest user message from state
# ---------------------------------------------------------------------


def _get_last_user_message(state: AgentState) -> str:
    """
    Return the most recent user message from the conversation.
    """
    for msg in reversed(state["messages"]):
        if msg["role"] == "user":
            return msg["content"]
    return ""


# ---------------------------------------------------------------------
# Node: intent classification
# ---------------------------------------------------------------------


def classify_intent(state: AgentState) -> AgentState:
    """
    Use an LLM to classify the user's intent.

    Possible intents:
      - 'policy_query'   -> ask about HR policies
      - 'leave_balance'  -> ask about current leave balance
      - 'leave_status'   -> ask about status/history of leave applications
      - 'leave_apply'    -> wants to apply for leave via conversation
      - 'other'          -> anything else / small talk
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    last_user = _get_last_user_message(state)

    system_prompt = (
        "You are an intent classifier for an HR assistant.\n"
        "Given ONLY the user's last message, choose EXACTLY one intent from:\n"
        "  - policy_query\n"
        "  - leave_balance\n"
        "  - leave_status\n"
        "  - leave_apply\n"
        "  - other\n\n"
        "If the user is clearly talking about applying for leave, dates, "
        "or asking to submit leave, choose 'leave_apply'.\n\n"
        "Return ONLY the intent label, nothing else."
    )

    resp = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": last_user},
        ]
    )

    intent = str(resp.content).strip().lower()
    if intent not in {
        "policy_query",
        "leave_balance",
        "leave_status",
        "leave_apply",
        "other",
    }:
        intent = "other"

    state["intent"] = intent
    return state


# ---------------------------------------------------------------------
# Node: handle policy queries via RAG
# ---------------------------------------------------------------------


def handle_policy_query(state: AgentState) -> AgentState:
    """
    RAG flow over HR policies.

    - Uses answer_policy_question() from rag.py
    - Appends an assistant message with answer + sources to state["messages"].
    """
    user_question = _get_last_user_message(state)

    answer, docs = answer_policy_question(user_question, k=3)

    # Build simple source list for transparency
    sources_text = ""
    if docs:
        parts = []
        for i, d in enumerate(docs, start=1):
            src = (d.metadata or {}).get("source", "unknown")
            parts.append(f"[{i}] {src}")
        sources_text = "\n\n**Sources:**\n" + "\n".join(parts)

    assistant_msg = answer + sources_text

    state["messages"].append({"role": "assistant", "content": assistant_msg})
    return state


# ---------------------------------------------------------------------
# Node: handle leave balance via MCP
# ---------------------------------------------------------------------


def handle_leave_balance(state: AgentState) -> AgentState:
    """
    Call MCP get_leave_balance and format a friendly message.
    """
    token = state["token"]
    res = mcp_get_leave_balance(token)

    if not res.get("success"):
        content = f"Failed to fetch your leave balance: {res.get('error_message', 'unknown error')}."
    else:
        balances = res["data"]["balances"]
        parts = [f"{lt}: {days}" for lt, days in balances.items()]
        content = "Your current leave balance is:\n\n- " + "\n- ".join(parts)

    state["messages"].append({"role": "assistant", "content": content})
    return state


# ---------------------------------------------------------------------
# Node: handle leave status via MCP
# ---------------------------------------------------------------------


def handle_leave_status(state: AgentState) -> AgentState:
    """
    Call MCP list_my_leave_requests and summarize last few requests.
    """
    token = state["token"]
    res = mcp_list_my_leave_requests(token)

    if not res.get("success"):
        content = f"Failed to fetch your leave requests: {res.get('error_message', 'unknown error')}."
        state["messages"].append({"role": "assistant", "content": content})
        return state

    data = res["data"]
    requests = data.get("requests", [])
    if not requests:
        content = "You don't have any leave applications on record."
        state["messages"].append({"role": "assistant", "content": content})
        return state

    lines = []
    for r in requests[:5]:  # show only the last few
        lt = r.get("leave_type")
        start = r.get("start_date")
        days = r.get("days")
        status = r.get("status")
        lines.append(f"- {lt} for {days} day(s) starting {start} – **{status}**")

    content = "Here are your recent leave applications:\n\n" + "\n".join(lines)
    state["messages"].append({"role": "assistant", "content": content})
    return state


# ---------------------------------------------------------------------
# Node: FULL conversational leave apply via MCP
# ---------------------------------------------------------------------


def _extract_leave_struct(state: AgentState) -> Dict[str, str | None]:
    """
    Use an LLM to extract leave application fields from the conversation.

    Returns a dict with keys:
      - leave_type: "CL" | "PL" | "ML" | "OTHER" | None
      - start_date: "YYYY-MM-DD" | None
      - end_date:   "YYYY-MM-DD" | None
      - reason:     str | None
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Build a compact conversation transcript for the model to inspect
    conv_lines = []
    # Limit to the last ~10 messages for brevity
    for m in state["messages"][-10:]:
        conv_lines.append(f"{m['role']}: {m['content']}")
    conv_text = "\n".join(conv_lines)

    system_prompt = (
        "You are a parser that extracts structured leave application data "
        "from the conversation.\n\n"
        "You MUST respond with a SINGLE JSON object only, no explanation.\n"
        "JSON keys:\n"
        "  - leave_type: one of ['CL', 'PL', 'ML', 'OTHER'] or null\n"
        "  - start_date: date in 'YYYY-MM-DD' format or null\n"
        "  - end_date: date in 'YYYY-MM-DD' format or null\n"
        "  - reason: string or null\n\n"
        "CL = casual leave, PL = privilege leave, ML = medical leave.\n"
        "If you are not sure about a field, set it to null."
    )

    user_prompt = (
        "Extract leave application fields from this conversation:\n\n"
        f"{conv_text}\n\n"
        "Return ONLY the JSON object, nothing else."
    )

    resp = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    raw = str(resp.content)
    try:
        data = json.loads(raw)
    except Exception:
        # If parsing fails, mark everything as missing
        return {
            "leave_type": None,
            "start_date": None,
            "end_date": None,
            "reason": None,
        }

    # Normalize keys
    return {
        "leave_type": data.get("leave_type"),
        "start_date": data.get("start_date"),
        "end_date": data.get("end_date"),
        "reason": data.get("reason"),
    }


def _parse_date_safe(value: str | None) -> dt_date | None:
    """
    Try to parse an ISO-like date string; return None on failure.
    """
    if not value:
        return None
    try:
        return dt_date.fromisoformat(value)
    except Exception:
        return None


def handle_leave_apply(state: AgentState) -> AgentState:
    """
    Fully conversational leave application handler.

    - Uses an LLM to extract leave_type, start_date, end_date, reason.
    - If any fields are missing, asks the user for the missing ones.
    - When all fields are present, computes number of days and calls MCP
      apply_leave, then returns a confirmation or error message.
    """
    token = state["token"]

    struct = _extract_leave_struct(state)
    leave_type = struct.get("leave_type")
    start_raw = struct.get("start_date")
    end_raw = struct.get("end_date")
    reason = struct.get("reason")

    # Parse dates to validate
    start_date = _parse_date_safe(start_raw)
    end_date = _parse_date_safe(end_raw)

    missing = []

    if leave_type not in {"CL", "PL", "ML", "OTHER"}:
        leave_type = None
        missing.append("leave type (CL, PL, ML, OTHER)")

    if start_date is None:
        missing.append("start date (YYYY-MM-DD)")

    if end_date is None:
        missing.append("end date (YYYY-MM-DD)")

    if not reason:
        missing.append("reason for your leave")

    if missing:
        # Ask user specifically for the missing info
        missing_str = "; ".join(missing)
        content = (
            "To submit your leave application, I still need the following:\n"
            f"- {missing_str}\n\n"
            "Please provide the missing details in your next message. "
            "You can give them together (for example: "
            "'Casual leave from 2025-12-10 to 2025-12-12 for a family function')."
        )
        state["messages"].append({"role": "assistant", "content": content})
        return state

    # At this point we have leave_type, start_date, end_date, reason
    # Compute number of days
    days = (end_date - start_date).days + 1
    if days <= 0:
        content = (
            "The end date seems to be before or equal to the start date. "
            "Please check the dates and try again."
        )
        state["messages"].append({"role": "assistant", "content": content})
        return state

    # Call MCP to apply leave
    res = mcp_apply_leave(
        token=token,
        leave_type=leave_type,
        days=float(days),
        start_date=start_date.isoformat(),
        reason=reason or "",
    )

    if not res.get("success"):
        content = (
            "I tried to submit your leave request but it failed.\n\n"
            f"Error: {res.get('error_message', 'unknown error')}"
        )
        state["messages"].append({"role": "assistant", "content": content})
        return state

    # Build success confirmation
    data = res.get("data", {})
    req = data.get("request") or {}
    status = req.get("status", "UNKNOWN")
    start_disp = req.get("start_date", start_date.isoformat())
    content = (
        "Your leave application has been submitted successfully.\n\n"
        f"- Leave type: {leave_type}\n"
        f"- Start date: {start_disp}\n"
        f"- End date: {end_date.isoformat()}\n"
        f"- Number of days: {days}\n"
        f"- Reason: {reason}\n"
        f"- Status: **{status}**\n"
    )

    state["messages"].append({"role": "assistant", "content": content})
    return state


# ---------------------------------------------------------------------
# Node: fallback for 'other' intent
# ---------------------------------------------------------------------


def handle_other(state: AgentState) -> AgentState:
    """
    Fallback node for chit-chat or unsupported intents.
    """
    content = (
        "I can help you with:\n"
        "- Questions about HR policies (leave, holidays, etc.)\n"
        "- Checking your leave balance\n"
        "- Viewing your recent leave applications\n"
        "- Applying for leave conversationally\n\n"
        "Please ask a policy-related question or something about your leave."
    )
    state["messages"].append({"role": "assistant", "content": content})
    return state


# ---------------------------------------------------------------------
# Routing function for conditional edges
# ---------------------------------------------------------------------


def route_after_classify(state: AgentState) -> Literal[
    "handle_policy_query",
    "handle_leave_balance",
    "handle_leave_status",
    "handle_leave_apply",
    "handle_other",
]:
    """
    Decide which node to go to next based on state['intent'].
    """
    intent = state.get("intent", "other")
    if intent == "policy_query":
        return "handle_policy_query"
    if intent == "leave_balance":
        return "handle_leave_balance"
    if intent == "leave_status":
        return "handle_leave_status"
    if intent == "leave_apply":
        return "handle_leave_apply"
    return "handle_other"


# ---------------------------------------------------------------------
# Build and compile the LangGraph app
# ---------------------------------------------------------------------

graph = StateGraph(AgentState)

graph.add_node("classify_intent", classify_intent)
graph.add_node("handle_policy_query", handle_policy_query)
graph.add_node("handle_leave_balance", handle_leave_balance)
graph.add_node("handle_leave_status", handle_leave_status)
graph.add_node("handle_leave_apply", handle_leave_apply)
graph.add_node("handle_other", handle_other)

# Entry point: classify intent first
graph.set_entry_point("classify_intent")

# Conditional routing based on intent
graph.add_conditional_edges("classify_intent", route_after_classify)

# All handler nodes end the graph
for node_name in [
    "handle_policy_query",
    "handle_leave_balance",
    "handle_leave_status",
    "handle_leave_apply",
    "handle_other",
]:
    graph.add_edge(node_name, END)

# Compiled LangGraph app used by the Streamlit UI
agent_app = graph.compile()
