from __future__ import annotations

import json
from datetime import date as dt_date
from typing import Dict, List, Literal, TypedDict

import logging
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langsmith import traceable


from rag import answer_policy_question
from mcp_client import (
    mcp_get_leave_balance,
    mcp_list_my_leave_requests,
    mcp_apply_leave,
    mcp_who_am_i,
    mcp_admin_list_employees,
    mcp_admin_create_employee,
)
from timesheet_operations import acknowledge_timesheet_request
from transport_operations import acknowledge_transport_request
from training_operations import acknowledge_training_status


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Agent state definition
# ---------------------------------------------------------------------


class AgentState(TypedDict):
    """
    Shared state passed between LangGraph nodes.

    messages: full conversation as a list of {"role": "user"|"assistant", "content": str}
    intent:   last classified intent label
    token:    auth token for MCP calls (per logged-in user)
    is_admin: whether the logged-in user has HR admin privileges
    """

    messages: List[Dict[str, str]]
    intent: str
    token: str
    is_admin: bool


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


def _keyword_intent_override(message: str) -> str | None:
    """
    Lightweight keyword routing so helper prompts map to the correct intent
    even before hitting the classifier LLM.
    """
    text = message.lower().strip()
    if not text:
        return None

    def contains_all(*words: str) -> bool:
        return all(word in text for word in words)

    if "timesheet" in text or contains_all("worked", "hours"):
        return "timesheet_entry"
    if "cab" in text or "pickup" in text or "transport" in text:
        return "transport_booking"
    if "training" in text and ("pending" in text or "overdue" in text or "status" in text):
        return "training_status"
    if "leave balance" in text:
        return "leave_balance"
    if contains_all("leave", "recent") or "leave requests" in text:
        return "leave_status"
    if "apply" in text and "leave" in text:
        return "leave_apply"
    if "policy" in text:
        return "policy_query"
    if "profile" in text:
        return "profile_info"
    if contains_all("create", "employee"):
        return "admin_create_employee"
    if "employee directory" in text or "list employees" in text:
        return "admin_list_employees"

    return None


# ---------------------------------------------------------------------
# Node: intent classification
# ---------------------------------------------------------------------


def classify_intent(state: AgentState) -> AgentState:
    """
    Use an LLM to classify the user's intent.

    Possible intents:
      - 'policy_query'        -> ask about HR policies
      - 'leave_balance'       -> ask about current leave balance
      - 'leave_status'        -> ask about status/history of leave applications
      - 'leave_apply'         -> wants to apply for leave via conversation
      - 'profile_info'        -> wants to know details about themselves / their role
      - 'admin_list_employees'-> HR admin wants roster/directory data
      - 'admin_create_employee'-> HR admin wants to onboard/create an employee
      - 'timesheet_entry'      -> capture/fill working hours or project codes
      - 'transport_booking'    -> arrange pickup/drop transport
      - 'training_status'      -> show mandatory/pending/overdue trainings
      - 'other'               -> anything else / small talk
    """
    last_user = _get_last_user_message(state)

    override_intent = _keyword_intent_override(last_user)
    if override_intent:
        state["intent"] = override_intent
        logger.info("Keyword override classified intent as '%s' for message '%s'", override_intent, last_user)
        return state

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    system_prompt = (
        "You are an intent classifier for an HR assistant.\n"
        "Given ONLY the user's last message, choose EXACTLY one intent from:\n"
        "  - policy_query\n"
        "  - leave_balance\n"
        "  - leave_status\n"
        "  - leave_apply\n"
        "  - profile_info\n"
        "  - admin_list_employees\n"
        "  - admin_create_employee\n"
        "  - timesheet_entry\n"
        "  - transport_booking\n"
        "  - training_status\n"
        "  - other\n\n"
        "If the user is clearly talking about applying for leave, dates, "
        "or asking to submit leave, choose 'leave_apply'.\n"
        "If the user wants to know their own profile/role/token information, "
        "choose 'profile_info'.\n"
        "If the user wants to list or review employees, choose 'admin_list_employees'.\n"
        "If the user wants to onboard or create an employee account, choose "
        "'admin_create_employee'.\n"
        "If the user is describing time logging, hours worked, or project codes, "
        "choose 'timesheet_entry'.\n"
        "If the user is asking for a cab/transport pickup or drop, choose 'transport_booking'.\n"
        "If the user wants to know pending or overdue trainings, choose 'training_status'.\n\n"
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
        "profile_info",
        "admin_list_employees",
        "admin_create_employee",
        "timesheet_entry",
        "transport_booking",
        "training_status",
        "other",
    }:
        intent = "other"

    state["intent"] = intent
    logger.info("Classified intent as '%s' for message '%s'", intent, last_user)
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
        err = res.get("error_message", "unknown error")
        logger.warning("Leave balance fetch failed: %s", err)
        content = f"Failed to fetch your leave balance: {err}."
    else:
        balances = res["data"]["balances"]
        parts = [f"{lt}: {days}" for lt, days in balances.items()]
        content = "Your current leave balance is:\n\n- " + "\n- ".join(parts)
        logger.info("Leave balance retrieved successfully.")

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
        err = res.get("error_message", "unknown error")
        logger.warning("Leave status fetch failed: %s", err)
        content = f"Failed to fetch your leave requests: {err}."
        state["messages"].append({"role": "assistant", "content": content})
        return state

    data = res["data"]
    requests = data.get("requests", [])
    if not requests:
        content = "You don't have any leave applications on record."
        logger.info("User has no leave requests on record.")
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
    logger.info("Leave status summary prepared with %d records.", len(lines))
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

    today = dt_date.today().isoformat()

    system_prompt = (
        "You are a parser that extracts structured leave application data "
        "from the conversation.\n\n"
        f"Today's date is {today}. Whenever the user mentions relative dates "
        "like 'today', 'tomorrow', 'next Monday', or 'day after tomorrow', "
        "convert them into absolute YYYY-MM-DD values referenced from today's "
        "date. If the user supplies natural language calendar dates such as "
        "'25th of Dec', 'December 25', or 'next Thursday in April', convert "
        "them as well. When a month/day is given without a year, assume the "
        "next occurrence of that date (use the current year if it hasn't "
        "passed yet, otherwise use the next year).\n\n"
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


def _extract_employee_struct(state: AgentState) -> Dict[str, str | None]:
    """
    Use an LLM to extract employee onboarding data from conversation.

    Expected keys:
      - employee_id
      - username
      - password
      - name
      - email
      - department (optional)
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    conv_lines = []
    for m in state["messages"][-10:]:
        conv_lines.append(f"{m['role']}: {m['content']}")
    conv_text = "\n".join(conv_lines)

    system_prompt = (
        "You extract structured employee onboarding fields from the conversation.\n"
        "Respond with a SINGLE JSON object only.\n"
        "JSON keys:\n"
        "  - employee_id (string or null)\n"
        "  - username (string or null)\n"
        "  - password (string or null)\n"
        "  - name (string or null)\n"
        "  - email (string or null)\n"
        "  - department (string or null)\n"
        "If a field is unknown, set it to null."
    )

    user_prompt = (
        "Extract employee onboarding fields from this conversation:\n\n"
        f"{conv_text}\n\n"
        "Return ONLY the JSON object."
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
        return {
            "employee_id": None,
            "username": None,
            "password": None,
            "name": None,
            "email": None,
            "department": None,
        }

    return {
        "employee_id": data.get("employee_id"),
        "username": data.get("username"),
        "password": data.get("password"),
        "name": data.get("name"),
        "email": data.get("email"),
        "department": data.get("department"),
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
        logger.info("Awaiting additional leave info: %s", missing_str)
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
        logger.warning("Invalid leave dates provided: start=%s end=%s", start_date, end_date)
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
        logger.error("MCP apply_leave failed: %s", res.get("error_message", "unknown error"))
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
    logger.info(
        "Leave application submitted: type=%s days=%d status=%s",
        leave_type,
        days,
        status,
    )
    return state


def handle_profile_info(state: AgentState) -> AgentState:
    """
    Call MCP who_am_i to display the user's profile details.
    """
    token = state["token"]
    res = mcp_who_am_i(token)

    if not res.get("success"):
        err = res.get("error_message", "unknown error")
        logger.warning("who_am_i failed: %s", err)
        content = f"Couldn't load your profile: {err}."
        state["messages"].append({"role": "assistant", "content": content})
        return state

    data = res.get("data")
    profile: Dict[str, str] = {}
    if isinstance(data, dict):
        profile = data.get("employee") or data.get("profile") or data
    elif isinstance(res, dict):
        # Some servers wrap the data in the top-level response instead of `data`
        profile = res.get("employee") or res.get("profile") or res

    if not isinstance(profile, dict):
        profile = {}

    name = profile.get("name") or profile.get("full_name") or "Unknown user"
    email = profile.get("email") or "Not provided"
    dept = profile.get("department") or profile.get("dept") or "Not provided"
    emp_id = (
        profile.get("id")
        or profile.get("employee_id")
        or profile.get("emp_id")
        or "Not provided"
    )
    role = "HR Admin" if profile.get("is_admin") else "Employee"

    content = (
        "Here's what I know about you:\n\n"
        f"- Name: {name}\n"
        f"- Employee ID: {emp_id}\n"
        f"- Role: {role}\n"
        f"- Email: {email}\n"
        f"- Department: {dept}"
    )
    state["messages"].append({"role": "assistant", "content": content})
    logger.info("Profile info shared with user (%s).", role)
    return state


def handle_admin_list_employees(state: AgentState) -> AgentState:
    """
    Allow admins to view a snapshot of the employee directory.
    """
    if not state.get("is_admin"):
        content = "Only HR admins can access the employee directory."
        logger.warning("Non-admin user attempted to access employee list.")
        state["messages"].append({"role": "assistant", "content": content})
        return state

    token = state["token"]
    res = mcp_admin_list_employees(token)

    if not res.get("success"):
        err = res.get("error_message", "unknown error")
        logger.warning("admin_list_employees failed: %s", err)
        content = f"Failed to fetch employees: {err}."
        state["messages"].append({"role": "assistant", "content": content})
        return state

    data = res.get("data")
    if isinstance(data, dict):
        employees = data.get("employees") or data.get("items") or []
    elif isinstance(data, list):
        employees = data
    else:
        employees = []

    if not employees:
        content = "The directory is empty or unavailable."
        logger.info("Employee directory returned no records.")
        state["messages"].append({"role": "assistant", "content": content})
        return state

    lines = []
    for emp in employees[:5]:
        if not isinstance(emp, dict):
            continue
        emp_name = emp.get("name") or emp.get("username") or "Unknown"
        emp_id = emp.get("id") or emp.get("employee_id") or "N/A"
        email = emp.get("email") or "No email"
        dept = emp.get("department") or "General"
        admin_flag = " (admin)" if emp.get("is_admin") else ""
        lines.append(f"- {emp_name}{admin_flag} — ID: {emp_id} — {dept} — {email}")

    content = "Here are the latest employees:\n\n" + "\n".join(lines)
    if len(employees) > 5:
        content += f"\n\n...and {len(employees) - 5} more."

    state["messages"].append({"role": "assistant", "content": content})
    logger.info("Provided employee snapshot with %d entries.", len(lines))
    return state


def handle_admin_create_employee(state: AgentState) -> AgentState:
    """
    Conversational onboarding for creating a new employee (admin only).
    """
    if not state.get("is_admin"):
        content = "You need HR admin permissions to create new employees."
        logger.warning("Non-admin attempted to create employee.")
        state["messages"].append({"role": "assistant", "content": content})
        return state

    token = state["token"]
    struct = _extract_employee_struct(state)

    employee_id = (struct.get("employee_id") or "").strip() or None
    username = (struct.get("username") or "").strip() or None
    password = struct.get("password") or None
    name = (struct.get("name") or "").strip() or None
    email = (struct.get("email") or "").strip() or None
    department = (struct.get("department") or "").strip() if struct.get("department") else ""

    missing = []
    if not employee_id:
        missing.append("employee ID")
    if not username:
        missing.append("username")
    if not password:
        missing.append("temporary password")
    if not name:
        missing.append("full name")
    if not email:
        missing.append("email address")

    if missing:
        missing_str = ", ".join(missing)
        content = (
            "To create the employee I still need the following details: "
            f"{missing_str}. Please provide them (you can share everything in one message)."
        )
        state["messages"].append({"role": "assistant", "content": content})
        logger.info("Awaiting missing employee fields: %s", missing_str)
        return state

    res = mcp_admin_create_employee(
        token,
        employee_id,
        username,
        password,
        name,
        email,
        department or "",
    )

    if not res.get("success"):
        err = res.get("error_message", "unknown error")
        logger.error("Employee creation via MCP failed: %s", err)
        content = f"I couldn't create the employee: {err}."
        state["messages"].append({"role": "assistant", "content": content})
        return state

    data = res.get("data")
    if isinstance(data, dict):
        created = data.get("employee") or data
    else:
        created = {}

    emp_id = created.get("id") or employee_id
    dept = created.get("department") or department or "General"
    content = (
        "Employee created successfully:\n\n"
        f"- ID: {emp_id}\n"
        f"- Username: {username}\n"
        f"- Name: {name}\n"
        f"- Email: {email}\n"
        f"- Department: {dept}\n"
        "They can now log in with the provided temporary password."
    )
    state["messages"].append({"role": "assistant", "content": content})
    logger.info("Employee %s/%s created via agent.", emp_id, username)
    return state


# ---------------------------------------------------------------------
# Node: HR operations placeholders (timesheet, transport, training)
# ---------------------------------------------------------------------


def handle_timesheet_entry(state: AgentState) -> AgentState:
    """
    Dummy node that records a timesheet request and returns a positive response.
    """
    user_message = _get_last_user_message(state)
    result = acknowledge_timesheet_request(user_message)
    logger.info("Timesheet placeholder handled with metadata=%s", result.metadata)
    state["messages"].append({"role": "assistant", "content": result.message})
    return state


def handle_transport_booking(state: AgentState) -> AgentState:
    """
    Dummy node for transport/cab bookings.
    """
    user_message = _get_last_user_message(state)
    result = acknowledge_transport_request(user_message)
    logger.info("Transport placeholder handled with metadata=%s", result.metadata)
    state["messages"].append({"role": "assistant", "content": result.message})
    return state


def handle_training_status(state: AgentState) -> AgentState:
    """
    Dummy node that reports pending/overdue trainings.
    """
    user_message = _get_last_user_message(state)
    result = acknowledge_training_status(user_message)
    logger.info("Training placeholder handled with metadata=%s", result.metadata)
    state["messages"].append({"role": "assistant", "content": result.message})
    return state


# ---------------------------------------------------------------------
# Node: fallback for 'other' intent
# ---------------------------------------------------------------------


def handle_other(state: AgentState) -> AgentState:
    """
    Fallback node for chit-chat or unsupported intents.
    """
    content = (
        "Here’s what I can help with right now:\n"
        "- Questions about HR policies (leave, holidays, etc.)\n"
        "- Checking your leave balance or recent leave requests\n"
        "- Applying for leave conversationally\n"
        "- Logging your timesheet details (placeholder flow)\n"
        "- Booking or updating a cab pickup/drop (placeholder flow)\n"
        "- Checking pending or overdue trainings (placeholder flow)\n"
        "- Showing your profile details\n"
        "- HR admin tasks like viewing or onboarding employees\n\n"
        "Let me know which of these you’d like to do."
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
    "handle_profile_info",
    "handle_admin_list_employees",
    "handle_admin_create_employee",
    "handle_timesheet_entry",
    "handle_transport_booking",
    "handle_training_status",
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
    if intent == "profile_info":
        return "handle_profile_info"
    if intent == "admin_list_employees":
        return "handle_admin_list_employees"
    if intent == "admin_create_employee":
        return "handle_admin_create_employee"
    if intent == "timesheet_entry":
        return "handle_timesheet_entry"
    if intent == "transport_booking":
        return "handle_transport_booking"
    if intent == "training_status":
        return "handle_training_status"
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
graph.add_node("handle_profile_info", handle_profile_info)
graph.add_node("handle_admin_list_employees", handle_admin_list_employees)
graph.add_node("handle_admin_create_employee", handle_admin_create_employee)
graph.add_node("handle_timesheet_entry", handle_timesheet_entry)
graph.add_node("handle_transport_booking", handle_transport_booking)
graph.add_node("handle_training_status", handle_training_status)
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
    "handle_profile_info",
    "handle_admin_list_employees",
    "handle_admin_create_employee",
    "handle_timesheet_entry",
    "handle_transport_booking",
    "handle_training_status",
    "handle_other",
]:
    graph.add_edge(node_name, END)

# Compiled LangGraph app used by the Streamlit UI
agent_app = graph.compile()
