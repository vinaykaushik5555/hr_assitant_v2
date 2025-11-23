from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, AsyncIterator

import anyio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools


# ---------------------------------------------------------------------
# MCP server configuration (FastMCP Cloud)
# ---------------------------------------------------------------------
# Adjust MCP_SERVER_NAME and MCP_SERVER_URL to match your deployment.
MCP_SERVER_NAME = "employee"
MCP_SERVER_URL = "https://employee.fastmcp.app/mcp"
MCP_TRANSPORT = "streamable_http"  # FastMCP remote HTTP transport

# Single shared MultiServerMCPClient instance
mcp_client = MultiServerMCPClient(
    {
        MCP_SERVER_NAME: {
            "url": MCP_SERVER_URL,
            "transport": MCP_TRANSPORT,
        }
    }
)


# ---------------------------------------------------------------------
# Helper: open a tool session
# ---------------------------------------------------------------------

@asynccontextmanager
async def tool_session() -> AsyncIterator[Dict[str, Any]]:
    """
    Opens a session to the MCP server and loads all tools.

    Returns:
        Dict[str, Tool]: mapping tool_name -> langchain Tool-like object.
    """
    async with mcp_client.session(MCP_SERVER_NAME) as session:
        tools = await load_mcp_tools(session)
        yield {tool.name: tool for tool in tools}


def parse_response(raw: Any) -> Dict[str, Any]:
    """
    Normalize MCP tool responses into a dict.

    Some MCP tools return a dict directly; others may return a JSON string.
    This helper handles both.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return {"success": False, "error_message": f"Invalid response: {raw}"}
    return {"success": False, "error_message": "Unsupported response format"}


# ---------------------------------------------------------------------
# Typed login result
# ---------------------------------------------------------------------

@dataclass
class LoginResult:
    success: bool
    token: str | None = None
    is_admin: bool = False
    name: str | None = None
    raw: Dict[str, Any] | None = None
    error_message: str | None = None


# ---------------------------------------------------------------------
# Async MCP calls (internal)
# ---------------------------------------------------------------------

async def _a_call_tool(tool_name: str, **kwargs) -> Dict[str, Any]:
    """
    Generic async helper to call an MCP tool by name and return parsed JSON.
    """
    async with tool_session() as tools:
        tool = tools.get(tool_name)
        if tool is None:
            return {"success": False, "error_message": f"Tool '{tool_name}' not found."}

        raw = await tool.ainvoke(kwargs)
        return parse_response(raw)


async def _a_mcp_login(username: str, password: str) -> LoginResult:
    """
    Async login via MCP 'login' tool.
    """
    res = await _a_call_tool("login", username=username, password=password)

    if not res.get("success"):
        return LoginResult(
            success=False,
            error_message=res.get("error_message", "Login failed"),
            raw=res,
        )

    data = res.get("data", {})
    return LoginResult(
        success=True,
        token=data.get("token"),
        is_admin=bool(data.get("is_admin", False)),
        name=data.get("name"),
        raw=res,
    )


async def _a_mcp_get_leave_balance(token: str) -> Dict[str, Any]:
    return await _a_call_tool("get_my_leave_balance", token=token)


async def _a_mcp_list_my_leave_requests(token: str) -> Dict[str, Any]:
    return await _a_call_tool("list_my_leave_requests", token=token)


async def _a_mcp_apply_leave(
    token: str,
    leave_type: str,
    days: float,
    start_date: str,
    reason: str,
) -> Dict[str, Any]:
    return await _a_call_tool(
        "apply_leave_for_me",
        token=token,
        leave_type=leave_type,
        days=days,
        start_date=start_date,
        reason=reason,
    )


# ---------------------------------------------------------------------
# Public sync wrappers used by Streamlit
# ---------------------------------------------------------------------

def mcp_login(username: str, password: str) -> LoginResult:
    """
    Synchronous wrapper around async MCP login.
    """
    return anyio.run(_a_mcp_login, username, password)


def mcp_get_leave_balance(token: str) -> Dict[str, Any]:
    """
    Synchronous wrapper to get current user's leave balance.
    """
    return anyio.run(_a_mcp_get_leave_balance, token)


def mcp_list_my_leave_requests(token: str) -> Dict[str, Any]:
    """
    Synchronous wrapper to list current user's leave requests.
    """
    return anyio.run(_a_mcp_list_my_leave_requests, token)


def mcp_apply_leave(
    token: str,
    leave_type: str,
    days: float,
    start_date: str,
    reason: str,
) -> Dict[str, Any]:
    """
    Synchronous wrapper to apply leave for current user.
    """
    return anyio.run(
        _a_mcp_apply_leave,
        token,
        leave_type,
        days,
        start_date,
        reason,
    )
# =====================================================================