from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from langsmith import traceable

from mcp_client import (
    mcp_login,
    mcp_logout,
    mcp_who_am_i,
    mcp_admin_list_employees,
    mcp_admin_create_employee,
)
from rag import build_or_rebuild_vector_store
from config import POLICY_DIR, POLICY_INDEX_DIR
from agent import agent_app
from guardrails_local import validate_input, validate_output
from logging_setup import configure_logging


configure_logging()
logger = logging.getLogger(__name__)


# ============================================================
# Utility: Load custom CSS
# ============================================================
def load_custom_css() -> None:
    """Inject custom CSS if assets/custom.css exists."""
    css_path = Path("assets/custom.css")
    if css_path.exists():
        css = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# =====================================================================
# Session State Initialization
# =====================================================================
def init_state() -> None:
    """Initialize Streamlit session state values."""
    if "token" not in st.session_state:
        st.session_state.token = None
    if "is_admin" not in st.session_state:
        st.session_state.is_admin = False
    if "user_name" not in st.session_state:
        st.session_state.user_name = None
    if "profile" not in st.session_state:
        st.session_state.profile = None
    if "profile_loaded" not in st.session_state:
        st.session_state.profile_loaded = False
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages: list[dict] = []
    if "employee_dir_cache" not in st.session_state:
        st.session_state.employee_dir_cache = None
    if "employee_dir_error" not in st.session_state:
        st.session_state.employee_dir_error = None


def _safe_mcp_call(func, *args, **kwargs) -> dict:
    """
    Execute an MCP client call and trap low-level exceptions.
    """
    try:
        return func(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - networking failures
        return {"success": False, "error_message": str(exc)}


def _update_session_profile(
    profile: dict | None,
    fallback_name: str | None = None,
    fallback_is_admin: bool | None = None,
) -> None:
    """
    Apply profile data (if any) to Streamlit session state.
    """
    st.session_state.profile = profile if profile else None
    st.session_state.profile_loaded = True

    if profile and profile.get("name"):
        st.session_state.user_name = profile["name"]
    elif fallback_name:
        st.session_state.user_name = fallback_name
    elif not st.session_state.user_name:
        st.session_state.user_name = "User"

    if profile is not None and "is_admin" in profile:
        st.session_state.is_admin = bool(profile["is_admin"])
    elif fallback_is_admin is not None:
        st.session_state.is_admin = bool(fallback_is_admin)


def ensure_profile_loaded() -> None:
    """
    Fetch profile details once per session after login so UI can show metadata.
    """
    if not st.session_state.token or st.session_state.profile_loaded:
        return

    res = _safe_mcp_call(mcp_who_am_i, st.session_state.token)
    if res.get("success"):
        logger.info("Profile loaded successfully for token.")
        _update_session_profile(
            res.get("data") or {},
            fallback_name=st.session_state.user_name,
            fallback_is_admin=st.session_state.is_admin,
        )
    else:
        logger.warning(
            "Failed to load profile: %s",
            res.get("error_message", "unknown error"),
        )
        _update_session_profile(
            None,
            fallback_name=st.session_state.user_name,
            fallback_is_admin=st.session_state.is_admin,
        )


# =====================================================================
# LOGIN PAGE
# =====================================================================
def login_page() -> None:
    """Render login screen and call MCP login."""
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        st.markdown(
            """
            <div style="text-align: center; padding-top: 2rem; padding-bottom: 1rem;">
                <h1 style="margin-bottom: 0.25rem;">HR Assistant</h1>
                <p style="opacity: 0.8; margin-top: 0;">Sign in to continue</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            with st.form("login_form"):
                st.markdown("### Login")
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Enter your password",
                )
                st.markdown(
                    "<div style='height: 0.5rem;'></div>",
                    unsafe_allow_html=True,
                )
                submit = st.form_submit_button("🔐 Sign in", use_container_width=True)

            if submit:
                if not username or not password:
                    st.error("Please enter both username and password.")
                    return

                logger.info("Login attempt for user '%s'", username)
                with st.spinner("Authenticating..."):
                    result = mcp_login(username, password)

                if not result.success:
                    logger.warning(
                        "Login failed for user '%s': %s",
                        username,
                        result.error_message or "unknown error",
                    )
                    st.error(result.error_message or "Login failed")
                    return

                logger.info("Login succeeded for user '%s'", username)
                st.session_state.token = result.token
                profile_payload = None
                if isinstance(result.raw, dict):
                    profile_payload = result.raw.get("data") or result.raw

                _update_session_profile(
                    profile_payload,
                    fallback_name=result.name or username,
                    fallback_is_admin=result.is_admin,
                )

                st.success("Login successful")
                st.rerun()

        st.caption("Your access is secured and role-based.")


# =====================================================================
# MAIN PAGE
# =====================================================================
def main_page() -> None:
    """Main application after login with role-based access."""
    role_label = "HR Admin" if st.session_state.is_admin else "Employee"
    profile = st.session_state.profile or {}

    # Top header
    with st.container():
        col_logo, col_title, col_gap, col_user = st.columns([0.7, 3, 1, 1.6])
        with col_logo:
            st.markdown("### 🧑‍💼")
        with col_title:
            st.markdown("## HR Assistant")
            st.caption("Ask about policies, leaves, and HR information.")
        with col_user:
            st.write(f"**{st.session_state.user_name}**")
            st.caption(role_label)

    st.divider()

    # Sidebar
    with st.sidebar:
        st.markdown("### Session")
        st.write(f"**User**: {st.session_state.user_name}")
        if profile.get("email"):
            st.caption(profile["email"])
        st.write(f"**Role**: {role_label}")
        if profile.get("department"):
            st.write(f"**Dept**: {profile['department']}")
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            if st.session_state.token:
                logger.info("Logout requested for user '%s'", st.session_state.user_name)
                with st.spinner("Signing out..."):
                    logout_res = _safe_mcp_call(mcp_logout, st.session_state.token)
                if not logout_res.get("success"):
                    msg = logout_res.get("error_message", "Logout failed.")
                    logger.warning("Logout failed: %s", msg)
                    st.warning(msg)
            for key in list(st.session_state.keys()):
                st.session_state.pop(key)
            st.rerun()

        st.markdown("---")
        st.caption("HR Assistant • RAG powered")

    # Tabs
    if st.session_state.is_admin:
        tab_chat, tab_policies, tab_library, tab_employees = st.tabs(
            [
                "💬 Chat Assistant",
                "📤 Upload Policies",
                "📚 Policy Library",
                "👥 Employee Admin",
            ]
        )
        with tab_chat:
            chat_tab()
        with tab_policies:
            upload_policies_tab()
        with tab_library:
            policy_library_tab()
        with tab_employees:
            employee_admin_tab()
    else:
        (tab_chat,) = st.tabs(["💬 Chat Assistant"])
        with tab_chat:
            chat_tab()


# =====================================================================
# CHAT TAB WITH AGENT + GUARDRAILS
# =====================================================================

@traceable(name="agent_turn")
def _run_agent(messages, token, is_admin):
    return agent_app.invoke(
        {
            "messages": messages,
            "intent": "",
            "token": token,
            "is_admin": is_admin,
        }
    )


def chat_tab() -> None:
    """
    Chat layout similar to ChatGPT:
    - Messages flow down the page.
    - Input box fixed at the bottom of the page by Streamlit.
    """
    st.markdown("#### Chat Assistant")
    st.caption("Ask about company policies, leave, and HR processes.")

    # Render previous messages (no fixed-height container -> no empty area)
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input stays at bottom; Streamlit keeps it there automatically
    user_input = st.chat_input("Type your question here...")
    if not user_input:
        return

    # Guardrail: input validation
    allowed, sanitized_or_error = validate_input(user_input)
    if not allowed:
        logger.info("Guardrail blocked user input: %s", sanitized_or_error)
        st.chat_message("assistant").markdown(sanitized_or_error)
        st.session_state.chat_messages.append(
            {"role": "assistant", "content": sanitized_or_error}
        )
        return

    sanitized_user = sanitized_or_error
    logger.info("User message accepted: %s", sanitized_user)

    # Append and display user message
    st.session_state.chat_messages.append({"role": "user", "content": sanitized_user})
    with st.chat_message("user"):
        st.markdown(sanitized_user)

    # Prepare messages for agent
    messages = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.chat_messages
    ]

    # Agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = _run_agent(
                messages,
                st.session_state.token,
                st.session_state.is_admin,
            )

        raw = result["messages"][-1]["content"]
        assistant_msg = validate_output(raw)
        logger.info("Agent response ready; content length=%d", len(assistant_msg))
        st.markdown(assistant_msg)

    # Store assistant message
    st.session_state.chat_messages.append(
        {"role": "assistant", "content": assistant_msg}
    )


# =====================================================================
# UPLOAD POLICIES TAB
# =====================================================================
def upload_policies_tab() -> None:
    """Upload & rebuild embeddings."""
    st.markdown("#### Upload HR Policy Documents")
    st.caption("Supported formats: PDF, TXT, MD, HTML, HTM")

    with st.container(border=True):
        uploaded_files = st.file_uploader(
            "Upload policy files",
            type=["pdf", "txt", "md", "html", "htm"],
            accept_multiple_files=True,
        )

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2])
        with col1:
            disabled = not uploaded_files
            trigger = st.button(
                "📥 Upload & Rebuild Index",
                type="primary",
                use_container_width=True,
                disabled=disabled,
            )

        if uploaded_files and trigger:
            POLICY_DIR.mkdir(parents=True, exist_ok=True)
            for f in uploaded_files:
                path = POLICY_DIR / f.name
                with path.open("wb") as out:
                    out.write(f.getbuffer())
                logger.info("Uploaded policy file '%s'", f.name)

            st.success("Files uploaded successfully.")
            with st.spinner("Rebuilding vector index..."):
                logger.info("Rebuilding vector index after upload (%d files)", len(uploaded_files))
                build_or_rebuild_vector_store()
            st.success("Index rebuilt successfully.")


# =====================================================================
# POLICY LIBRARY TAB (VERSIONING + DELETE)
# =====================================================================
def policy_library_tab() -> None:
    """List uploaded policies, allow delete & clear embeddings."""
    st.markdown("#### Policy Library & Versioning")

    if not POLICY_DIR.exists() or not any(POLICY_DIR.iterdir()):
        st.info("No policy documents available yet. Upload policies in the previous tab.")
        return

    st.markdown("##### Documents")

    files = sorted(POLICY_DIR.glob("*"), key=os.path.getmtime, reverse=True)

    for file in files:
        stat = file.stat()
        modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        size_kb = round(stat.st_size / 1024, 2)

        with st.container(border=True):
            st.markdown(f"**📄 {file.name}**")
            st.caption(f"Modified: {modified} · Size: {size_kb} KB")

            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                with open(file, "rb") as f:
                    st.download_button(
                        "⬇ Download",
                        f,
                        file_name=file.name,
                        use_container_width=True,
                    )

            with col2:
                if st.button(
                    "❌ Delete",
                    key=f"delete-{file.name}",
                    use_container_width=True,
                ):
                    os.remove(file)
                    st.warning(f"Deleted {file.name}")
                    st.rerun()

    st.divider()
    st.markdown("##### Embeddings & Index Management")

    colA, colB = st.columns(2)
    with colA:
        if st.button("🔁 Rebuild Vector Index", use_container_width=True):
            with st.spinner("Rebuilding vector index..."):
                logger.info("Manual rebuild of vector index triggered.")
                build_or_rebuild_vector_store()
            st.success("Index rebuilt successfully.")

    with colB:
        if st.button(
            "🔥 Delete All Embeddings (Reset RAG)",
            use_container_width=True,
        ):
            if POLICY_INDEX_DIR.exists():
                logger.warning("Deleting all embeddings from %s", POLICY_INDEX_DIR)
                for item in POLICY_INDEX_DIR.rglob("*"):
                    if item.is_file():
                        item.unlink()
                for item in reversed(list(POLICY_INDEX_DIR.rglob("*"))):
                    if item.is_dir():
                        item.rmdir()
                POLICY_INDEX_DIR.rmdir()
            st.warning("All embeddings removed. RAG reset.")
            st.rerun()


def employee_admin_tab() -> None:
    """Admin-only employee management panel."""
    st.markdown("#### Employee Directory & Onboarding")
    st.caption("Review existing employees and create new accounts.")

    token = st.session_state.token
    if not token:
        st.info("You must be logged in to view employee data.")
        return

    if "employee_dir_cache" not in st.session_state:
        st.session_state.employee_dir_cache = None
    if "employee_dir_error" not in st.session_state:
        st.session_state.employee_dir_error = None

    with st.container(border=True):
        st.markdown("##### Directory")
        col_btn, col_status = st.columns([1, 2])
        with col_btn:
            refresh = st.button("🔄 Refresh Directory", use_container_width=True)
        with col_status:
            if st.session_state.employee_dir_cache:
                cached_total = st.session_state.employee_dir_cache.get("total", 0)
                st.caption(f"Showing cached data ({cached_total} employees).")
            else:
                st.caption("No data loaded yet.")

        if refresh:
            with st.spinner("Fetching employees..."):
                logger.info("Refreshing employee directory from MCP.")
                res = _safe_mcp_call(mcp_admin_list_employees, token)

            if not res.get("success"):
                st.session_state.employee_dir_error = res.get(
                    "error_message", "Unable to load employees."
                )
                logger.warning("Employee directory fetch failed: %s", st.session_state.employee_dir_error)
                st.session_state.employee_dir_cache = None
            else:
                data = res.get("data")
                if isinstance(data, dict):
                    employees_raw = data.get("employees") or data.get("items") or []
                elif isinstance(data, list):
                    employees_raw = data
                else:
                    employees_raw = []

                rows: list[dict[str, str]] = []
                for emp in employees_raw:
                    if not isinstance(emp, dict):
                        continue
                    emp_id = emp.get("id") or emp.get("employee_id") or "—"
                    rows.append(
                        {
                            "ID": emp_id,
                            "Username": emp.get("username") or "—",
                            "Name": emp.get("name") or "—",
                            "Email": emp.get("email") or "—",
                            "Department": emp.get("department") or "—",
                            "Admin": "Yes" if emp.get("is_admin") else "No",
                        }
                    )

                st.session_state.employee_dir_cache = {
                    "rows": rows,
                    "total": len(employees_raw),
                }
                logger.info("Loaded %d employees into cache", len(employees_raw))
                st.session_state.employee_dir_error = None

        if st.session_state.employee_dir_error:
            st.error(st.session_state.employee_dir_error)
        elif st.session_state.employee_dir_cache and st.session_state.employee_dir_cache.get("rows"):
            rows = st.session_state.employee_dir_cache["rows"]
            st.dataframe(rows, hide_index=True, use_container_width=True)
            total = st.session_state.employee_dir_cache.get("total", len(rows))
            if total > len(rows):
                st.caption(f"Showing {len(rows)} of {total} employees.")
        else:
            st.info("Click **Refresh Directory** to load employee data.")

    st.divider()
    st.markdown("##### Create Employee")

    with st.form("create_employee_form"):
        col1, col2 = st.columns(2)
        with col1:
            new_id = st.text_input("Employee ID *", placeholder="EMP001")
            new_username = st.text_input("Username *", placeholder="jdoe")
            new_name = st.text_input("Full Name *", placeholder="Jane Doe")
        with col2:
            new_email = st.text_input("Email *", placeholder="jane@example.com")
            new_department = st.text_input("Department")
            new_password = st.text_input(
                "Temporary Password *",
                type="password",
                placeholder="Choose a secure password",
            )
        submitted = st.form_submit_button("Create Employee", use_container_width=True)

    if not submitted:
        return

    required_fields = [new_id, new_username, new_password, new_name, new_email]
    if not all(required_fields):
        st.error("Please complete all required fields marked with *.")
        return

    with st.spinner("Creating employee..."):
        logger.info("Creating employee id=%s username=%s", new_id.strip(), new_username.strip())
        create_res = _safe_mcp_call(
            mcp_admin_create_employee,
            token,
            new_id.strip(),
            new_username.strip(),
            new_password,
            new_name.strip(),
            new_email.strip(),
            new_department.strip() if new_department else "",
        )

    if not create_res.get("success"):
        msg = create_res.get("error_message", "Employee creation failed.")
        logger.warning("Employee creation failed: %s", msg)
        st.error(msg)
        return

    st.success("Employee created successfully.")
    logger.info("Employee %s created successfully.", new_id.strip())
    st.rerun()


# =====================================================================
# ENTRYPOINT
# =====================================================================
def main() -> None:
    st.set_page_config(
        page_title="HR Assistant",
        page_icon="🧑‍💼",
        layout="wide",
    )
    load_custom_css()
    init_state()

    if st.session_state.token is not None:
        ensure_profile_loaded()

    if st.session_state.token is None:
        login_page()
    else:
        main_page()


if __name__ == "__main__":
    main()
