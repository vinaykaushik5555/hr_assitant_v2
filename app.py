from __future__ import annotations

from pathlib import Path
import os
from datetime import datetime

import streamlit as st
from langsmith import traceable

from mcp_client import mcp_login
from rag import build_or_rebuild_vector_store
from config import POLICY_DIR, POLICY_INDEX_DIR
from agent import agent_app
from guardrails_local import validate_input, validate_output


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
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages: list[dict] = []


# =====================================================================
# LOGIN PAGE
# =====================================================================
def login_page() -> None:
    """Render login screen and call MCP login."""
    st.title("HR Assistant - Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        with st.spinner("Authenticating..."):
            result = mcp_login(username, password)

        if not result.success:
            st.error(result.error_message or "Login failed")
            return

        st.session_state.token = result.token
        st.session_state.is_admin = result.is_admin
        st.session_state.user_name = result.name or username

        st.success("Login successful")
        st.rerun()


# =====================================================================
# MAIN PAGE
# =====================================================================
def main_page() -> None:
    """Main application after login with role-based access."""
    role_label = "HR Admin" if st.session_state.is_admin else "Employee"

    st.title("HR Assistant")
    st.write(f"Welcome **{st.session_state.user_name}** ({role_label})")

    # Sidebar with logout
    with st.sidebar:
        st.header("Session")
        st.write(f"User: {st.session_state.user_name}")
        st.write(f"Role: {role_label}")
        if st.button("Logout"):
            for key in list(st.session_state.keys()):
                st.session_state.pop(key)
            st.rerun()

    st.markdown("---")

    # Admin sees 3 tabs, employees see only chat
    if st.session_state.is_admin:
        tab_chat, tab_policies, tab_library = st.tabs(
            ["Chat Assistant", "Upload Policies", "Policy Library"]
        )
        with tab_chat:
            chat_tab()
        with tab_policies:
            upload_policies_tab()
        with tab_library:
            policy_library_tab()
    else:
        (tab_chat,) = st.tabs(["Chat Assistant"])
        with tab_chat:
            chat_tab()


# =====================================================================
# CHAT TAB WITH AGENT + GUARDRAILS
# =====================================================================

@traceable(name="agent_turn")
def _run_agent(messages, token):
    return agent_app.invoke({"messages": messages, "intent": "", "token": token})


def chat_tab() -> None:
    """Unified policy + leave chat agent."""
    st.subheader("Chat Assistant")

    for msg in st.session_state.chat_messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    user_input = st.chat_input("Ask anything about company policy or your leave...")
    if not user_input:
        return

    allowed, sanitized_or_error = validate_input(user_input)
    if not allowed:
        st.chat_message("assistant").markdown(sanitized_or_error)
        st.session_state.chat_messages.append({"role": "assistant", "content": sanitized_or_error})
        return

    sanitized_user = sanitized_or_error
    st.session_state.chat_messages.append({"role": "user", "content": sanitized_user})
    st.chat_message("user").markdown(sanitized_user)

    messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_messages]

    with st.spinner("Thinking..."):
        result = _run_agent(messages, st.session_state.token)

    raw = result["messages"][-1]["content"]
    assistant_msg = validate_output(raw)

    st.session_state.chat_messages.append({"role": "assistant", "content": assistant_msg})
    st.chat_message("assistant").markdown(assistant_msg)


# =====================================================================
# UPLOAD POLICIES TAB
# =====================================================================
def upload_policies_tab() -> None:
    """Upload & rebuild embeddings."""
    st.subheader("Upload HR Policy Documents")
    st.write("Supported formats: **.pdf**, **.txt**, **.md**, **.html**, **.htm**")
    uploaded_files = st.file_uploader("Upload Policy Files", type=["pdf", "txt", "md", "html", "htm"], accept_multiple_files=True)

    if uploaded_files and st.button("Upload & Rebuild Index"):
        POLICY_DIR.mkdir(parents=True, exist_ok=True)
        for f in uploaded_files:
            path = POLICY_DIR / f.name
            with path.open("wb") as out:
                out.write(f.getbuffer())

        st.success("Files uploaded successfully.")
        with st.spinner("Rebuilding vector index..."):
            build_or_rebuild_vector_store()
        st.success("Index rebuilt successfully.")


# =====================================================================
# POLICY LIBRARY TAB (VERSIONING + DELETE)
# =====================================================================
def policy_library_tab() -> None:
    """List uploaded policies, allow delete & clear embeddings."""
    st.subheader("Policy Library / Versioning")

    if not POLICY_DIR.exists() or not any(POLICY_DIR.iterdir()):
        st.info("No policy documents available yet.")
        return

    st.markdown("### 📚 Documents")

    files = sorted(POLICY_DIR.glob("*"), key=os.path.getmtime, reverse=True)

    for file in files:
        stat = file.stat()
        modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        size_kb = round(stat.st_size / 1024, 2)

        with st.container(border=True):
            st.markdown(f"**📄 {file.name}**")
            st.text(f"Modified: {modified} | Size: {size_kb} KB")

            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                with open(file, "rb") as f:
                    st.download_button("⬇ Download", f, file_name=file.name)

            with col2:
                if st.button("❌ Delete File", key=f"delete-{file.name}"):
                    os.remove(file)
                    st.warning(f"Deleted {file.name}")
                    st.rerun()

            with col3:
                pass

    st.markdown("---")
    st.markdown("### ⚙ Embeddings & Index Management")

    colA, colB = st.columns(2)
    with colA:
        if st.button("Rebuild Vector Index"):
            with st.spinner("Rebuilding..."):
                build_or_rebuild_vector_store()
            st.success("Index rebuilt successfully.")

    with colB:
        if st.button("🔥 Delete All Embeddings (Reset RAG)"):
            if POLICY_INDEX_DIR.exists():
                for item in POLICY_INDEX_DIR.rglob("*"):
                    if item.is_file():
                        item.unlink()
                for item in reversed(list(POLICY_INDEX_DIR.rglob("*"))):
                    if item.is_dir():
                        item.rmdir()
                POLICY_INDEX_DIR.rmdir()
            st.warning("All embeddings removed. RAG reset.")
            st.rerun()


# =====================================================================
# ENTRYPOINT
# =====================================================================
def main() -> None:
    st.set_page_config(page_title="HR Assistant", layout="wide")
    init_state()

    if st.session_state.token is None:
        login_page()
    else:
        main_page()


if __name__ == "__main__":
    main()
