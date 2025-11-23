mkdir hr_assistant_ui
cd hr_assistant_ui

# Initialize a new Python project
uv init .

# Add dependencies
uv add streamlit httpx pydantic langchain langchain-mcp-adapters anyio
