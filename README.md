# HR AI Assistant System

Conversational HR Support with RAG + MCP Leave Management + LangGraph Agent

---

## 📌 Overview

The HR AI Assistant is a conversational system that enables employees to interact with HR policy information and leave management workflows through natural language.

It integrates RAG-based document search, MCP API tools, LangGraph orchestration, and Streamlit UI with role-based access (Employee / Admin).

- Employees can ask questions about HR policies, view policy library, apply for leave, and check leave balances.
- HR Admins can upload policy documents, manage versions, delete files, rebuild embeddings, and view summaries & diffs.

---

## 🚀 Key Features

| Capability                  | Description                                                                |
|-----------------------------|----------------------------------------------------------------------------|
| Conversational AI Assistant | Natural language HR support powered by LLM + LangGraph                    |
| RAG Policy Search           | Semantic search over uploaded HR policy docs via Chroma                   |
| Policy Upload & Versioning  | Admin manages PDF, TXT, MD, HTML files with auto summary & diff           |
| MCP Leave Management        | Apply leave, check balance, credit leave, list history                    |
| Guardrails Safety           | Filters abusive/toxic language & responds safely                          |
| Role-Based UI               | Employee vs Admin views dynamically rendered                              |
| LangSmith Tracing           | Monitoring agent reasoning & traces                                       |
| No Chat Persistence         | Clean session resets per conversation                                     |

---

## 🏗 System Architecture

flowchart TD

subgraph UI["🖥 Streamlit UI"]
Login
ChatAssistant
UploadPolicies
PolicyLibrary
end

subgraph Guard["🛡 Guardrails"]
InputFilter
OutputValidation
end

subgraph Agent["🧠 LangGraph Agent"]
IntentClassifier
WorkflowRouter
end

subgraph RAG["📚 RAG Engine"]
Chunker
Embeddings
ChromaDB
end

subgraph MCP["🔧 MCP Leave API"]
login
get_leave_balance
apply_leave
credit_leave
list_request
end

subgraph LLM["🤖 OpenAI GPT-4o"]
end

UI --> Guard --> Agent --> RAG --> LLM --> Agent --> Guard --> UI
Agent --> MCP --> UI
UI --> RAG

text

---

## 📂 Project Modules

/app
app.py # Streamlit UI
agent.py # LangGraph workflow logic
rag_engine.py # Embedding + Vector DB operations
guardrails.py # Toxic input/output filtering
policy_manager.py # Upload, delete, version, diff, summaries
mcp_client.py # LangChain-MCP adapter client implementation

text

MCP Leave API service is deployed externally via FastAPI + FastMCP.

---

## ⚙ MCP Tools Available

| Tool                         | Description                        |
|------------------------------|------------------------------------|
| login                        | Authenticate employee/admin        |
| get_leave_balance            | Balance per leave bucket           |
| apply_leave                  | Submit a leave request             |
| credit_leave                 | HR credit leave                    |
| list_employee_leave_requests | History of leave                   |
| initialize_employee_balance  | Default leave creation             |

---

## 👥 User Roles

| Role       | Capabilities                                                            |
|------------|------------------------------------------------------------------------|
| Employee   | Ask questions, search policies, apply leave, check balance             |
| Admin / HR | Manage policies (upload/delete/version), indexing, diff viewer         |

---

## 💬 Sample Conversations

**✔ Policy Q&A**  
User: What is the maternity leave policy?  
Assistant: According to HR Policy (section 2.3), maternity leave allows...

**✔ Leave Action**  
User: Apply 3 days sick leave starting Monday  
Assistant: You currently have 6 days sick leave. Confirm application?

**❌ Guardrails Example**  
User: you are stupid  
Assistant: Your message violates company policy.  
Sanitized: you are ~~stupid~~

---

## 🎞 Sequence Diagrams

**Policy Retrieval**

sequenceDiagram
User->>UI: Question about HR policy
UI->>Guard: Validate input
Guard->>Agent: Forward cleaned input
Agent->>RAG: Retrieve similar chunks
RAG->>LLM: Provide context
LLM->>Agent: Answer
Agent->>UI: Response

text

**Apply Leave Workflow**

sequenceDiagram
User->>UI: "Apply 3 days PL leave"
UI->>Guard: Validate
Guard->>Agent: Forward
Agent->>MCP: get_leave_balance
MCP->>Agent: balance
Agent->>User: Confirm?
User->>Agent: Yes
Agent->>MCP: apply_leave
MCP->>Agent: Result
Agent->>UI: Confirmation

text

---

## 🚧 Roadmap (Next Phases)

- Leave approval workflow
- Persistent chat history & CRM logging
- Email & Teams notifications
- Analytics dashboard for HR
- Optional cloud migration (Azure / AWS)

---

## 🧪 Tech Stack

| Layer         | Technology                         |
|---------------|------------------------------------|
| UI            | Streamlit                          |
| Orchestration | LangGraph                          |
| RAG           | Chroma + OpenAI embeddings         |
| Leave Backend | FastAPI + SQLite + FastMCP         |
| LLM           | OpenAI GPT-4o / GPT-4o-mini        |
| Monitoring    | LangSmith                          |
| Safety Filter | Custom Guardrails                  |

---

## 📦 Deployment

**Local UI**  
uv run streamlit run app.py

text

**MCP Server Execution**  
python main.py mcp

text

---

## ✨ Current Status

| Component            | Status                                 |
|----------------------|----------------------------------------|
| MCP Server           | ✔ deployed & stable                    |
| Streamlit UI         | ✔ working with admin/employee roles    |
| Policy Upload & RAG  | ✔ completed                            |
| Leave workflow       | ✔ working                              |
| Guardrails           | ✔ enabled                              |
| LangGraph agent flow | ✔ complete                             |
| Chat history         | ✖ intentionally disabled for now       |

---

## 🤝 Contributions

PRs, improvements and features welcome.

---

## 📄 License

Internal / Private — Not for public distribution

---

## 🏁 End of README
