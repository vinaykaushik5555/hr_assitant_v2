from __future__ import annotations

from typing import List, Tuple

import logging

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    BSHTMLLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langsmith import traceable
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from config import POLICY_DIR, POLICY_INDEX_DIR


logger = logging.getLogger(__name__)

# Name of the Chroma collection used for HR policies
COLLECTION_NAME = "hr_policies"


def _policy_index_has_data() -> bool:
    """Return True if the Chroma persist directory already has any files."""
    if not POLICY_INDEX_DIR.exists():
        return False
    return any(POLICY_INDEX_DIR.iterdir())


def _policy_dir_has_files() -> bool:
    """Return True if there are any file entries under POLICY_DIR."""
    if not POLICY_DIR.exists():
        return False
    return any(path.is_file() for path in POLICY_DIR.rglob("*"))


def _chroma_settings() -> Settings:
    """
    Shared Chroma client settings ensuring read/write access.
    """
    return Settings(
        allow_reset=True,
    )


def _load_policy_documents() -> List[Document]:
    """
    Load all policy documents from POLICY_DIR and return as LangChain Documents.

    Supported file types:
      - .pdf        -> PyPDFLoader
      - .txt, .md   -> TextLoader
      - .html, .htm -> BSHTMLLoader
    """
    docs: List[Document] = []
    if not POLICY_DIR.exists():
        return docs

    for path in POLICY_DIR.rglob("*"):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        if suffix == ".pdf":
            loader = PyPDFLoader(str(path))
            docs.extend(loader.load())
        elif suffix in {".txt", ".md"}:
            loader = TextLoader(str(path), encoding="utf-8")
            docs.extend(loader.load())
        elif suffix in {".html", ".htm"}:
            loader = BSHTMLLoader(str(path), open_encoding="utf-8")
            docs.extend(loader.load())

    return docs


def _get_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Create a text splitter for chunking policy documents into overlapping segments.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )


def _get_embeddings() -> OpenAIEmbeddings:
    """
    Create an OpenAI embeddings client.

    OPENAI_API_KEY is read from environment (.env).
    """
    return OpenAIEmbeddings(model="text-embedding-3-small")


@traceable(name="build_or_rebuild_vector_store")
def build_or_rebuild_vector_store() -> Chroma:
    """
    Build (or rebuild) the Chroma vector store from all policy files.

    Called after admin uploads or updates policy documents.
    Existing index contents are replaced.
    """
    docs = _load_policy_documents()
    if not docs:
        raise ValueError(f"No documents found in {POLICY_DIR}")

    splitter = _get_text_splitter()
    split_docs = splitter.split_documents(docs)

    embeddings = _get_embeddings()
    POLICY_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=str(POLICY_INDEX_DIR),
        collection_name=COLLECTION_NAME,
        client_settings=_chroma_settings(),
    )
    vectordb.persist()
    return vectordb


def get_vector_store() -> Chroma:
    """
    Load an existing Chroma vector store from disk.

    Used at query time by the chat assistant.
    """
    if not _policy_index_has_data():
        if _policy_dir_has_files():
            logger.info("Policy index not found; triggering auto rebuild before query.")
            return build_or_rebuild_vector_store()
        logger.warning("Policy index missing and no policy documents found.")
    embeddings = _get_embeddings()
    return Chroma(
        embedding_function=embeddings,
        persist_directory=str(POLICY_INDEX_DIR),
        collection_name=COLLECTION_NAME,
        client_settings=_chroma_settings(),
    )


def search_policies(query: str, k: int = 3):
    """
    Low-level search helper: return top-k (Document, score) tuples
    for the given query.
    """
    vectordb = get_vector_store()
    return vectordb.similarity_search_with_score(query, k=k)


@traceable(name="rag_answer_policy_question")
def answer_policy_question(query: str, k: int = 3) -> Tuple[str, List[Document]]:
    """
    High-level RAG helper used by the Chat tab.

    1. Retrieve top-k relevant chunks from the Chroma policy index.
    2. Call OpenAI chat model with the retrieved context.
    3. Return (answer_text, retrieved_documents).

    The answer is constrained to the provided context; the system prompt
    tells the model to say "not sure" if the information is not present.
    """
    vectordb = get_vector_store()
    docs_and_scores = vectordb.similarity_search_with_score(query, k=k)

    if not docs_and_scores:
        return "I couldn't find any policy text related to your question.", []

    docs = [d for d, _ in docs_and_scores]

    # Build context string with numbered snippets and sources
    context_blocks = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        source = meta.get("source", "unknown")
        context_blocks.append(f"[{i}] Source: {source}\n{doc.page_content.strip()}")

    context_text = "\n\n".join(context_blocks)

    system_prompt = (
        "You are an HR policy assistant for the company. "
        "Answer using ONLY the policy context provided. "
        "If the answer is not clearly in the context, say you are not sure. "
        "Always reference which snippet numbers you used, like [1], [2]."
    )

    user_prompt = (
        f"User question:\n{query}\n\n"
        f"Policy context:\n{context_text}\n\n"
        "Answer clearly and concisely."
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    resp = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    answer = resp.content if isinstance(resp.content, str) else str(resp.content)
    return answer, docs
