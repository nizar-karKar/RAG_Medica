

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from rank_bm25 import BM25Okapi
import os
import re
from typing import Optional


def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer, lowercased."""
    return re.findall(r"\w+", text.lower())


def keyword_retrieve(
    query: str,
    vector_store_path: str,
    k: int = 1,
    filename: Optional[str] = None,
) -> list[dict]:
    """
    Retrieve documents from ChromaDB using BM25 keyword matching.

    Args:
        query: The user's question.
        vector_store_path: Path to the ChromaDB directory.
        k: Number of top results to return.
        filename: Optional metadata filter to restrict to a specific file.

    Returns:
        A list of dicts with keys: 'content', 'metadata', 'score'.
    """

    # ── Load all documents from ChromaDB ──────────────────────────────────
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma(
        collection_name="medical-local-rag",
        embedding_function=embedding_model,
        persist_directory=vector_store_path,
    )

    collection = vector_store._collection

    if filename:
        raw = collection.get(
            where={"filename": filename},
            include=["documents", "metadatas"],
        )
    else:
        raw = collection.get(include=["documents", "metadatas"])

    documents = raw["documents"]
    metadatas = raw["metadatas"]

    if not documents:
        return []

    # ── Build BM25 index ──────────────────────────────────────────────────
    tokenized_docs = [tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)

    # ── Score the query against all documents ─────────────────────────────
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    # ── Rank and return top-k ─────────────────────────────────────────────
    scored_results = []
    for idx, score in enumerate(scores):
        scored_results.append({
            "content": documents[idx],
            "metadata": metadatas[idx],
            "score": float(score),
        })

    # Sort by score descending (higher = more relevant in BM25)
    scored_results.sort(key=lambda x: x["score"], reverse=True)

    return scored_results[:k]


def keyword_retrieve_as_context(
    query: str,
    vector_store_path: str,
    k: int = 4,
    filename: Optional[str] = None,
) -> str:
    """
    Same as keyword_retrieve but returns a joined context string,
    matching the interface of retriever.py's retrieve_document().
    """
    results = keyword_retrieve(query, vector_store_path, k=k, filename=filename)

    if not results:
        return ""

    return "\n\n".join([r["content"] for r in results])
