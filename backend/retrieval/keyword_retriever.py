import re
from typing import List, Optional

from rank_bm25 import BM25Okapi

from retrieval.vector_store import VectorStoreManager


def tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer, lowercased."""
    return re.findall(r"\w+", text.lower())


class KeywordRetriever:
    """BM25 keyword retrieval over documents stored in ChromaDB."""

    def __init__(
        self,
        vector_store: Optional[VectorStoreManager] = None,
        vector_store_path: Optional[str] = None,
    ):
        if vector_store is None:
            if vector_store_path is None:
                raise ValueError(
                    "Either vector_store or vector_store_path must be provided."
                )
            vector_store = VectorStoreManager(persist_directory=vector_store_path)
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        k: int = 1,
        filename: Optional[str] = None,
    ) -> List[dict]:
        raw = self.vector_store.get_all(filename=filename)
        documents = raw["documents"]
        metadatas = raw["metadatas"]

        if not documents:
            return []

        tokenized_docs = [tokenize(doc) for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)
        scores = bm25.get_scores(tokenize(query))

        scored = [
            {
                "content": documents[i],
                "metadata": metadatas[i],
                "score": float(score),
            }
            for i, score in enumerate(scores)
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]

    def retrieve_as_context(
        self,
        query: str,
        k: int = 4,
        filename: Optional[str] = None,
    ) -> str:
        results = self.retrieve(query, k=k, filename=filename)
        if not results:
            return ""
        return "\n\n".join(r["content"] for r in results)


def keyword_retrieve(
    query: str,
    vector_store_path: str,
    k: int = 1,
    filename: Optional[str] = None,
) -> List[dict]:
    """Backward-compatible wrapper."""
    return KeywordRetriever(vector_store_path=vector_store_path).retrieve(
        query, k=k, filename=filename
    )


def keyword_retrieve_as_context(
    query: str,
    vector_store_path: str,
    k: int = 4,
    filename: Optional[str] = None,
) -> str:
    """Backward-compatible wrapper."""
    return KeywordRetriever(vector_store_path=vector_store_path).retrieve_as_context(
        query, k=k, filename=filename
    )
