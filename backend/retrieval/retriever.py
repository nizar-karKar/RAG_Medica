from typing import Any, List, Optional

from langchain_core.documents import Document

from retrieval.vector_store import VectorStoreManager


class SemanticRetriever:
    """Embedding-based similarity retrieval over a Chroma vector store."""

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

    def retrieve_documents(
        self,
        query: str,
        k: int = 2,
        filename: Optional[str] = None,
    ) -> List[Document]:
        search_kwargs: dict[str, Any] = {"k": k}
        if filename:
            search_kwargs["filter"] = {"filename": filename}

        retriever = self.vector_store.store.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs,
        )
        return retriever.invoke(query)

    def retrieve_as_dicts(
        self,
        query: str,
        k: int = 2,
        filename: Optional[str] = None,
    ) -> List[dict]:
        docs = self.retrieve_documents(query, k=k, filename=filename)
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": 0.0,
            }
            for doc in docs
        ]

    def retrieve_as_context(
        self,
        query: str,
        k: int = 2,
        filename: Optional[str] = None,
    ) -> str:
        docs = self.retrieve_documents(query, k=k, filename=filename)
        if not docs:
            return ""
        return "\n\n".join(doc.page_content for doc in docs)


def retrieve_document(
    query: str,
    vector_store_path: str,
    k: int = 2,
    filename: Optional[str] = None,
):
    """Backward-compatible wrapper."""
    return SemanticRetriever(vector_store_path=vector_store_path).retrieve_as_context(
        query, k=k, filename=filename
    )
