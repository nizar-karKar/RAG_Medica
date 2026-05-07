import hashlib
import os
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db")
DEFAULT_COLLECTION_NAME = "medical-local-rag"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"


class VectorStoreManager:
    """Owns the ChromaDB connection and exposes ingestion + retrieval helpers."""

    def __init__(
        self,
        persist_directory: str = DEFAULT_CHROMA_DB_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self._embeddings = OllamaEmbeddings(model=embedding_model)
        self._store: Optional[Chroma] = None

    @property
    def embeddings(self) -> OllamaEmbeddings:
        return self._embeddings

    @property
    def store(self) -> Chroma:
        if self._store is None:
            self._store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self._embeddings,
                persist_directory=self.persist_directory,
            )
        return self._store

    def add_documents(
        self,
        chunks_with_metadata: List[Document],
        filename: Optional[str] = None,
    ) -> Chroma:
        ids = []
        for i, chunk in enumerate(chunks_with_metadata):
            fn = chunk.metadata.get("filename", filename or "unknown")
            upload_date = chunk.metadata.get("upload_date", "no_date")
            unique_id = hashlib.md5(
                f"{fn}_{upload_date}_{i}_{chunk.page_content}".encode()
            ).hexdigest()
            ids.append(unique_id)

        self.store.add_documents(documents=chunks_with_metadata, ids=ids)

        print(
            f"Stored {len(chunks_with_metadata)} new chunks for file: "
            f"{filename or 'unknown'}"
        )
        print(f"Total documents in collection: {self.store._collection.count()}")
        return self.store

    def get_all(self, filename: Optional[str] = None) -> dict:
        if filename:
            return self.store._collection.get(
                where={"filename": filename},
                include=["documents", "metadatas"],
            )
        return self.store._collection.get(include=["documents", "metadatas"])


def store_in_chromadb(chunks_with_metadata, filename: Optional[str] = None):
    """Backward-compatible wrapper."""
    manager = VectorStoreManager()
    return manager.add_documents(chunks_with_metadata, filename=filename)
