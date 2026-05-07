import os
import sys
from typing import List, Optional

from langchain_core.documents import Document

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from ingestion.chunking import DocumentChunker
from ingestion.loader import DocumentLoader
from retrieval.vector_store import VectorStoreManager


class IngestionPipeline:
    """End-to-end ingestion pipeline: load → chunk → tag → store."""

    def __init__(
        self,
        document_path: Optional[str] = None,
        doc_title: str = "Document",
        chunker: Optional[DocumentChunker] = None,
        vector_store: Optional[VectorStoreManager] = None,
    ):
        self.document_path = document_path
        self.doc_title = doc_title
        self.chunker = chunker or DocumentChunker()
        self.vector_store = vector_store or VectorStoreManager()

    def load(self) -> List[Document]:
        print(f"Loading documents from: {self.document_path}")
        return DocumentLoader(self.document_path).load()

    def chunk(self, documents: List[Document]) -> List[Document]:
        print("Chunking documents...")
        return self.chunker.chunk_and_tag(documents, doc_title=self.doc_title)

    def store(self, chunks_with_metadata: List[Document]) -> None:
        print("Storing in ChromaDB...")
        self.vector_store.add_documents(
            chunks_with_metadata, filename=self.doc_title
        )

    def run_pipeline(self) -> None:
        print("Starting ingestion pipeline...")
        documents = self.load()
        if not documents:
            print("No valid documents found to process.")
            return

        chunks_with_metadata = self.chunk(documents)
        self.store(chunks_with_metadata)
        print("Pipeline completed successfully!")
