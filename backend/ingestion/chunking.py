import os
from datetime import datetime
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    """Splits documents into chunks and enriches them with metadata."""

    def __init__(self, chunk_size: int = 900, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split(self, documents: List[Document]) -> List[Document]:
        return self._splitter.split_documents(documents)

    def add_metadata(
        self,
        chunks: List[Document],
        doc_title: Optional[str] = None,
    ) -> List[Document]:
        upload_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        enriched: List[Document] = []

        for i, chunk in enumerate(chunks):
            source_path = chunk.metadata.get("source", "")
            file_name = (
                os.path.basename(source_path)
                if source_path
                else (doc_title or "unknown_document")
            )

            chunk.metadata.update(
                {
                    "filename": file_name,
                    "title": file_name,
                    "author": "RAG System",
                    "upload_date": upload_timestamp,
                    "chunk_index": i,
                }
            )
            enriched.append(chunk)

        return enriched

    def chunk_and_tag(
        self,
        documents: List[Document],
        doc_title: Optional[str] = None,
    ) -> List[Document]:
        chunks = self.split(documents)
        return self.add_metadata(chunks, doc_title=doc_title)


def document_chunking(loaded_document):
    """Backward-compatible wrapper."""
    return DocumentChunker().split(loaded_document)


def add_metadata(chunks, doc_title=None):
    """Backward-compatible wrapper."""
    return DocumentChunker().add_metadata(chunks, doc_title=doc_title)
