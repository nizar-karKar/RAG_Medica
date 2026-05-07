import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


class DocumentLoader:
    """Loads PDF documents from a folder into LangChain Document objects."""

    def __init__(self, document_folder_path: str):
        self.document_folder_path = document_folder_path

    def load(self) -> List[Document]:
        if not os.path.isdir(self.document_folder_path):
            raise FileNotFoundError(
                f"Document folder does not exist: {self.document_folder_path}"
            )

        pdf_files = [
            f for f in os.listdir(self.document_folder_path) if f.endswith(".pdf")
        ]

        pages: List[Document] = []
        for pdf_file in pdf_files:
            file_path = os.path.join(self.document_folder_path, pdf_file)
            print(f"Processing file: {file_path}")

            loader = PyPDFLoader(file_path=file_path)
            pages.extend(loader.load())

        return pages


def load_document(document_folder_path: str) -> List[Document]:
    """Backward-compatible function wrapper around DocumentLoader."""
    return DocumentLoader(document_folder_path).load()
