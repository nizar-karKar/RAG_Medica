from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import os
from typing import Any, Optional

def retrieve_document(query: str, vector_store_path: str, k: int = 2, filename: Optional[str] = None):
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    vector_store = Chroma(
        collection_name="medical-local-rag",
        embedding_function=embedding_model,
        persist_directory=vector_store_path
    )

    search_kwargs: dict[str, Any] = {"k": k}
    if filename:
        # Use metadata filtering to restrict search to a specific file
        search_kwargs["filter"] = {"filename": filename}

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )

    retrieved_docs = retriever.invoke(query)

    if not retrieved_docs:
        return ""

    # Join multiple relevant chunks into one context string
    return "\n\n".join([doc.page_content for doc in retrieved_docs])


# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# vector_store_path=os.path.join(BASE_DIR, "chroma_db") 
# query = "what is the age of Marie Dupont?"
# retrieved_document=retrieve_document(query,vector_store_path)
# print(retrieved_document)
# print("\n\n".join(retrieved_document))