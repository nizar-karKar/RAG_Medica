import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings  # Updated import
from ingestion.loader import load_document
from ingestion.embedder import generate_embeddings
from ingestion.chunking import document_chunking, add_metadata

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db")  # Where DB will be saved

def store_in_chromadb(chunks_with_metadata):
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    nvidia_vector_db = Chroma.from_documents(
        documents=chunks_with_metadata,
        embedding=embedding_model,
        collection_name="nvidia-local-rag",
        persist_directory=CHROMA_DB_PATH  # ← This is the key fix
    )

    print(f"✅ Vector DB created at: {CHROMA_DB_PATH}")
    print(f"✅ Documents stored: {nvidia_vector_db._collection.count()}")
    return nvidia_vector_db


# document_path = os.path.join(BASE_DIR, "pdf_documents")
# chunks = document_chunking(load_document(document_path))
# chunks_with_metadata = add_metadata(chunks, "NVIDIA Financial Report")

# store_in_chromadb(chunks_with_metadata)