import hashlib
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import os
from typing import Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db")

def store_in_chromadb(chunks_with_metadata, filename: Optional[str] = None):
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    vector_db = Chroma(
        collection_name="medical-local-rag",
        embedding_function=embedding_model,
        persist_directory=CHROMA_DB_PATH
    )

    ids = []
    for i, chunk in enumerate(chunks_with_metadata):
        # ✅ Use the enriched metadata, with the parameter as a fallback
        fn = chunk.metadata.get("filename", filename or "unknown")
        upload_date = chunk.metadata.get("upload_date", "no_date")
        
        # ✅ Generate a unique ID using filename, upload date, chunk index and content
        unique_id = hashlib.md5(
            f"{fn}_{upload_date}_{i}_{chunk.page_content}".encode()
        ).hexdigest()
        ids.append(unique_id)

    vector_db.add_documents(documents=chunks_with_metadata, ids=ids)

    print(f"✅ Stored {len(chunks_with_metadata)} new chunks for file: {filename or 'unknown'}")
    print(f"✅ Total documents in collection: {vector_db._collection.count()}")

    return vector_db

# document_path = os.path.join(BASE_DIR, "pdf_documents")
# chunks = document_chunking(load_document(document_path))
# chunks_with_metadata = add_metadata(chunks, "NVIDIA Financial Report")

# store_in_chromadb(chunks_with_metadata)