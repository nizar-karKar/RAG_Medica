import hashlib
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db")

def store_in_chromadb(chunks_with_metadata):
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    vector_db = Chroma(
        collection_name="nvidia-local-rag",
        embedding_function=embedding_model,
        persist_directory=CHROMA_DB_PATH
    )

    ids = []
    for i, chunk in enumerate(chunks_with_metadata):
        # ✅ Fix 1: Use 'title' (stable filename set in add_metadata)
        # instead of 'source' (unstable UUID temp path that gets deleted)
        title = chunk.metadata.get("title", "unknown")

        # ✅ Fix 2: Use full page_content instead of [:50]
        # to avoid collisions between chunks with similar openings
        unique_id = hashlib.md5(
            f"{title}_{i}_{chunk.page_content}".encode()
        ).hexdigest()
        ids.append(unique_id)

    vector_db.add_documents(documents=chunks_with_metadata, ids=ids)

    print(f"✅ Stored {len(chunks_with_metadata)} new chunks")
    print(f"✅ Total chunks in DB: {vector_db._collection.count()}")

    return vector_db

# document_path = os.path.join(BASE_DIR, "pdf_documents")
# chunks = document_chunking(load_document(document_path))
# chunks_with_metadata = add_metadata(chunks, "NVIDIA Financial Report")

# store_in_chromadb(chunks_with_metadata)