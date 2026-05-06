import chromadb
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db")

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collections = client.list_collections()
print(f"Collections found in {CHROMA_DB_PATH}:")
for col in collections:
    print(f" - {col.name} (Count: {col.count()})")
