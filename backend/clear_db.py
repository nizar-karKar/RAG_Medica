import shutil
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db")

def clear_chroma_db():
    if os.path.exists(CHROMA_DB_PATH):
        print(f"🗑️  Deleting ChromaDB folder at: {CHROMA_DB_PATH}")
        try:
            shutil.rmtree(CHROMA_DB_PATH)
            print("✅ ChromaDB successfully cleared!")
        except Exception as e:
            print(f"❌ Failed to clear ChromaDB: {e}")
    else:
        print(f"⚠️  ChromaDB folder '{CHROMA_DB_PATH}' does not exist at '{CHROMA_DB_PATH}'. Nothing to clear.")

if __name__ == "__main__":
    clear_chroma_db()
