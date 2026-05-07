"""
Debug: Why does the retriever favor John Doe over Nizar Karkar?
Shows all documents in ChromaDB with their similarity scores for the query.
"""
import sys, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

VECTOR_STORE_PATH = os.path.join(BASE_DIR, "chroma_db")
QUERY = "What is the age of Nizar karkar ?"

# ── Step 1: See how many total documents exist ────────────────────────────
print("=" * 80)
print("  STEP 1 — ALL DOCUMENTS IN THE COLLECTION")
print("=" * 80)

embedding_model = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma(
    collection_name="medical-local-rag",
    embedding_function=embedding_model,
    persist_directory=VECTOR_STORE_PATH,
)

# Get the raw ChromaDB collection
collection = vector_store._collection
total = collection.count()
print(f"\n  Total documents in collection: {total}\n")

# Peek at ALL documents with metadata
all_docs = collection.get(include=["documents", "metadatas"])

for i, (doc, meta) in enumerate(zip(all_docs["documents"], all_docs["metadatas"])):
    filename = meta.get("filename", "unknown")
    preview = doc[:80].replace("\n", " ") + "..."
    print(f"  [{i}] filename={filename}  |  preview: {preview}")

# ── Step 2: Similarity search WITH scores ─────────────────────────────────
print("\n" + "=" * 80)
print(f"  STEP 2 — SIMILARITY SCORES FOR QUERY: \"{QUERY}\"")
print("=" * 80)

# similarity_search_with_score returns (Document, score) tuples
# In Chroma, LOWER score = MORE similar (L2 distance)
results_with_scores = vector_store.similarity_search_with_score(QUERY, k=total)

print(f"\n  Ranking all {len(results_with_scores)} chunks by similarity (lower = better):\n")

for rank, (doc, score) in enumerate(results_with_scores, start=1):
    filename = doc.metadata.get("filename", "unknown")
    # Show first 100 chars of content
    preview = doc.page_content[:100].replace("\n", " ")
    has_nizar = "Nizar" in doc.page_content

    marker = " ✅ CONTAINS 'Nizar'" if has_nizar else ""
    selected = " ◀ SELECTED" if rank <= 2 else ""  # k=2 in your retriever

    print(f"  #{rank}  score={score:.4f}  file={filename}")
    print(f"       preview: {preview}...")
    print(f"       {marker}{selected}")
    print()

# ── Step 3: Explain the root cause ────────────────────────────────────────
print("=" * 80)
print("  STEP 3 — ROOT CAUSE ANALYSIS")
print("=" * 80)

# Count duplicates
from collections import Counter
content_hashes = Counter()
for doc in all_docs["documents"]:
    content_hashes[doc] += 1

duplicates = {k: v for k, v in content_hashes.items() if v > 1}
print(f"\n  Duplicate chunks found: {len(duplicates)}")
for content, count in duplicates.items():
    preview = content[:80].replace("\n", " ")
    print(f"    → {count}x copies: \"{preview}...\"")

print()
