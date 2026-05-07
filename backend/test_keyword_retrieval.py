
import sys, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from retrieval.retriever import retrieve_document
from retrieval.keyword_retriever import keyword_retrieve

VECTOR_STORE_PATH = os.path.join(BASE_DIR, "chroma_db")
QUERY = "What is the age of Nizar Karkar ?"

# print(f"\n🔍 Query: {QUERY}\n")

# # ── Semantic Retrieval (current approach) ─────────────────────────────────
# print("=" * 80)
# print("  SEMANTIC RETRIEVAL (embedding similarity)")
# print("=" * 80)

# semantic_result = retrieve_document(QUERY, VECTOR_STORE_PATH, k=2)
# print(semantic_result if semantic_result else "  (no results)")

# # ── Keyword Retrieval (BM25) ─────────────────────────────────────────────
# print("\n" + "=" * 80)
# print("  KEYWORD RETRIEVAL (BM25)")
# print("=" * 80)

keyword_results = keyword_retrieve(QUERY, VECTOR_STORE_PATH, k=1)

for i, result in enumerate(keyword_results, 1):
    filename = result["metadata"].get("filename", "unknown")
    score = result["score"]
    preview = result["content"][:200].replace("\n", " ")
    print(f"\n  #{i}  BM25 score={score:.4f}  file={filename}")
    print(f"      {preview}...")

# ── Show the context that BM25 would feed to the LLM ─────────────────────
print("\n\n" + "=" * 80)
print("  CONTEXT THAT BM25 WOULD FEED TO THE LLM")
print("=" * 80)

from retrieval.keyword_retriever import keyword_retrieve_as_context
bm25_context = keyword_retrieve_as_context(QUERY, VECTOR_STORE_PATH, k=1)
print(bm25_context)
