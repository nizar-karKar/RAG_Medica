import sys, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from retrieval.hybrid_retriever import hybrid_retrieve, hybrid_retrieve_as_context

VECTOR_STORE_PATH = os.path.join(BASE_DIR, "chroma_db")
QUERY = "What is the age of Nizar Karkar ?"

print(f"\n🔍 Query: {QUERY}\n")

# ── Hybrid Retrieval (BM25 + Semantic) ─────────────────────────────────────────────
print("\n" + "=" * 80)
print("  HYBRID RETRIEVAL (BM25 + Semantic with RRF)")
print("=" * 80)

if __name__ == "__main__":
    print("Starting hybrid_retrieve...")
    hybrid_results = hybrid_retrieve(QUERY, VECTOR_STORE_PATH, k=3)
    print("Finished hybrid_retrieve...")

    for i, result in enumerate(hybrid_results, 1):
        filename = result["metadata"].get("filename", "unknown")
        score = result["score"]
        preview = result["content"][:200].replace("\n", " ")
        print(f"\n  #{i}  RRF score={score:.4f}  file={filename}")
        print(f"      {preview}...")

    # ── Show the context that Hybrid Retrieval would feed to the LLM ─────────────────────
    print("\n\n" + "=" * 80)
    print("  CONTEXT THAT HYBRID RETRIEVER WOULD FEED TO THE LLM")
    print("=" * 80)

    print("Starting hybrid_retrieve_as_context...")
    hybrid_context = hybrid_retrieve_as_context(QUERY, VECTOR_STORE_PATH, k=2)
    print("Finished hybrid_retrieve_as_context...")
    print(hybrid_context)
