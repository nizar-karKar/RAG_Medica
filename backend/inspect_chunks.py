"""
Shows exactly what retrieve_document() returns — the raw context string
that gets injected into the LLM prompt.
"""
import sys, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from retrieval.retriever import retrieve_document

VECTOR_STORE_PATH = os.path.join(BASE_DIR, "chroma_db")

# ── Change your query here ──
QUERY = "What is the age of Nizar karkar ?"

print(f"\n🔍 Query: {QUERY}\n")
print("=" * 80)
print("  EXACT STRING RETURNED BY retrieve_document() → fed to the LLM")
print("=" * 80)

context = retrieve_document(QUERY, VECTOR_STORE_PATH)

print(context)

print("=" * 80)
print(f"  {len(context)} characters | ~{len(context)//4} tokens")
print("=" * 80)
