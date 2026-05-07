
from typing import Optional

from retrieval.retriever import retrieve_document
from retrieval.keyword_retriever import keyword_retrieve


def _reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k_rrf: int = 60,
    weights: Optional[list[float]] = None,
) -> list[dict]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion (RRF).

    Each document's fused score = sum of  weight * (1 / (k_rrf + rank))  across all
    lists where it appears.  k_rrf is a smoothing constant (default 60,
    the standard value from the original RRF paper).

    Returns a single list sorted by fused score (descending).
    """
    fused_scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    if weights is None:
        weights = [1.0] * len(ranked_lists)

    for list_idx, ranked_list in enumerate(ranked_lists):
        weight = weights[list_idx]
        for rank, result in enumerate(ranked_list, start=1):
            # Use content as the deduplication key
            doc_key = result["content"]

            if doc_key not in doc_map:
                doc_map[doc_key] = result

            fused_scores[doc_key] = fused_scores.get(doc_key, 0.0) + weight * (1.0 / (
                k_rrf + rank
            ))

    # Build final list with the fused score
    fused_results = []
    for doc_key, fused_score in fused_scores.items():
        entry = dict(doc_map[doc_key])  # shallow copy
        entry["score"] = fused_score
        fused_results.append(entry)

    fused_results.sort(key=lambda x: x["score"], reverse=True)
    return fused_results


def _semantic_results_from_retriever(
    query: str,
    vector_store_path: str,
    k: int,
    filename: Optional[str] = None,
) -> list[dict]:
    """
    Wrap retrieve_document() output into the same list[dict] format used
    by keyword_retrieve(), so both can be fed into RRF.

    retrieve_document() returns a joined string, so we call the underlying
    Chroma retriever directly to get individual documents with metadata.
    """
    from langchain_chroma import Chroma
    from langchain_ollama import OllamaEmbeddings
    from typing import Any

    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    vector_store = Chroma(
        collection_name="medical-local-rag",
        embedding_function=embedding_model,
        persist_directory=vector_store_path,
    )

    search_kwargs: dict[str, Any] = {"k": k}
    if filename:
        search_kwargs["filter"] = {"filename": filename}

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )

    retrieved_docs = retriever.invoke(query)

    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": 0.0,  # similarity retriever doesn't expose scores
        }
        for doc in retrieved_docs
    ]


# ─── Public API ───────────────────────────────────────────────────────────────


def hybrid_retrieve(
    query: str,
    vector_store_path: str,
    k: int = 2,
    keyword_k: Optional[int] = None,
    semantic_k: Optional[int] = None,
    filename: Optional[str] = None,
    keyword_weight: float = 2.0,  # Higher weight for keyword search by default
    semantic_weight: float = 1.0, # Lower weight for semantic search
) -> list[dict]:
    """
    Hybrid retrieval combining BM25 keyword search and semantic (embedding)
    search, fused via weighted Reciprocal Rank Fusion (RRF).

    Args:
        query:             The user's question.
        vector_store_path: Path to the ChromaDB directory.
        k:                 Number of final results to return.
        keyword_k:         How many candidates to pull from BM25 (default: 2*k).
        semantic_k:        How many candidates to pull from semantic (default: 2*k).
        filename:          Optional metadata filter to restrict to a specific file.
        keyword_weight:    Weight multiplier for keyword search results (default 2.0).
        semantic_weight:   Weight multiplier for semantic search results (default 1.0).

    Returns:
        A list of dicts with keys: 'content', 'metadata', 'score',
        sorted by fused score descending.
    """
    keyword_k = keyword_k or 2 * k
    semantic_k = semantic_k or 2 * k

    # ── Run both retrievers ───────────────────────────────────────────────
    keyword_results = keyword_retrieve(
        query, vector_store_path, k=keyword_k, filename=filename
    )
    semantic_results = _semantic_results_from_retriever(
        query, vector_store_path, k=semantic_k, filename=filename
    )

    # ── Fuse with RRF ─────────────────────────────────────────────────────
    fused = _reciprocal_rank_fusion(
        [keyword_results, semantic_results],
        weights=[keyword_weight, semantic_weight]
    )

    return fused[:k]


def hybrid_retrieve_as_context(
    query: str,
    vector_store_path: str,
    k: int = 4,
    filename: Optional[str] = None,
) -> str:
    """
    Same as hybrid_retrieve but returns a joined context string,
    matching the interface of retriever.py's retrieve_document().
    """
    results = hybrid_retrieve(
        query,
        vector_store_path,
        k=k,
        filename=filename,
    )

    if not results:
        return ""

    return "\n\n".join([r["content"] for r in results])
