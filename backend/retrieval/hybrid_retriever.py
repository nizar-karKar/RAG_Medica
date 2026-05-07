from typing import List, Optional

from retrieval.keyword_retriever import KeywordRetriever
from retrieval.retriever import SemanticRetriever
from retrieval.vector_store import VectorStoreManager


class ReciprocalRankFusion:
    """Merges multiple ranked result lists using weighted Reciprocal Rank Fusion."""

    def __init__(self, k_rrf: int = 60):
        self.k_rrf = k_rrf

    def fuse(
        self,
        ranked_lists: List[List[dict]],
        weights: Optional[List[float]] = None,
    ) -> List[dict]:
        fused_scores: dict[str, float] = {}
        doc_map: dict[str, dict] = {}

        if weights is None:
            weights = [1.0] * len(ranked_lists)

        for list_idx, ranked_list in enumerate(ranked_lists):
            weight = weights[list_idx]
            for rank, result in enumerate(ranked_list, start=1):
                doc_key = result["content"]
                if doc_key not in doc_map:
                    doc_map[doc_key] = result
                fused_scores[doc_key] = fused_scores.get(doc_key, 0.0) + weight * (
                    1.0 / (self.k_rrf + rank)
                )

        fused = []
        for doc_key, score in fused_scores.items():
            entry = dict(doc_map[doc_key])
            entry["score"] = score
            fused.append(entry)

        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused


class HybridRetriever:
    """
    Combines BM25 keyword retrieval and semantic retrieval, fused with
    weighted Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        vector_store: Optional[VectorStoreManager] = None,
        vector_store_path: Optional[str] = None,
        keyword_weight: float = 2.0,
        semantic_weight: float = 1.0,
        k_rrf: int = 60,
    ):
        if vector_store is None:
            if vector_store_path is None:
                raise ValueError(
                    "Either vector_store or vector_store_path must be provided."
                )
            vector_store = VectorStoreManager(persist_directory=vector_store_path)

        self.vector_store = vector_store
        self.keyword_retriever = KeywordRetriever(vector_store=vector_store)
        self.semantic_retriever = SemanticRetriever(vector_store=vector_store)
        self.fusion = ReciprocalRankFusion(k_rrf=k_rrf)
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight

    def retrieve(
        self,
        query: str,
        k: int = 2,
        keyword_k: Optional[int] = None,
        semantic_k: Optional[int] = None,
        filename: Optional[str] = None,
    ) -> List[dict]:
        keyword_k = keyword_k or 2 * k
        semantic_k = semantic_k or 2 * k

        keyword_results = self.keyword_retriever.retrieve(
            query, k=keyword_k, filename=filename
        )
        semantic_results = self.semantic_retriever.retrieve_as_dicts(
            query, k=semantic_k, filename=filename
        )

        fused = self.fusion.fuse(
            [keyword_results, semantic_results],
            weights=[self.keyword_weight, self.semantic_weight],
        )
        return fused[:k]

    def retrieve_as_context(
        self,
        query: str,
        k: int = 4,
        filename: Optional[str] = None,
    ) -> str:
        results = self.retrieve(query, k=k, filename=filename)
        if not results:
            return ""
        return "\n\n".join(r["content"] for r in results)


def hybrid_retrieve(
    query: str,
    vector_store_path: str,
    k: int = 2,
    keyword_k: Optional[int] = None,
    semantic_k: Optional[int] = None,
    filename: Optional[str] = None,
    keyword_weight: float = 2.0,
    semantic_weight: float = 1.0,
) -> List[dict]:
    """Backward-compatible wrapper."""
    retriever = HybridRetriever(
        vector_store_path=vector_store_path,
        keyword_weight=keyword_weight,
        semantic_weight=semantic_weight,
    )
    return retriever.retrieve(
        query,
        k=k,
        keyword_k=keyword_k,
        semantic_k=semantic_k,
        filename=filename,
    )


def hybrid_retrieve_as_context(
    query: str,
    vector_store_path: str,
    k: int = 4,
    filename: Optional[str] = None,
) -> str:
    """Backward-compatible wrapper."""
    return HybridRetriever(vector_store_path=vector_store_path).retrieve_as_context(
        query, k=k, filename=filename
    )
