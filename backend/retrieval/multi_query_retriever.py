import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from retrieval.retriever import SemanticRetriever
from retrieval.vector_store import VectorStoreManager

load_dotenv()
if os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


class MultiQueryRetriever:
    """
    Generates multiple paraphrases of the user query (via an LLM) and unions the
    documents retrieved for each paraphrase.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStoreManager] = None,
        vector_store_path: Optional[str] = None,
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        num_queries: int = 5,
        k_per_query: int = 5,
    ):
        if vector_store is None:
            if vector_store_path is None:
                raise ValueError(
                    "Either vector_store or vector_store_path must be provided."
                )
            vector_store = VectorStoreManager(persist_directory=vector_store_path)

        self.vector_store = vector_store
        self.semantic_retriever = SemanticRetriever(vector_store=vector_store)
        self.llm = ChatOpenAI(model=llm_model, temperature=temperature)
        self.num_queries = num_queries
        self.k_per_query = k_per_query

    def generate_queries(self, query: str) -> List[str]:
        prompt = (
            f"You are an AI language model assistant. Your task is to generate "
            f"{self.num_queries} different versions of the given user question to "
            "retrieve relevant documents from a vector database. By generating "
            "multiple perspectives on the user question, your goal is to help the "
            "user overcome some of the limitations of the distance-based similarity "
            "search. Provide ONLY the questions separated by newlines, no numbering, "
            f"no extra text.\nOriginal question: {query}"
        )
        structured_response = self.llm.with_structured_output(list[str])
        result = structured_response.invoke(prompt)

        if isinstance(result, dict):
            return list(result.values())[0] if result else []
        if isinstance(result, list):
            return result
        return [str(result)]

    def retrieve(self, query: str, filename: Optional[str] = None) -> List[str]:
        queries = self.generate_queries(query)
        seen: set[str] = set()
        for q in queries:
            docs = self.semantic_retriever.retrieve_documents(
                q, k=self.k_per_query, filename=filename
            )
            for doc in docs:
                seen.add(doc.page_content)
        return list(seen)


def multi_query_retriever(query: str, vector_store_path: str) -> List[str]:
    """Backward-compatible wrapper."""
    return MultiQueryRetriever(vector_store_path=vector_store_path).retrieve(query)
