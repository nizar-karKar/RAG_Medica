import sys
import os
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BACKEND_DIR)

from ingestion.pipeline import IngestionPipeline
from generation.generate import generate_response_from_multi_query_retriever

class RagPipeline():
    def __init__(self):
        self.vector_store_path = os.path.join(BACKEND_DIR, "chroma_db")

    def run(self, query: str):
        response = generate_response_from_multi_query_retriever(query, self.vector_store_path)
        return {
            "answer": response
        }

rag_pipeline = RagPipeline()
# # Instantiate the pipeline with default paths
# rag_pipeline = RagPipeline()

# if __name__ == "__main__":
#     # Example execution to show it functions correctly
#     test_query = "What were Nvidia's revenue and earnings in the latest quarter?"
#     print(f"Executing RAG pipeline for query: '{test_query}'...\n")
#     try:
#         result = rag_pipeline.run_rag(test_query)
#         print("✅ Pipeline Response:")
#         print(result["answer"])
#     except Exception as e:
#         print(f"❌ Error occurred: {e}")
