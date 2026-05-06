import os
import sys

# Ensure backend directory is in the path for proper module imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from ingestion.loader import load_document
from ingestion.chunking import document_chunking, add_metadata
from ingestion.embedder import generate_embeddings
from retrieval.vector_store import store_in_chromadb


class IngestionPipeline:
    def __init__(self, document_path=None, doc_title="Document"):
        self.document_path = document_path
        self.doc_title = doc_title

    def load_document(self):
        print(f"Loading documents from: {self.document_path}")
        return load_document(self.document_path)
    
    def chunk_document(self, loaded_documents):
        print("Chunking documents...")
        return document_chunking(loaded_documents)

    def embed_document(self, text_chunks):
        print("Generating embeddings via Ollama (demonstration)...")
        return generate_embeddings(text_chunks)
        
    def store_in_chromadb(self, chunks_with_metadata):
        print("Storing in ChromaDB...")
        # Pass the doc_title as the explicit filename parameter
        store_in_chromadb(chunks_with_metadata, filename=self.doc_title)
    
    def run_pipeline(self):
        print("Starting ingestion pipeline...")
        loaded_documents = self.load_document()
        if not loaded_documents:
            print("No valid documents found to process.")
            return

        chunks = self.chunk_document(loaded_documents)
        chunks_with_metadata = add_metadata(chunks, self.doc_title)

        self.store_in_chromadb(chunks_with_metadata)
        
        print("Pipeline completed successfully!")


ingestion_pipeline = IngestionPipeline()

# if __name__ == "__main__":
#     # Example execution
#     project_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     doc_path = os.path.join(project_backend_dir, "pdf_documents")
    
#     pipeline = IngestionPipeline(document_path=doc_path, doc_title="NVIDIA Financial Report")
#     pipeline.run_pipeline()