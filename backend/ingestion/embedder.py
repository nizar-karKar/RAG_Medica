import ollama
from ingestion.chunking import document_chunking
from ingestion.loader import load_document
import os

# Function to generate embeddings for text chunks
def generate_embeddings(text_chunks, model_name='nomic-embed-text'):
    embeddings = []

    for chunk in text_chunks:
        # Generate the embedding for each chunk
        embedding = ollama.embeddings(
            model=model_name,
            prompt=chunk
        )

        embeddings.append(embedding)

    return embeddings



# if __name__ == '__main__':
#     BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#     # Path to pdf_documents
#     document_path = os.path.join(BASE_DIR, "pdf_documents")
#     chunks=document_chunking(load_document(document_path))

#     embedded_chunks=generate_embeddings(chunks)
#     print(len(embedded_chunks))