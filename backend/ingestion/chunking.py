from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingestion.loader import load_document
from datetime import datetime
import os 

def document_chunking(loaded_document):

    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100
    )

    # Use split_documents to preserve Document objects and their metadata (like 'source')
    chunks=text_splitter.split_documents(loaded_document)

    return chunks



def add_metadata(chunks, doc_title=None):
    metadata_chunks = []

    for chunk in chunks:
        # Extract filename from the original PyPDF metadata, fallback to doc_title if unavailable
        source_path = chunk.metadata.get("source", "")
        file_name = os.path.basename(source_path) if source_path else doc_title
        
        # Extend the existing metadata
        chunk.metadata.update({
            "title": file_name,
            "author": "company",  # Update based on document data
            "date": str(datetime.today())
        })

        metadata_chunks.append(chunk)

    return metadata_chunks


# if __name__ == '__main__':
#     # Get project backend directory
#     BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#     # Path to pdf_documents
#     document_path = os.path.join(BASE_DIR, "pdf_documents")
#     chunks=document_chunking(load_document(document_path))
#     print(len(chunks))
