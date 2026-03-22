from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingestion.loader import load_document
from datetime import datetime
import os 

def document_chunking(loaded_document):

    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=7500,
        chunk_overlap=100
    )

    nvidia_text_chunks=[]
    for page in loaded_document:
        chunks=text_splitter.split_text(page.page_content)
        nvidia_text_chunks.extend(chunks)

    return nvidia_text_chunks



def add_metadata(chunks, doc_title):
    metadata_chunks = []

    for chunk in chunks:
        metadata = {
            "title": doc_title,
            "author": "company",  # Update based on document data
            "date": str(datetime.today())
        }

        metadata_chunks.append(Document(page_content=chunk, metadata=metadata))

    return metadata_chunks


# if __name__ == '__main__':
#     # Get project backend directory
#     BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#     # Path to pdf_documents
#     document_path = os.path.join(BASE_DIR, "pdf_documents")
#     chunks=document_chunking(load_document(document_path))
#     print(len(chunks))
