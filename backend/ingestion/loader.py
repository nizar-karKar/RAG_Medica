from langchain_community.document_loaders import PyPDFLoader
import os

def load_document(document_folder_path):
    pdf_files = [f for f in os.listdir(document_folder_path) if f.endswith(".pdf")]
    nvidia_pages=[]
    for pdf_file in pdf_files:
        file_path = os.path.join(document_folder_path, pdf_file)
        print(f"Processing file: {file_path}\n")

        loader = PyPDFLoader(file_path=file_path)
        pages = loader.load()
        nvidia_pages.extend(pages)
    return nvidia_pages


