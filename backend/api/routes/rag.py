import os
import shutil
import uuid

from fastapi import APIRouter, File, UploadFile, HTTPException
from api.schemas.request import QueryRequest
from api.schemas.response import QueryResponse
from api.dependencies.rag_pipeline import rag_pipeline
from ingestion.pipeline import IngestionPipeline

router = APIRouter()

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PDF_DOCS_DIR = os.path.join(BACKEND_DIR, "pdf_documents")

if not os.path.exists(PDF_DOCS_DIR):
    os.makedirs(PDF_DOCS_DIR)

@router.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    result = rag_pipeline.run(request.query)
    return result

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Create a unique temporary directory to host this specific file
    unique_dir_name = f"temp_{uuid.uuid4().hex}"
    temp_dir_path = os.path.join(PDF_DOCS_DIR, unique_dir_name)
    os.makedirs(temp_dir_path, exist_ok=True)
    
    # Must prefix with NVIDIA so the existing loader.py picks it up
    filename = file.filename if file.filename.startswith("NVIDIA") else f"NVIDIA_{file.filename}"
    file_path = os.path.join(temp_dir_path, filename)
    
    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        print(f"Saved temporary file: {file_path}")
        
        # Safety check: Ensure the PDF actually has extractable text before running the pipeline 
        # because PyPDFLoader cannot read text from scanned images, which causes ChromaDB to crash on empty lists.
        from langchain_community.document_loaders import PyPDFLoader
        try:
            test_loader = PyPDFLoader(file_path=file_path)
            pages = test_loader.load()
            total_text = sum([len(p.page_content.strip()) for p in pages])
            if total_text == 0:
                raise ValueError("PDF contains no extractable text. Scanned or image-only PDFs are not supported without OCR.")
        except Exception as pdf_e:
            raise HTTPException(status_code=400, detail=str(pdf_e))
            
        # Run ingestion pipeline providing the temporary directory path
        pipeline = IngestionPipeline(document_path=temp_dir_path, doc_title=file.filename)
        pipeline.run_pipeline()
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Ensure cleanup on failure
        if os.path.exists(temp_dir_path):
            shutil.rmtree(temp_dir_path, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))
        
    # Remove the temporary directory and file after processing
    if os.path.exists(temp_dir_path):
        shutil.rmtree(temp_dir_path)
        
    return {"message": "File processed, stored in ChromaDB, and removed successfully"}