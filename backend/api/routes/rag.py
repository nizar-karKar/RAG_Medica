import os
import shutil
import uuid
import tempfile

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from typing import Optional
from api.schemas.request import QueryRequest
from api.schemas.response import QueryResponse
from api.dependencies.rag_pipeline import rag_pipeline
from ingestion.pipeline import IngestionPipeline
from generation.generate import generate_response_from_voice

router = APIRouter()

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PDF_DOCS_DIR = os.path.join(BACKEND_DIR, "pdf_documents")

if not os.path.exists(PDF_DOCS_DIR):
    os.makedirs(PDF_DOCS_DIR)

@router.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    result = rag_pipeline.run(request.query, filename=request.filename)
    return result


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    unique_dir_name = f"temp_{uuid.uuid4().hex}"
    temp_dir_path = os.path.join(PDF_DOCS_DIR, unique_dir_name)
    os.makedirs(temp_dir_path, exist_ok=True)

    filename = file.filename
    file_path = os.path.join(temp_dir_path, filename)

    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"Saved temporary file: {file_path}")

        # Safety check: Ensure the PDF actually has extractable text
        from langchain_community.document_loaders import PyPDFLoader
        try:
            test_loader = PyPDFLoader(file_path=file_path)
            pages = test_loader.load()
            total_text = sum([len(p.page_content.strip()) for p in pages])
            if total_text == 0:
                raise ValueError("PDF contains no extractable text. Scanned or image-only PDFs are not supported without OCR.")
        except Exception as pdf_e:
            raise HTTPException(status_code=400, detail=str(pdf_e))

        # ✅ Run pipeline while file still exists on disk
        pipeline = IngestionPipeline(document_path=temp_dir_path, doc_title=file.filename)
        pipeline.run_pipeline()

        # ✅ Return BEFORE finally block triggers cleanup
        return {"message": "File processed, stored in ChromaDB, and removed successfully"}

    except HTTPException:
        # Re-raise HTTP exceptions directly without wrapping
        raise

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # ✅ Always runs after return or exception — guarantees cleanup
        # and ensures file is never deleted before ChromaDB finishes committing
        if os.path.exists(temp_dir_path):
            shutil.rmtree(temp_dir_path, ignore_errors=True)
            print(f"🧹 Cleaned up temp dir: {temp_dir_path}")


@router.post("/voice-query", response_model=QueryResponse)
async def voice_query(
    audio: UploadFile = File(...),
    filename: Optional[str] = Form(None),
):
    """
    Accept a browser-recorded audio blob, transcribe it via ElevenLabs STT,
    then run the RAG pipeline and return the AI answer.
    The optional `filename` form-field filters results to a specific uploaded document.
    """
    CHROMA_DB_PATH = os.path.join(BACKEND_DIR, "chroma_db")

    # Save the incoming audio blob to a temp file
    suffix = ".webm"  # browsers produce WebM/Opus blobs via MediaRecorder
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(audio.file, tmp)
        tmp_audio_path = tmp.name

    try:
        answer = generate_response_from_voice(
            vector_store_path=CHROMA_DB_PATH,
            audio_path=tmp_audio_path,
            filename=filename or None,
        )
        return {"answer": answer}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)