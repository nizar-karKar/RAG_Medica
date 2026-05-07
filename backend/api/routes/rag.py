import os
import shutil
import tempfile
import traceback
import uuid
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from langchain_community.document_loaders import PyPDFLoader

from api.dependencies.rag_pipeline import rag_pipeline
from api.schemas.request import QueryRequest
from api.schemas.response import QueryResponse
from ingestion.pipeline import IngestionPipeline

router = APIRouter()

BACKEND_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
PDF_DOCS_DIR = os.path.join(BACKEND_DIR, "pdf_documents")
os.makedirs(PDF_DOCS_DIR, exist_ok=True)


class PdfUploadHandler:
    """Encapsulates PDF validation, ingestion-pipeline triggering, and cleanup."""

    def __init__(self, base_dir: str = PDF_DOCS_DIR):
        self.base_dir = base_dir

    def _allocate_temp_dir(self) -> str:
        unique_dir = f"temp_{uuid.uuid4().hex}"
        path = os.path.join(self.base_dir, unique_dir)
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def _validate_pdf(file_path: str) -> None:
        loader = PyPDFLoader(file_path=file_path)
        pages = loader.load()
        total_text = sum(len(p.page_content.strip()) for p in pages)
        if total_text == 0:
            raise ValueError(
                "PDF contains no extractable text. Scanned or image-only PDFs are "
                "not supported without OCR."
            )

    def handle(self, file: UploadFile) -> dict:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        temp_dir_path = self._allocate_temp_dir()
        file_path = os.path.join(temp_dir_path, file.filename)

        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            print(f"Saved temporary file: {file_path}")

            try:
                self._validate_pdf(file_path)
            except Exception as pdf_e:
                raise HTTPException(status_code=400, detail=str(pdf_e))

            pipeline = IngestionPipeline(
                document_path=temp_dir_path, doc_title=file.filename
            )
            pipeline.run_pipeline()

            return {
                "message": "File processed, stored in ChromaDB, and removed successfully"
            }
        except HTTPException:
            raise
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if os.path.exists(temp_dir_path):
                shutil.rmtree(temp_dir_path, ignore_errors=True)
                print(f"Cleaned up temp dir: {temp_dir_path}")


pdf_upload_handler = PdfUploadHandler()


@router.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    return rag_pipeline.run(request.query, filename=request.filename)


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    return pdf_upload_handler.handle(file)


@router.post("/voice-query", response_model=QueryResponse)
async def voice_query(
    audio: UploadFile = File(...),
    filename: Optional[str] = Form(None),
):
    """
    Accept a browser-recorded audio blob, transcribe it via ElevenLabs STT,
    then run the RAG pipeline and return the AI answer.
    """
    suffix = ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(audio.file, tmp)
        tmp_audio_path = tmp.name

    try:
        result = rag_pipeline.run_voice(
            audio_path=tmp_audio_path, filename=filename or None
        )
        return {"answer": result["answer"], "query": result["query"]}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)
