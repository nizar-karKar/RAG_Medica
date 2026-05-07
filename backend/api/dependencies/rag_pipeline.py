import os
import sys
from typing import Optional

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BACKEND_DIR)

from generation.generate import ResponseGenerator, VoiceResponseGenerator
from generation.voice_to_text import VoiceTranscriber


class RagPipeline:
    """Application-level orchestrator wired into the FastAPI dependencies."""

    def __init__(
        self,
        vector_store_path: Optional[str] = None,
        response_generator: Optional[ResponseGenerator] = None,
        voice_generator: Optional[VoiceResponseGenerator] = None,
    ):
        self.vector_store_path = vector_store_path or os.path.join(
            BACKEND_DIR, "chroma_db"
        )
        self.response_generator = response_generator or ResponseGenerator(
            vector_store_path=self.vector_store_path
        )
        self.voice_generator = voice_generator or VoiceResponseGenerator(
            response_generator=self.response_generator,
            transcriber=VoiceTranscriber(),
        )

    def run(self, query: str, filename: Optional[str] = None) -> dict:
        answer = self.response_generator.generate(query, filename=filename)
        return {"answer": answer}

    def run_voice(self, audio_path: str, filename: Optional[str] = None) -> dict:
        return self.voice_generator.generate(audio_path=audio_path, filename=filename)


rag_pipeline = RagPipeline()
