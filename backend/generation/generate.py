import os
import sys
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from generation.voice_to_text import VoiceTranscriber
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.multi_query_retriever import MultiQueryRetriever
from retrieval.vector_store import VectorStoreManager

load_dotenv()
if os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


MEDICAL_PROMPT_TEMPLATE = """
You are a Medical assistant specialising in analyzing patient's informations.
Answer the question using ONLY the context below.
Critical : Give only the necessary information for the questions
If the answer is not in the context, say "I don't have enough information."

Context which contains the patient's informations:
{context}

Question:
{query}

Answer:
"""


NVIDIA_PROMPT_TEMPLATE = """
You are an assistant specialising in Nvidia financial reports.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question:
{query}

Answer:
"""


class ResponseGenerator:
    """
    Orchestrates retrieval + LLM generation using a configurable retriever
    and prompt template.
    """

    def __init__(
        self,
        vector_store_path: str,
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        prompt_template: str = MEDICAL_PROMPT_TEMPLATE,
        retriever_strategy: str = "hybrid",
    ):
        self.vector_store = VectorStoreManager(persist_directory=vector_store_path)
        self.llm = ChatOpenAI(model=llm_model, temperature=temperature)
        self.prompt_template = prompt_template
        self.retriever_strategy = retriever_strategy

        self._hybrid = HybridRetriever(vector_store=self.vector_store)
        self._multi_query = None  # lazy

    @property
    def multi_query(self) -> MultiQueryRetriever:
        if self._multi_query is None:
            self._multi_query = MultiQueryRetriever(vector_store=self.vector_store)
        return self._multi_query

    def _build_context(
        self,
        query: str,
        filename: Optional[str] = None,
    ) -> str:
        if self.retriever_strategy == "hybrid":
            return self._hybrid.retrieve_as_context(query, filename=filename)
        if self.retriever_strategy == "multi_query":
            return "\n\n".join(self.multi_query.retrieve(query, filename=filename))
        raise ValueError(f"Unknown retriever strategy: {self.retriever_strategy}")

    def generate(self, query: str, filename: Optional[str] = None) -> str:
        context = self._build_context(query, filename=filename)
        prompt = self.prompt_template.format(context=context, query=query)
        response = self.llm.invoke(prompt)
        return response.content


class VoiceResponseGenerator:
    """Pairs a VoiceTranscriber with a ResponseGenerator."""

    def __init__(
        self,
        response_generator: ResponseGenerator,
        transcriber: Optional[VoiceTranscriber] = None,
    ):
        self.response_generator = response_generator
        self.transcriber = transcriber or VoiceTranscriber()

    def generate(self, audio_path: str, filename: Optional[str] = None) -> dict:
        query = self.transcriber.transcribe(audio_path=audio_path)
        answer = self.response_generator.generate(query, filename=filename)
        return {"query": query, "answer": answer}


def generate_response(query: str, vector_store_path: str, filename: Optional[str] = None) -> str:
    """Backward-compatible wrapper using the hybrid retriever."""
    return ResponseGenerator(vector_store_path=vector_store_path).generate(
        query, filename=filename
    )


def generate_response_from_multi_query_retriever(
    query: str, vector_store_path: str
) -> str:
    """Backward-compatible wrapper using the multi-query retriever (Nvidia prompt)."""
    return ResponseGenerator(
        vector_store_path=vector_store_path,
        prompt_template=NVIDIA_PROMPT_TEMPLATE,
        retriever_strategy="multi_query",
    ).generate(query)


def generate_response_from_voice(
    vector_store_path: str,
    audio_path: Optional[str] = None,
    filename: Optional[str] = None,
) -> dict:
    """Backward-compatible wrapper that transcribes audio then runs RAG."""
    generator = ResponseGenerator(vector_store_path=vector_store_path)
    voice_gen = VoiceResponseGenerator(response_generator=generator)
    return voice_gen.generate(audio_path=audio_path, filename=filename)
