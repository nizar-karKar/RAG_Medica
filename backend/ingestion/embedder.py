from typing import List

import ollama


class Embedder:
    """Wraps an Ollama embedding model."""

    def __init__(self, model_name: str = "nomic-embed-text"):
        self.model_name = model_name

    def embed(self, text_chunks: List) -> List:
        embeddings = []
        for chunk in text_chunks:
            embedding = ollama.embeddings(model=self.model_name, prompt=chunk)
            embeddings.append(embedding)
        return embeddings


def generate_embeddings(text_chunks, model_name: str = "nomic-embed-text"):
    """Backward-compatible wrapper."""
    return Embedder(model_name=model_name).embed(text_chunks)
