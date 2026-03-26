from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import os 
def retrieve_document(query: str, vector_store_path: str, k: int = 1):
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    vector_store = Chroma(
        collection_name="nvidia-local-rag",
        embedding_function=embedding_model,
        persist_directory=vector_store_path
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    retrieved_docs = retriever.invoke(query)

    return retrieved_docs


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

vector_store_path=os.path.join(BASE_DIR, "chroma_db") 
query = "what is the Medical History of John Doe?"
retrieved_document=retrieve_document(query,vector_store_path)
print(retrieved_document)
# print("\n\n".join(retrieved_document))