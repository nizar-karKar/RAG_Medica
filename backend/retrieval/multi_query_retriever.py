from dotenv import load_dotenv

from langchain_openai import ChatOpenAI


from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import os 

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def retrieve_document(query: str, vector_store_path: str, k: int = 5):
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

def generate_multiple_queries(query:str)->list[str]:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    prompt=f"""You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from
a vector database. By generating multiple perspectives on the user question, your
goal is to help the user overcome some of the limitations of the distance-based
similarity search. Provide ONLY the 5 questions separated by newlines, no numbering,
no extra text.
Original question: {query}"""
    structured_response=llm.with_structured_output(list[str])
    multi_queries=structured_response.invoke(prompt)

    return multi_queries['iterable']

def multi_query_retriever(query:str,vector_store_path:str)->list[str]:
    multi_queries=generate_multiple_queries(query)
    set_documents=set()
    for query in multi_queries:

        retrieved_docs=retrieve_document(query,vector_store_path)
        for doc in retrieved_docs:
            set_documents.add(doc.page_content)

    list_documents=list(set_documents)
    return list_documents



# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# vector_store_path=os.path.join(BASE_DIR, "chroma_db") 
# query="What were Nvidia's revenue and earnings in the latest quarter?"

# print(len(multi_query_retriever(query,vector_store_path)))



