from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatOllama
from generation.voice_to_text import transcribe_audio
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from retrieval.retriever import retrieve_document
from retrieval.multi_query_retriever import multi_query_retriever
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def generate_response(query:str, vector_store_path:str, filename: str = None)->str:
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    retrieved_document = retrieve_document(query, vector_store_path, filename=filename)
    RAG_PROMPT =f"""
    You are a Medical  assistant specialising in analyzing patient's informations .
    Answer the question using ONLY the context below.
    Critical : Give only the necessary information for the questions 
    If the answer is not in the context, say "I don't have enough information."

    Context wich contains the patient's informations:
    {retrieved_document}

    Question:
    {query}

    Answer:
    """  
    response=llm.invoke(RAG_PROMPT)

    return response.content

def generate_response_from_multi_query_retriever(query:str,vector_store_path:str)->str:
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    retrieved_document="\n\n".join(multi_query_retriever(query,vector_store_path))
    RAG_PROMPT =f"""
    You are an assistant specialising in Nvidia financial reports.
    Answer the question using ONLY the context below.
    If the answer is not in the context, say "I don't have enough information."

    Context:
    {retrieved_document}

    Question:
    {query}

    Answer:
    """  
    response=llm.invoke(RAG_PROMPT)

    return response.content



def generate_response_from_voice(vector_store_path: str, audio_path: str = None, filename: str = None) -> dict:
    """Transcribe audio then run the RAG pipeline.

    Args:
        vector_store_path: Path to the ChromaDB directory.
        audio_path: Path to the audio file to transcribe. If None, records from mic.
        filename: Optional metadata filter to restrict retrieval to a specific document.

    Returns:
        A dict with keys ``query`` (the transcribed text) and ``answer`` (the RAG response).
    """
    query = transcribe_audio(audio_path=audio_path)
    answer = generate_response(query, vector_store_path, filename=filename)
    return {"query": query, "answer": answer}

    
    

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# vector_store_path=os.path.join(BASE_DIR, "chroma_db") 

# print(generate_response_from_multi_query_retriever("What were Nvidia's revenue and earnings in the latest quarter?",vector_store_path))
# #print(generate_response_from_voice(vector_store_path))


    
