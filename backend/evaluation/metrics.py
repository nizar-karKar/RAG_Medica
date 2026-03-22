from typing_extensions import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langsmith import Client
import pandas as pd
import os
import sys
from dotenv import load_dotenv
from retrieval.retriever import retrieve_document

# Load environment variables
load_dotenv()
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

## Correctness Output Schema
class CorrectnessGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]

# Relevance Output Schema
class RelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[bool, ..., "Provide the score on whether the answer addresses the question"]

# Groundedness Output Schema
class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    grounded: Annotated[bool, ..., "Provide the score on if the answer hallucinates from the documents"]

# Retrieval Relevance Output Schema
class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[bool, ..., "True if the retrieved documents are relevant to the question, False otherwise"]


## ---------- Define prompts for LLM evaluator-----------

correctness_instructions = """You are a teacher grading a quiz. 

You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. 

Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer. 
(2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the ground truth answer.

Correctness:
A correctness value of True means that the student's answer meets all of the criteria.
A correctness value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""


relevance_instructions = """You are a teacher grading a quiz. 

You will be given a QUESTION and a STUDENT ANSWER. 

Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION

Relevance:
A relevance value of True means that the student's answer meets all of the criteria.
A relevance value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""


grounded_instructions = """You are a teacher grading a quiz. 

You will be given FACTS and a STUDENT ANSWER. 

Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is grounded in the FACTS. 
(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Grounded:
A grounded value of True means that the student's answer meets all of the criteria.
A grounded value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""


retrieval_relevance_instructions = """You are a teacher grading a quiz. 

You will be given a QUESTION and a set of FACTS provided by the student. 

Here is the grade criteria to follow:
(1) Your goal is to identify FACTS that are completely unrelated to the QUESTION
(2) If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant
(3) It is OK if the facts have SOME information that is unrelated to the question as long as (2) is met

Relevance:
A relevance value of True means that the FACTS contain ANY keywords or semantic meaning related to the QUESTION and are therefore relevant.
A relevance value of False means that the FACTS are completely unrelated to the QUESTION.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

#### ---- compute correctness metric ---------

def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    grader_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(CorrectnessGrade, method="json_schema", strict=True)

    """An evaluator for RAG answer accuracy"""
    answers = f"""\
QUESTION: {inputs['question']}
GROUND TRUTH ANSWER: {reference_outputs['answer']}
STUDENT ANSWER: {outputs['answer']}"""

    # Run evaluator
    grade = grader_llm.invoke([
        {"role": "system", "content": correctness_instructions}, 
        {"role": "user", "content": answers}
    ])
    return grade["correct"]



def relevance(inputs: dict, outputs: dict) -> bool:
    relevance_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(RelevanceGrade, method="json_schema", strict=True)
    """A simple evaluator for RAG answer helpfulness."""
    answer = f"QUESTION: {inputs['question']}\nSTUDENT ANSWER: {outputs['answer']}"
    grade = relevance_llm.invoke([
        {"role": "system", "content": relevance_instructions}, 
        {"role": "user", "content": answer}
    ])
    return grade["relevant"]



def groundedness(inputs: dict, outputs: dict) -> bool:
    grounded_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(GroundedGrade, method="json_schema", strict=True)
    """A simple evaluator for RAG answer groundedness."""
    doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
    answer = f"FACTS: {doc_string}\nSTUDENT ANSWER: {outputs['answer']}"
    grade = grounded_llm.invoke([{"role": "system", "content": grounded_instructions}, {"role": "user", "content": answer}])
    return grade["grounded"]






def retrieval_relevance(inputs: dict, outputs: dict) -> bool:
    retrieval_relevance_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(RetrievalRelevanceGrade, method="json_schema", strict=True)
    """An evaluator for document relevance"""
    doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
    answer = f"FACTS: {doc_string}\nQUESTION: {inputs['question']}"

    # Run evaluator
    grade = retrieval_relevance_llm.invoke([
        {"role": "system", "content": retrieval_relevance_instructions}, 
        {"role": "user", "content": answer}
    ])
    return grade["relevant"]

# Target Function definition for evaluating RAG bot
def rag_bot(query: str) -> dict:
    vector_store_path = os.path.join(BASE_DIR, "chroma_db")
    docs = retrieve_document(query, vector_store_path)
    retrieved_document = "\n\n".join(doc.page_content for doc in docs)
    
    RAG_PROMPT = f"""
    You are an assistant specialising in Nvidia financial reports.
    Answer the question using ONLY the context below.
    If the answer is not in the context, say "I don't have enough information."

    Context:
    {retrieved_document}

    Question:
    {query}

    Answer:
    """  
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    response = llm.invoke(RAG_PROMPT)
    
    return {"answer": response.content, "documents": docs}


import threading

evaluation_counter = 0
counter_lock = threading.Lock()

def target(inputs: dict) -> dict:
    global evaluation_counter
    with counter_lock:
        evaluation_counter += 1
        current = evaluation_counter
    print(f"[{current}] Processing query: {inputs.get('question', '')}")
    return rag_bot(inputs["question"])


if __name__ == "__main__":
    client = Client()
    dataset_name = "RAG Evaluation"
    
    print("Starting evaluation...")
    experiment_results = client.evaluate(
        target,
        data=dataset_name,
        evaluators=[correctness, groundedness, relevance, retrieval_relevance],
        experiment_prefix="rag-doc-relevance",
        metadata={"version": "LCEL context, gpt-4o-mini evaluation"},
    )
    
    # Explore results locally as a dataframe
    df = experiment_results.to_pandas()
    
    csv_path = os.path.join(os.path.dirname(__file__), "experiments_results.csv")
    df.to_csv(csv_path, index=False)
    
    print("Evaluation completed successfully!")
    print(f"Dataset saved to: {csv_path}")
    print(df.head())