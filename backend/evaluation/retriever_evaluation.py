import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
)

def evaluate_retriever_metrics(
    questions: list[str], 
    contexts: list[list[str]], 
    ground_truths: list[list[str]]
) -> pd.DataFrame:
    
    # Ensure all data lists have the same length
    if not (len(questions) == len(contexts) == len(ground_truths)):
        raise ValueError("Questions, contexts, and ground_truths must be of the same length.")

    # Prepare data dictionary for HuggingFace Dataset
    # Depending on your Ragas version, the key might be 'ground_truths' or 'ground_truth'.
    data = {
        "question": questions,
        "contexts": contexts,
        "ground_truths": ground_truths, # Change to "ground_truth": [gt[0] for gt in ground_truths] if using newer Ragas versions
    }
    
    # Create the dataset
    dataset = Dataset.from_dict(data)
    
    # Define the retriever metrics to evaluate
    metrics = [
        context_precision,
        context_recall,
    ]
    
    # Evaluate the dataset using Ragas
    print("Evaluating retriever metrics using Ragas...")
    evaluation_result = evaluate(
        dataset=dataset,
        metrics=metrics,
    )
    
    # Convert the results to a pandas DataFrame and return
    return evaluation_result.to_pandas()
