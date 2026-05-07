import os

from dotenv import load_dotenv
from langsmith import Client

load_dotenv()
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ.setdefault("LANGCHAIN_PROJECT", "RAG_Medica")

print("Tracing:", os.environ["LANGCHAIN_TRACING_V2"])
print("Project:", os.environ["LANGCHAIN_PROJECT"])

client = Client()

DATASET_NAME = "RAG Evaluation"

# Examples grounded in the medical reports indexed in ChromaDB:
#   - medical_report_John_Doe.pdf
#   - medical_report_marie_dupont.pdf
#   - medical_report_Nizar_Karkar.pdf
EXAMPLES = [
    # ---------- John Doe ----------
    {
        "inputs": {"question": "How old is John Doe?"},
        "outputs": {"answer": "45 years old."},
    },
    {
        "inputs": {"question": "What is John Doe's patient ID?"},
        "outputs": {"answer": "123456."},
    },
    {
        "inputs": {"question": "When was John Doe's date of visit?"},
        "outputs": {"answer": "March 20, 2026."},
    },
    {
        "inputs": {"question": "What is John Doe's medical history?"},
        "outputs": {"answer": "Hypertension and mild asthma, with no prior surgeries reported."},
    },
    {
        "inputs": {"question": "What symptoms is John Doe presenting?"},
        "outputs": {"answer": "Fatigue, shortness of breath during exertion, and occasional chest discomfort."},
    },
    {
        "inputs": {"question": "What was John Doe's blood pressure?"},
        "outputs": {"answer": "150/95 mmHg."},
    },
    {
        "inputs": {"question": "What was John Doe's heart rate?"},
        "outputs": {"answer": "88 bpm."},
    },
    {
        "inputs": {"question": "What is John Doe's diagnosis?"},
        "outputs": {"answer": "Stage 1 Hypertension with possible early signs of cardiovascular strain."},
    },
    {
        "inputs": {"question": "What treatment plan was prescribed for John Doe?"},
        "outputs": {
            "answer": "Antihypertensive medication, lifestyle changes including diet and exercise, "
                      "and a follow-up visit in 4 weeks."
        },
    },
    {
        "inputs": {"question": "Who is the attending physician for John Doe?"},
        "outputs": {"answer": "Dr. Sarah Smith."},
    },

    # ---------- Marie Dupont ----------
    {
        "inputs": {"question": "How old is Marie Dupont?"},
        "outputs": {"answer": "45 years old."},
    },
    {
        "inputs": {"question": "What is Marie Dupont's gender?"},
        "outputs": {"answer": "Female."},
    },
    {
        "inputs": {"question": "When was Marie Dupont's medical report dated?"},
        "outputs": {"answer": "24 March 2026."},
    },
    {
        "inputs": {"question": "What is Marie Dupont's diagnosis?"},
        "outputs": {"answer": "Hypertension (high blood pressure)."},
    },
    {
        "inputs": {"question": "What was Marie Dupont's blood pressure reading?"},
        "outputs": {"answer": "150/95 mmHg (normal range is 120/80 mmHg)."},
    },
    {
        "inputs": {"question": "What was Marie Dupont's cholesterol level?"},
        "outputs": {"answer": "220 mg/dL (normal range is below 200 mg/dL)."},
    },
    {
        "inputs": {"question": "What was Marie Dupont's blood sugar level?"},
        "outputs": {"answer": "110 mg/dL (normal range is 70-100 mg/dL)."},
    },
    {
        "inputs": {"question": "What medication was prescribed to Marie Dupont?"},
        "outputs": {"answer": "Amlodipine 5mg once daily."},
    },
    {
        "inputs": {"question": "What lifestyle changes were recommended for Marie Dupont?"},
        "outputs": {"answer": "Reduce salt intake and exercise regularly."},
    },
    {
        "inputs": {"question": "Who is Marie Dupont's doctor?"},
        "outputs": {"answer": "Dr. Smith."},
    },

    # ---------- Nizar Karkar ----------
    
]


def get_or_create_dataset(name: str):
    """Return an existing dataset by name, or create it if it doesn't exist."""
    existing = list(client.list_datasets(dataset_name=name))
    if existing:
        print(f"Dataset '{name}' already exists — reusing it.")
        return existing[0]
    print(f"Creating dataset '{name}'...")
    return client.create_dataset(name)


if __name__ == "__main__":
    dataset = get_or_create_dataset(DATASET_NAME)
    client.create_examples(dataset_id=dataset.id, examples=EXAMPLES)
    print(f"Added {len(EXAMPLES)} examples to '{DATASET_NAME}'.")
