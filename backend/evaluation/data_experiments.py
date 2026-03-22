from langsmith import Client
import os 
from dotenv import load_dotenv
load_dotenv()
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

print("Tracing:", os.environ["LANGCHAIN_TRACING_V2"])
print("Project:", os.environ["LANGCHAIN_PROJECT"])

client = Client()

# Define dataset: these are your test cases
dataset_name = "RAG Evaluation"
dataset = client.create_dataset(dataset_name)
client.create_examples(
    dataset_id=dataset.id,
    examples=[
    {
        "inputs": {"question": "What was NVIDIA's total revenue for Q1 FY2026?"},
        "outputs": {"answer": "$44.1 billion (or $44,062 million)"},
    },
    {
        "inputs": {"question": "By how much did NVIDIA's revenue grow year-over-year in Q1 FY2026?"},
        "outputs": {"answer": "69% year-over-year"},
    },
    {
        "inputs": {"question": "By how much did NVIDIA's revenue grow quarter-over-quarter in Q1 FY2026?"},
        "outputs": {"answer": "12% quarter-over-quarter"},
    },
    {
        "inputs": {"question": "What was NVIDIA's Data Center revenue in Q1 FY2026?"},
        "outputs": {"answer": "$39.1 billion (or $39,100 million)"},
    },
    {
        "inputs": {"question": "How much did NVIDIA's Data Center revenue grow year-over-year in Q1 FY2026?"},
        "outputs": {"answer": "73% year-over-year"},
    },
    {
        "inputs": {"question": "What was NVIDIA's GAAP gross margin for Q1 FY2026?"},
        "outputs": {"answer": "60.5%"},
    },
    {
        "inputs": {"question": "What was NVIDIA's non-GAAP gross margin for Q1 FY2026?"},
        "outputs": {"answer": "61.0%"},
    },
    {
        "inputs": {"question": "What would the non-GAAP gross margin have been excluding the H20 charge?"},
        "outputs": {"answer": "71.3%"},
    },
    {
        "inputs": {"question": "What was the H20 excess inventory and purchase obligation charge NVIDIA incurred in Q1 FY2026?"},
        "outputs": {"answer": "$4.5 billion (or $4,538 million)"},
    },
    {
        "inputs": {"question": "Why did NVIDIA incur the H20 charge in Q1 FY2026?"},
        "outputs": {"answer": "On April 9, 2025, NVIDIA was informed by the U.S. government that a license is required for exports of its H20 products into China, which diminished demand for H20 and resulted in excess inventory and purchase obligations."},
    },
    {
        "inputs": {"question": "How much H20 revenue did NVIDIA generate before the new export licensing requirements?"},
        "outputs": {"answer": "$4.6 billion"},
    },
    {
        "inputs": {"question": "How much additional H20 revenue was NVIDIA unable to ship in Q1 FY2026?"},
        "outputs": {"answer": "$2.5 billion"},
    },
    {
        "inputs": {"question": "What was NVIDIA's GAAP diluted earnings per share in Q1 FY2026?"},
        "outputs": {"answer": "$0.76"},
    },
    {
        "inputs": {"question": "What was NVIDIA's non-GAAP diluted earnings per share in Q1 FY2026?"},
        "outputs": {"answer": "$0.81"},
    },
    {
        "inputs": {"question": "What would NVIDIA's non-GAAP diluted EPS have been excluding the H20 charge and related tax impact?"},
        "outputs": {"answer": "$0.96"},
    },
    {
        "inputs": {"question": "What was NVIDIA's GAAP net income for Q1 FY2026?"},
        "outputs": {"answer": "$18,775 million"},
    },
    {
        "inputs": {"question": "What was NVIDIA's non-GAAP net income for Q1 FY2026?"},
        "outputs": {"answer": "$19,894 million"},
    },
    {
        "inputs": {"question": "What was NVIDIA's GAAP operating income for Q1 FY2026?"},
        "outputs": {"answer": "$21,638 million"},
    },
    {
        "inputs": {"question": "What was NVIDIA's GAAP operating expenses for Q1 FY2026?"},
        "outputs": {"answer": "$5,030 million"},
    },
    {
        "inputs": {"question": "What was NVIDIA's non-GAAP operating expenses for Q1 FY2026?"},
        "outputs": {"answer": "$3,583 million"},
    },
    {
        "inputs": {"question": "What was NVIDIA's Gaming revenue in Q1 FY2026?"},
        "outputs": {"answer": "$3.8 billion, a record high"},
    },
    {
        "inputs": {"question": "How did NVIDIA's Gaming revenue change year-over-year in Q1 FY2026?"},
        "outputs": {"answer": "Up 42% from a year ago"},
    },
    {
        "inputs": {"question": "How did NVIDIA's Gaming revenue change quarter-over-quarter in Q1 FY2026?"},
        "outputs": {"answer": "Up 48% from the previous quarter"},
    },
    {
        "inputs": {"question": "What was NVIDIA's Professional Visualization revenue in Q1 FY2026?"},
        "outputs": {"answer": "$509 million"},
    },
    {
        "inputs": {"question": "What was NVIDIA's Automotive revenue in Q1 FY2026?"},
        "outputs": {"answer": "$567 million"},
    },
    {
        "inputs": {"question": "How did NVIDIA's Automotive revenue change year-over-year in Q1 FY2026?"},
        "outputs": {"answer": "Up 72% from a year ago"},
    },
    {
        "inputs": {"question": "What is NVIDIA's revenue outlook for Q2 FY2026?"},
        "outputs": {"answer": "$45.0 billion, plus or minus 2%"},
    },
    {
        "inputs": {"question": "How much H20 revenue loss is reflected in NVIDIA's Q2 FY2026 outlook?"},
        "outputs": {"answer": "Approximately $8.0 billion due to export control limitations"},
    },
    {
        "inputs": {"question": "What are NVIDIA's expected GAAP and non-GAAP gross margins for Q2 FY2026?"},
        "outputs": {"answer": "GAAP gross margin of 71.8% and non-GAAP gross margin of 72.0%, plus or minus 50 basis points"},
    },
    {
        "inputs": {"question": "What are NVIDIA's expected GAAP and non-GAAP operating expenses for Q2 FY2026?"},
        "outputs": {"answer": "GAAP operating expenses of approximately $5.7 billion and non-GAAP operating expenses of approximately $4.0 billion"},
    },
    {
        "inputs": {"question": "What gross margin range is NVIDIA targeting later in fiscal 2026?"},
        "outputs": {"answer": "Mid-70% range"},
    },
    {
        "inputs": {"question": "What is NVIDIA's expected full year fiscal 2026 operating expense growth rate?"},
        "outputs": {"answer": "Mid-30% range"},
    },
    {
        "inputs": {"question": "What dividend did NVIDIA announce and when will it be paid?"},
        "outputs": {"answer": "$0.01 per share, payable on July 3, 2025, to shareholders of record on June 11, 2025"},
    },
    {
        "inputs": {"question": "What was NVIDIA's free cash flow in Q1 FY2026?"},
        "outputs": {"answer": "$26,135 million"},
    },
    {
        "inputs": {"question": "What was NVIDIA's net cash provided by operating activities in Q1 FY2026?"},
        "outputs": {"answer": "$27,414 million"},
    },
    {
        "inputs": {"question": "What was NVIDIA's cash, cash equivalents and marketable securities as of April 27, 2025?"},
        "outputs": {"answer": "$53,691 million"},
    },
    {
        "inputs": {"question": "How much did NVIDIA spend on stock repurchases in Q1 FY2026?"},
        "outputs": {"answer": "$14,095 million"},
    },
    {
        "inputs": {"question": "What were NVIDIA's total assets as of April 27, 2025?"},
        "outputs": {"answer": "$125,254 million"},
    },
    {
        "inputs": {"question": "What was NVIDIA's total shareholders' equity as of April 27, 2025?"},
        "outputs": {"answer": "$83,843 million"},
    },
    {
        "inputs": {"question": "What was NVIDIA's long-term debt as of April 27, 2025?"},
        "outputs": {"answer": "$8,464 million"},
    },
    {
        "inputs": {"question": "What AI supercomputer did NVIDIA announce is in full-scale production?"},
        "outputs": {"answer": "The Blackwell NVL72 AI supercomputer"},
    },
    {
        "inputs": {"question": "By how much has AI inference token generation surged according to Jensen Huang?"},
        "outputs": {"answer": "Tenfold in just one year"},
    },
    {
        "inputs": {"question": "What stock split did NVIDIA execute and when?"},
        "outputs": {"answer": "A ten-for-one stock split, effective June 7, 2024"},
    },
    {
        "inputs": {"question": "What was NVIDIA's revenue in Q1 FY2025 (a year ago)?"},
        "outputs": {"answer": "$26,044 million"},
    },
    {
        "inputs": {"question": "What was NVIDIA's GAAP gross margin in Q4 FY2025?"},
        "outputs": {"answer": "73.0%"},
    },
    {
        "inputs": {"question": "What was NVIDIA's research and development expense in Q1 FY2026?"},
        "outputs": {"answer": "$3,989 million"},
    },
    {
        "inputs": {"question": "What is the expected GAAP and non-GAAP tax rate for Q2 FY2026?"},
        "outputs": {"answer": "16.5%, plus or minus 1%, excluding any discrete items"},
    },
    {
        "inputs": {"question": "What gaming consoles or platforms were announced to use NVIDIA technology?"},
        "outputs": {"answer": "Nintendo Switch 2, powered by an NVIDIA processor and AI-powered DLSS delivering up to 4K gaming"},
    },
    {
        "inputs": {"question": "How many games is NVIDIA DLSS 4 available in?"},
        "outputs": {"answer": "Over 125 games"},
    },
    {
        "inputs": {"question": "What partnership did NVIDIA announce in Saudi Arabia?"},
        "outputs": {"answer": "A partnership with HUMAIN to build AI factories in the Kingdom of Saudi Arabia to drive the next wave of AI development"},
    }
]
)