import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def test_raw():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    prompt="What are cats?"
    structured_response=llm.with_structured_output(list[str])
    res = structured_response.invoke(prompt)
    print(f"Type: {type(res)}")
    print(f"Content: {res}")

if __name__ == "__main__":
    test_raw()
