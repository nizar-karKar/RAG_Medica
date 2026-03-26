import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.multi_query_retriever import generate_multiple_queries

if __name__ == "__main__":
    result = generate_multiple_queries("What were Nvidia's revenue and earnings in the latest quarter?")
    print(type(result))
    print(result)
