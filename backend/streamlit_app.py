import streamlit as st
import requests
import os
import sys

# Ensure backend directory is in the path for proper module imports
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

st.set_page_config(page_title="RAG Stock Analyst", page_icon="📈", layout="centered")

st.title("📈 NVIDIA Financial RAG Assistant")
st.markdown("Ask any questions regarding NVIDIA's latest financial reports. This interface connects directly to your local Retrieval-Augmented Generation (RAG) system.")

# Choose execution mode
execution_mode = st.sidebar.radio("Execution Mode:", ["Direct Pipeline", "FastAPI Server"])

st.sidebar.markdown("---")
st.sidebar.markdown("""
**How it works**:
- **Direct Pipeline**: Imports `rag_pipeline` and runs it within Streamlit.
- **FastAPI Server**: Sends a POST request to `http://localhost:8000/rag/query`. Ensure you run `uvicorn api.main:app --reload` first.
""")

# Chat history state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
query = st.chat_input("E.g. What was Nvidia's revenue last quarter?")

if query:
    # Display user message in chat message container
    st.chat_message("user").markdown(query)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Analyzing financial reports..."):
            try:
                answer = ""
                
                if execution_mode == "FastAPI Server":
                    # Send request to FastAPI backend
                    response = requests.post(
                        "http://localhost:8000/rag/query",
                        json={"query": query},
                        timeout=120
                    )
                    if response.status_code == 200:
                        answer = response.json().get("answer", "No answer provided.")
                    else:
                        answer = f"**Error from API:** {response.status_code} - {response.text}"
                        
                elif execution_mode == "Direct Pipeline":
                    # Import natively (avoids needing uvicorn running)
                    from api.dependencies.rag_pipeline import rag_pipeline
                    
                    result = rag_pipeline.run(query)
                    answer = result.get("answer", "No answer provided.")

                # Display the response
                message_placeholder.markdown(answer)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = f"**An error occurred:** {str(e)}"
                message_placeholder.error(error_msg)
                if execution_mode == "FastAPI Server":
                    st.info("Make sure the FastAPI server is running: `uvicorn api.main:app --reload`")
