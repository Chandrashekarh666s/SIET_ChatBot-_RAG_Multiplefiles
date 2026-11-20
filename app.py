import streamlit as st
from pathlib import Path
from rag_engine import load_base, ask_rag

st.set_page_config(page_title="SIET Chatbot",  layout="centered")

# Sidebar for info and controls
with st.sidebar:
    st.header("About")
    st.write("Shridevi Institute of Engineering and Technology")
    st.caption("RAG Pipeline with Multiple Files")
    st.caption("Created by Chandrashekar H")
    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_hist = []
        st.rerun()

st.title("Shridevi Institute of Engineering and Technology - ChatBot")
st.caption("Ask anything about college information, Courses, Placements, FAQs, Rules and Regulations")

# Safe API key loading
if "GROQ_API_KEY" not in st.secrets:
    st.error("GROQ_API_KEY is missing in Streamlit Secrets.")
    st.stop()

mykey = st.secrets["GROQ_API_KEY"]

# Resolve folder path safely relative to app.py
DOCS_DIR = Path(__file__).parent / "college_docs"

@st.cache_resource(show_spinner="Loading college knowledge base...")
def initiate_rag():
    return load_base(str(DOCS_DIR))

try:
    index, embed_model, chunks = initiate_rag()
except Exception as e:
    st.error(f"Failed to load documents: {e}")
    st.stop()

if "chat_hist" not in st.session_state:
    st.session_state.chat_hist = []

for qa in st.session_state.chat_hist:
    with st.chat_message(qa["role"]):
        st.write(qa["content"])

question = st.chat_input("Ask anything about college....")

if question:
    st.session_state.chat_hist.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents for answer...."):
            try:
                answer = ask_rag(question, embed_model, chunks, index, api_key=mykey)
            except Exception as e:
                answer = f"Sorry, an error occurred while generating the answer: {e}"
            st.write(answer)

    st.session_state.chat_hist.append({"role": "assistant", "content": answer})
