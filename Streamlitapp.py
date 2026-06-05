
import streamlit as st
from openai import AzureOpenAI
from pypdf import PdfReader
import numpy as np
import os
from dotenv import load_dotenv

# --- Azure Config ---
load_dotenv()
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
API_ENDPOINT = os.getenv("API_ENDPOINT")
API_VERSION = os.getenv("API_VERSION")
EMBED_MODEL = os.getenv("EMBED_MODEL")
CHAT_MODEL = os.getenv("CHAT_MODEL")

client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version=API_VERSION,
    azure_endpoint=API_ENDPOINT
)

# --- Helpers ---
def load_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t
    return text


def chunk_text(text, size=300):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]


def create_embeddings(chunks):
    embs = []
    for c in chunks:
        res = client.embeddings.create(model=EMBED_MODEL, input=c)
        embs.append(res.data[0].embedding)
    return np.array(embs)


def search(query, embeddings, chunks):
    q_emb = client.embeddings.create(model=EMBED_MODEL, input=query).data[0].embedding
    q_emb = np.array(q_emb)

    scores = np.dot(embeddings, q_emb)
    top_k = 3
    idxs = np.argsort(scores)[-top_k:]

    return " ".join([chunks[i] for i in idxs])


def ask_llm(query, context):
    prompt = f"""
    Answer strictly using the context.

    Context:
    {context}

    Question:
    {query}
    """

    res = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return res.choices[0].message.content


# --- UI ---
st.title("📄 AI Document Assistant (Chat Mode)")

# Session state init
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Upload PDF
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file and st.session_state.embeddings is None:
    with st.spinner("Processing PDF..."):
        text = load_pdf(uploaded_file)
        chunks = chunk_text(text)
        embeddings = create_embeddings(chunks)

        st.session_state.chunks = chunks
        st.session_state.embeddings = embeddings

    st.success("✅ Document ready!")

# Chat interface
if st.session_state.embeddings is not None:

    query = st.chat_input("Ask something...")

    if query:
        # Exit condition
        if query.lower() == "exit":
            st.session_state.embeddings = None
            st.session_state.chunks = None
            st.session_state.chat_history = []
            st.success("Session reset ✅")
            st.stop()

        # Generate answer
        with st.spinner("Thinking..."):
            context = search(query, st.session_state.embeddings, st.session_state.chunks)
            answer = ask_llm(query, context)

        # Save chat
        st.session_state.chat_history.append(("You", query))
        st.session_state.chat_history.append(("AI", answer))

    # Display chat history
    st.subheader("💬 Conversation")
    for role, msg in st.session_state.chat_history:
        if role == "You":
            st.chat_message("user").write(msg)
        else:
            st.chat_message("assistant").write(msg)
    
    if st.button("Clear Chat"):
        st.session_state.chat_history = []

