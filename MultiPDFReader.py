import streamlit as st
from openai import AzureOpenAI
from pypdf import PdfReader
import numpy as np
import os
import pickle
from dotenv import load_dotenv

load_dotenv()  # ✅ loads .env file

# --- Azure Config ---
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

DATA_PATH = "data/"


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
    Answer only using the context.

    Context:
    {context}

    Question:
    {query}
    """

    res = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are an enterprise assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return res.choices[0].message.content


# --- Persistence ---
def save_data(embeddings, chunks):
    os.makedirs(DATA_PATH, exist_ok=True)
    pickle.dump(embeddings, open(DATA_PATH + "embeddings.pkl", "wb"))
    pickle.dump(chunks, open(DATA_PATH + "chunks.pkl", "wb"))

def load_data():
    try:
        embeddings = pickle.load(open(DATA_PATH + "embeddings.pkl", "rb"))
        chunks = pickle.load(open(DATA_PATH + "chunks.pkl", "rb"))
        return embeddings, chunks
    except:
        return None, None

# Display chat
def display_chat():
    for role, msg in st.session_state.chat_history:
        st.chat_message(role).write(msg)

# --- UI ---
st.title("📄 Multi-PDF AI Assistant (Persistent)")

if "embeddings" not in st.session_state:
    emb, ch = load_data()
    st.session_state.embeddings = emb
    st.session_state.chunks = ch
    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = []

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# Upload multiple PDFs
uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_key}"

)

# Process PDFs
new_file_names = [f.name for f in uploaded_files] if uploaded_files else []

if uploaded_files and new_file_names != st.session_state.uploaded_names:
    st.session_state.uploaded_names = new_file_names
    with st.spinner("Processing PDFs..."):
        all_chunks = []
        
        for file in uploaded_files:
            text = load_pdf(file)
            chunks = chunk_text(text)
            all_chunks.extend(chunks)
                
        embeddings = create_embeddings(all_chunks)

        save_data(embeddings, all_chunks)

        st.session_state.embeddings = embeddings
        st.session_state.chunks = all_chunks

    st.success("✅ PDFs processed & saved!")

# Chat
if st.session_state.embeddings is not None:
    display_chat()
    query = st.chat_input("Ask a question (type 'exit' to reset)...")

    if query:
        st.session_state.chat_history.append(("user", query))
        st.chat_message("user").write(query)
        # ✅ Placeholder for spinner (important)
        spinner_placeholder = st.empty()

        if query.lower() == "exit":
            st.session_state.embeddings = None
            st.session_state.chunks = None
            st.session_state.chat_history = []
            st.success("Session reset ✅")
            st.stop()

        # Generate answer
        # ✅ Show spinner ABOVE input using placeholder
        with spinner_placeholder:
            with st.spinner("Thinking..."):
                context = search(query, st.session_state.embeddings, st.session_state.chunks)
                answer = ask_llm(query, context)

        
        # Clear spinner (optional)
        spinner_placeholder.empty()

        st.session_state.chat_history.append(("assistant", answer))

        # Display response immediately
        st.chat_message("assistant").write(answer)

# Clear button
if st.button("Clear All Data"):
    try:
        os.remove(DATA_PATH + "embeddings.pkl")
        os.remove(DATA_PATH + "chunks.pkl")
    except:
        pass

    # ✅ Reset uploader by changing key
    st.session_state.uploader_key += 1
    
    st.session_state.embeddings = None
    st.session_state.chunks = None
    st.session_state.chat_history = []
    st.success("All data cleared!")
    st.rerun()
