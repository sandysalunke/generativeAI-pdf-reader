'''
FLOW:
Audio File
   ↓
Speech-to-Text (Whisper)
   ↓
Transcript
   ↓
LLM Processing
   ├── Summary
   ├── Key Points
   ├── Action Items
   ↓
UI Output (Streamlit)

TECH:
Python
OpenAI / Whisper
LangChain (optional)
Streamlit (UI)
'''

from openai import AzureOpenAI
import streamlit as st
import os
from dotenv import load_dotenv
import numpy as np
import pickle

# --- Azure Config ---
load_dotenv()

AZURE_API_KEY = os.getenv("AZURE_API_KEY")
API_ENDPOINT = os.getenv("API_ENDPOINT")
API_VERSION = os.getenv("API_VERSION")
WHISPER_MODEL = os.getenv("WHISPER_MODEL")
EMBED_MODEL = os.getenv("EMBED_MODEL")
CHAT_MODEL = os.getenv("CHAT_MODEL")
DATA_PATH = "data/"

client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version=API_VERSION,
    azure_endpoint=API_ENDPOINT
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chat_history" not in st.session_state:
    st.session_state.transcript = []

if "chat_history" not in st.session_state:
    st.session_state.meeting_insights = []

if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = []

def load_data():
    try:
        embeddings = pickle.load(open(DATA_PATH + "embeddings.pkl", "rb"))
        chunks = pickle.load(open(DATA_PATH + "chunks.pkl", "rb"))
        return embeddings, chunks
    except:
        return None, None

if "embeddings" not in st.session_state:
    emb, ch = load_data()
    st.session_state.embeddings = emb
    st.session_state.chunks = ch
    
def chunk_text(text, size=300):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

def create_embeddings(chunks):
    embs = []
    for c in chunks:
        res = client.embeddings.create(model=EMBED_MODEL, input=c)
        embs.append(res.data[0].embedding)
    return np.array(embs)

def save_data(embeddings, chunks):
    os.makedirs(DATA_PATH, exist_ok=True)
    pickle.dump(embeddings, open(DATA_PATH + "embeddings.pkl", "wb"))
    pickle.dump(chunks, open(DATA_PATH + "chunks.pkl", "wb"))

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

def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            file=audio_file,
            model=WHISPER_MODEL   # or your Azure deployment name
        )
    return transcript.text

def process_meeting(transcript):
    response = client.chat.completions.create(
        model=CHAT_MODEL,  # your deployment name
        messages=[
            {
                "role": "system",
                "content": "You are an AI meeting assistant."
            },
            {
                "role": "user",
                "content": f"""
                Analyze this meeting transcript:

                Provide:
                1. Summary
                2. Key Points
                3. Action Items (with owner if possible)
                4. List of attendees (if mentioned)

                Transcript:
                {transcript}
                """
            }
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

# Display chat
def display_chat():
    st.subheader("Transcript")
    st.write(st.session_state.transcript)
    st.subheader("Meeting Insights")
    st.write(st.session_state.meeting_insights)
    for role, msg in st.session_state.chat_history:
        st.chat_message(role).write(msg)

# Streamlit UI
st.title("AI Meeting Assistant")

uploaded_file = st.file_uploader("Upload Meeting Audio", type=["mp4", "wav"])

new_file_names = uploaded_file if uploaded_file else []
if uploaded_file and new_file_names != st.session_state.uploaded_names:
    st.session_state.uploaded_names = new_file_names
    with open("audio.wav", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Transcribing..."):
        transcript = transcribe_audio("audio.wav")
        st.session_state.transcript = transcript

    st.subheader("Transcript")
    st.write(transcript)
    
    with st.spinner("Analyzing..."):
        insights = process_meeting(transcript)
        st.session_state.meeting_insights = insights

    st.subheader("Meeting Insights")
    st.write(insights)

    chunks = chunk_text(transcript)
    embeddings = create_embeddings(chunks)
    save_data(embeddings, chunks)

    st.session_state.embeddings = embeddings
    st.session_state.chunks = chunks

    st.success("✅ Transcript processed, you can ask questions now!")

# Chat
if st.session_state.embeddings is not None:
    
    query = st.chat_input("Ask a question (type 'exit' to reset)...")

    if query:
        display_chat()
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
    
    st.session_state.embeddings = None
    st.session_state.chunks = None
    st.session_state.chat_history = []
    st.success("All data cleared!")
    st.rerun()
