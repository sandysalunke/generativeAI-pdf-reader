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

import streamlit as st
import os
from config import DATA_PATH
from helpers.transcription import transcribe_audio
from helpers.embeddings import create_embeddings, search
from helpers.utils import load_data, save_data, chunk_text
from helpers.llm import ask_llm, process_meeting

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "transcript" not in st.session_state:
    st.session_state.transcript = []

if "meeting_insights" not in st.session_state:
    st.session_state.meeting_insights = []

if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = []

if "embeddings" not in st.session_state:
    emb, ch = load_data()
    st.session_state.embeddings = emb
    st.session_state.chunks = ch

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

    with st.spinner("Getting you ready to ask questions..."):
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
