import streamlit as st
from helpers.embeddings import create_embeddings, search
from helpers.utils import chunk_text, load_pdf
from helpers.llm import ask_llm

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

