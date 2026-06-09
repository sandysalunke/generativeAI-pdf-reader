'''
RAG - Retrival Augmented Generation
R - Retrieval: Find relevant info from a large corpus (e.g. PDFs)
A - Augmentation: Use retrieved info to enhance LLM's knowledge
G - Generation: LLM generates answer using augmented context

PDF → chunks → embeddings
        ↓
Query → embedding
        ↓
[File Embeddings + Query Embedding] → Similarity Search (Find Top chunks/vectors)
        ↓
[Top chunks + Query] → LLM
        ↓
[Answer]
'''
import streamlit as st
from helpers.embeddings import create_embeddings, search
from helpers.utils import chunk_text, load_pdf, save_data, load_data
from helpers.llm import ask_llm

# --- Persistence ---

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
