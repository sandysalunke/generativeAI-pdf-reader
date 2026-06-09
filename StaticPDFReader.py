import streamlit as st
from config import DATA_PATH
from helpers.embeddings import create_embeddings, search
from helpers.utils import chunk_text, load_pdf
from helpers.llm import ask_llm

# --- MAIN ---
if __name__ == "__main__":
    file_path = DATA_PATH + "Sandip Salunke-Resume.pdf"  # 🔹 replace

    print("📄 Loading PDF...")
    text = load_pdf(file_path)

    print("✂️ Chunking text...")
    chunks = chunk_text(text)

    print("🧠 Creating embeddings...")
    embeddings = create_embeddings(chunks)

    print("✅ Ready! Ask questions.\n")

    while True:
        query = input("Ask (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        context = search(query, embeddings, chunks)
        answer = ask_llm(query, context)

        print("\n--- Answer ---")
        print(answer)
