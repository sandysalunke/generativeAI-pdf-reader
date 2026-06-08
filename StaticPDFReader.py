
from openai import AzureOpenAI
from pypdf import PdfReader
import numpy as np
import os
from dotenv import load_dotenv
from config import DATA_PATH, EMBED_MODEL, CHAT_MODEL

load_dotenv()

# --- Azure Config ---
api_key = os.getenv("AZURE_API_KEY")
api_endpoint = os.getenv("API_ENDPOINT")

client = AzureOpenAI(
    api_key=api_key,
    api_version="2024-02-15-preview",
    azure_endpoint=api_endpoint
)

# --- STEP 1: Load PDF ---
def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    return text


# --- STEP 2: Chunk Text ---
def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


# --- STEP 3: Create Embeddings ---
def create_embeddings(chunks):
    embeddings = []

    for chunk in chunks:
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=chunk
        )
        embeddings.append(response.data[0].embedding)

    return np.array(embeddings)


# --- STEP 4: Find Relevant Context ---
def search(query, embeddings, chunks):
    query_embedding = client.embeddings.create(
        model=EMBED_MODEL,
        input=query
    ).data[0].embedding

    query_embedding = np.array(query_embedding)

    similarities = np.dot(embeddings, query_embedding)
    top_index = np.argmax(similarities)

    return chunks[top_index]


# --- STEP 5: Ask LLM ---
def ask_llm(query, context):
    prompt = f"""
    Answer using ONLY the context below.

    Context:
    {context}

    Question:
    {query}
    """

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are a precise assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


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
