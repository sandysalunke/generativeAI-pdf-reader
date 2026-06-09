import os
import pickle
from pypdf import PdfReader
from config import DATA_PATH

def load_data():
    try:
        embeddings = pickle.load(open(DATA_PATH + "embeddings.pkl", "rb"))
        chunks = pickle.load(open(DATA_PATH + "chunks.pkl", "rb"))
        return embeddings, chunks
    except:
        return None, None

def save_data(embeddings, chunks):
    os.makedirs(DATA_PATH, exist_ok=True)
    pickle.dump(embeddings, open(DATA_PATH + "embeddings.pkl", "wb"))
    pickle.dump(chunks, open(DATA_PATH + "chunks.pkl", "wb"))

def chunk_text(text, size=300):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

# --- Helpers ---
def load_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t
    return text
