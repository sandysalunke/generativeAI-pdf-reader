from config import client, WHISPER_MODEL, EMBED_MODEL, CHAT_MODEL, DATA_PATH
import numpy as np
import pickle

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
