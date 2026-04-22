from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# ✅ Load model once (global, not per request)
model = SentenceTransformer("all-MiniLM-L6-v2")
model.max_seq_length = 256  # 🔥 speed optimization

# ✅ Correct file path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(BASE_DIR, "rag/data/derma_knowledge.txt")

# ✅ Load and clean docs
with open(file_path, "r") as f:
    docs = [line.strip() for line in f.readlines() if line.strip()]

# ✅ Precompute embeddings ONCE
embeddings = model.encode(docs, convert_to_numpy=True)

# ✅ Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)


# 🔥 FAST retrieval
def retrieve(query, k=2):
    q_emb = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(q_emb, k)

    # return top-k relevant docs
    return [docs[i] for i in indices[0] if i < len(docs)]