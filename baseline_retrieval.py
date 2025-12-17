import os
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ----------------------------
# CONFIG
# ----------------------------
EMBEDDING_FILE = "assessment_embeddings.npy"
FAISS_INDEX_FILE = "faiss.index"


df = pd.read_csv("shl_assessments.csv")

def build_assessment_text(row):
    parts = [
        str(row.get("assessment_name", "")),
        str(row.get("description", "")),
        f"Test type: {row.get('test_type', '')}",
        f"Assessment length: {row.get('assessment_length', '')}",
        f"Job levels: {row.get('job_levels', '')}",
        f"Languages: {row.get('languages', '')}"
    ]
    return ". ".join([p for p in parts if p and p != "Not Available"])

texts = df.apply(build_assessment_text, axis=1).tolist()

model = SentenceTransformer("all-MiniLM-L6-v2")

if os.path.exists(EMBEDDING_FILE) and os.path.exists(FAISS_INDEX_FILE):
    print("[INFO] Loading cached embeddings and FAISS index...")
    embeddings = np.load(EMBEDDING_FILE)
    index = faiss.read_index(FAISS_INDEX_FILE)
else:
    print("[INFO] Creating embeddings for the first time...")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    
    np.save(EMBEDDING_FILE, embeddings)
    faiss.write_index(index, FAISS_INDEX_FILE)

    print("[INFO] Embeddings and index saved to disk")

def retrieve_assessments(query, top_k=40):
    query_embedding = model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)

    results = df.iloc[indices[0]].copy()
    results["score"] = distances[0]
    return results
import pandas as pd

def balance_recommendations(df, final_k=10):
    """
    Rerank recommendations to ensure balanced test types
    """

    if df is None or df.empty:
        return df

    selected = []
    used_indices = set()
    def pick(condition):
        for idx, row in df.iterrows():
            if idx not in used_indices and condition(row):
                used_indices.add(idx)
                selected.append(row)
                return True
        return False

    used_indices.add(df.index[0])
    selected.append(df.iloc[0])

    pick(lambda r: r["test_type"] in ["Ability & Aptitude", "Knowledge & Skills"])

    pick(lambda r: r["test_type"] in [
        "Personality & Behaviour",
        "Biodata & Situational Judgement"
    ])

    for idx, row in df.iterrows():
        if len(selected) >= final_k:
            break
        if idx not in used_indices:
            used_indices.add(idx)
            selected.append(row)

    return pd.DataFrame(selected)
