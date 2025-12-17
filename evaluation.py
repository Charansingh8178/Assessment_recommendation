import pandas as pd
import numpy as np


from baseline_retrieval import retrieve_assessments, balance_recommendations


train_df = pd.read_excel("train_data.xlsx" , sheet_name='Train-Set')
train_df.columns = train_df.columns.str.strip().str.lower()

print(f"[INFO] Loaded {len(train_df)} training queries")

def normalize_url(url):
    
    if not isinstance(url, str):
        return None

    url = url.strip().lower()
    url = url.split("?")[0]
    url = url.rstrip("/")
    if "view/" in url:
        return url.split("view/")[-1]

    return url

def recall_at_k(predicted_ids, true_ids, k=10):
    predicted_ids = predicted_ids[:k]
    true_ids = set(true_ids)

    if not true_ids:
        return 0.0

    hits = sum(1 for pid in predicted_ids if pid in true_ids)
    return hits / len(true_ids)

recall_scores = []

for idx, row in train_df.iterrows():
    query = row["query"]

    true_ids = [
        normalize_url(u)
        for u in str(row["assessment_url"]).split(",")
    ]

    candidates = retrieve_assessments(query, top_k=40)
    final_results = balance_recommendations(candidates, final_k=10)

    predicted_ids = [
        normalize_url(u)
        for u in final_results["url"].tolist()
    ]

    score = recall_at_k(predicted_ids, true_ids, k=10)
    recall_scores.append(score)

    print(f"[QUERY {idx+1}] Recall@10 = {score:.3f}")

mean_recall = np.mean(recall_scores)

print(f"MEAN RECALL@10 = {mean_recall:.3f}")
