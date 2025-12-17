import pandas as pd

from baseline_retrieval import retrieve_assessments, balance_recommendations


test_df = pd.read_excel("train_data.xlsx", sheet_name="Test-Set")
test_df.columns = test_df.columns.str.strip().str.lower()

print(f"[INFO] Loaded {len(test_df)} test queries")


output_rows = []

for idx, row in test_df.iterrows():
    query = row["query"]
    candidates = retrieve_assessments(query, top_k=40)
    final_results = balance_recommendations(candidates, final_k=10)

    urls = final_results["url"].tolist()

    if len(urls) < 10:
        extra = candidates["url"].tolist()
        for u in extra:
            if u not in urls:
                urls.append(u)
            if len(urls) == 10:
                break

    urls = urls[:10]

    output_rows.append({
        "Query": query,
        "Assessment_url": ",".join(urls)
    })

    print(f"[TEST QUERY {idx+1}] Generated {len(urls)} predictions")


output_df = pd.DataFrame(output_rows)
output_df.to_csv("test_predictions.csv", index=False)

print("\n[DONE] Saved test_predictions.csv")
