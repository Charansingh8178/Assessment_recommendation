

from pydantic import BaseModel

from fastapi import FastAPI
from pydantic import BaseModel

from typing import List

from baseline_retrieval import retrieve_assessments, balance_recommendations

app = FastAPI(
    title="SHL Assessment  API",
    description="Semantic RAG-based system for recommending SHL assessments",
    
)


class RecommendRequest(BaseModel):
    query: str

class AssessmentResponse(BaseModel):
    assessment_name: str
    url: str
    test_type: str

    assessment_length: str
    remote_testing: str
    adaptive_support: str

class RecommendResponse(BaseModel):
    query: str
    recommendations: List[AssessmentResponse]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    query = req.query

    
    candidates = retrieve_assessments(query, top_k=40)

    final_results = balance_recommendations(candidates, final_k=10)

    recommendations = []

    for _, row in final_results.iterrows():
        recommendations.append(
            AssessmentResponse(
                assessment_name=row["assessment_name"],
                url=row["url"],
                test_type=row["test_type"],
                assessment_length=row["assessment_length"],
                remote_testing=row["remote_testing"],
                adaptive_support=row["adaptive_support"],
            )
        )

    return RecommendResponse(
        query=query,
        recommendations=recommendations
    )
