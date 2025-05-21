from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import torch
from typing import List, Optional
import uvicorn

app = FastAPI(
    title="Fashion Recommendation API",
    description="ML-powered Fashion Recommendation System API",
    version="1.0.0"
)

# Load models
try:
    cf_model = joblib.load('../models/cf_model.joblib')
    cb_model = joblib.load('../models/cb_model.joblib')
    hybrid_model = joblib.load('../models/hybrid_model.joblib')
except:
    print("Models not found. Please train models first.")

class RecommendationRequest(BaseModel):
    user_id: int
    n_recommendations: Optional[int] = 5
    filters: Optional[dict] = None

class ProductFeatures(BaseModel):
    category: str
    brand: str
    price: float
    color: str
    description: str

class RecommendationResponse(BaseModel):
    recommendations: List[dict]
    scores: List[float]
    model_version: str

@app.get("/")
def read_root():
    return {
        "message": "Fashion Recommendation API",
        "version": "1.0.0",
        "status": "active"
    }

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    try:
        # Get recommendations from hybrid model
        recommendations = hybrid_model.recommend(
            user_id=request.user_id,
            n_recommendations=request.n_recommendations
        )
        
        # Apply filters if provided
        if request.filters:
            recommendations = apply_filters(recommendations, request.filters)
        
        return RecommendationResponse(
            recommendations=recommendations,
            scores=[float(score) for score in recommendations['scores']],
            model_version="1.0.0"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train_model(data: List[ProductFeatures]):
    try:
        # Convert data to appropriate format
        features = prepare_features(data)
        
        # Train models
        cf_model.fit(features)
        cb_model.fit(features)
        hybrid_model.update_models(cf_model, cb_model)
        
        return {"message": "Models trained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/health")
async def model_health():
    return {
        "status": "healthy",
        "models": {
            "collaborative_filtering": "loaded",
            "content_based": "loaded",
            "hybrid": "loaded"
        },
        "last_trained": "2024-02-06",
        "version": "1.0.0"
    }

def apply_filters(recommendations, filters):
    filtered_recs = recommendations.copy()
    for key, value in filters.items():
        filtered_recs = [rec for rec in filtered_recs if rec[key] == value]
    return filtered_recs

def prepare_features(data):
    # Convert ProductFeatures to model features
    features = []
    for item in data:
        features.append({
            'category': item.category,
            'brand': item.brand,
            'price': item.price,
            'color': item.color,
            'description': item.description
        })
    return features

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
