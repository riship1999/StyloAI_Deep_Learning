from dataclasses import dataclass
from typing import List, Optional, Dict

@dataclass
class ModelConfig:
    model_name: str
    version: str
    input_features: List[str]
    output_features: List[str]
    model_type: str
    hyperparameters: Dict
    
# Model configurations
COLLABORATIVE_FILTERING_CONFIG = ModelConfig(
    model_name="collaborative_filtering",
    version="1.0.0",
    input_features=["user_id", "item_id"],
    output_features=["rating"],
    model_type="matrix_factorization",
    hyperparameters={
        "n_factors": 100,
        "learning_rate": 0.001,
        "regularization": 0.01
    }
)

CONTENT_BASED_CONFIG = ModelConfig(
    model_name="content_based",
    version="1.0.0",
    input_features=["category", "brand", "description", "price", "color"],
    output_features=["similarity_score"],
    model_type="tfidf_cosine",
    hyperparameters={
        "max_features": 5000,
        "ngram_range": (1, 2)
    }
)

HYBRID_CONFIG = ModelConfig(
    model_name="hybrid_recommender",
    version="1.0.0",
    input_features=["user_features", "item_features"],
    output_features=["recommendation_score"],
    model_type="hybrid",
    hyperparameters={
        "cf_weight": 0.4,
        "cb_weight": 0.3,
        "dl_weight": 0.3
    }
)
