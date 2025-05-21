import joblib
import numpy as np
from typing import Dict, List, Any
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.models = {}
        self.load_models()
        
    def load_models(self):
        """Load all trained models"""
        try:
            self.models['cf'] = joblib.load(f"{self.model_path}/cf_model.joblib")
            self.models['cb'] = joblib.load(f"{self.model_path}/cb_model.joblib")
            self.models['hybrid'] = joblib.load(f"{self.model_path}/hybrid_model.joblib")
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def preprocess_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Preprocess input features"""
        try:
            # Convert features to model input format
            processed_features = np.array([
                features.get('user_id', 0),
                features.get('item_id', 0),
                features.get('price', 0.0),
                features.get('category_encoded', 0),
                features.get('brand_encoded', 0)
            ]).reshape(1, -1)
            
            return processed_features
        except Exception as e:
            logger.error(f"Error preprocessing features: {str(e)}")
            raise
    
    def predict(self, features: Dict[str, Any], model_type: str = 'hybrid') -> Dict[str, Any]:
        """Generate predictions using specified model"""
        try:
            # Preprocess features
            processed_features = self.preprocess_features(features)
            
            # Get model predictions
            model = self.models.get(model_type)
            if not model:
                raise ValueError(f"Model type {model_type} not found")
            
            predictions = model.predict(processed_features)
            
            # Format response
            response = {
                'predictions': predictions.tolist(),
                'model_type': model_type,
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0'
            }
            
            return response
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            raise
    
    def batch_predict(self, features_batch: List[Dict[str, Any]], model_type: str = 'hybrid') -> List[Dict[str, Any]]:
        """Generate predictions for a batch of inputs"""
        return [self.predict(features, model_type) for features in features_batch]
    
    def get_model_metadata(self) -> Dict[str, Any]:
        """Get metadata about loaded models"""
        return {
            'models': list(self.models.keys()),
            'versions': {
                'cf': '1.0.0',
                'cb': '1.0.0',
                'hybrid': '1.0.0'
            },
            'last_updated': datetime.now().isoformat(),
            'features_supported': [
                'user_id',
                'item_id',
                'price',
                'category',
                'brand'
            ]
        }
