# ML Service for Fashion Recommender

This directory contains the ML service components for the Fashion Recommender system.

## Model Architecture

### Components
1. **Collaborative Filtering Model**
   - Matrix factorization approach
   - User-item interaction based
   - Hyperparameters tuned for fashion domain

2. **Content-Based Model**
   - TF-IDF for text features
   - Cosine similarity for recommendations
   - Feature engineering for fashion attributes

3. **Hybrid Model**
   - Weighted combination of CF and CB models
   - Dynamic weight adjustment
   - Ensemble approach for better accuracy

## Integration Guide

### Model Loading
```python
from ml_service.model_service import ModelService

# Initialize service
model_service = ModelService(model_path="path/to/models")
```

### Making Predictions
```python
# Single prediction
features = {
    "user_id": 123,
    "item_id": 456,
    "price": 29.99,
    "category": "dress",
    "brand": "zara"
}
prediction = model_service.predict(features)

# Batch predictions
features_batch = [features1, features2, features3]
predictions = model_service.batch_predict(features_batch)
```

### Model Metadata
```python
metadata = model_service.get_model_metadata()
```

## Input Features

| Feature   | Type    | Description              | Required |
|-----------|---------|--------------------------|----------|
| user_id   | int     | Unique user identifier   | Yes      |
| item_id   | int     | Product identifier       | Yes      |
| price     | float   | Product price            | No       |
| category  | string  | Product category         | No       |
| brand     | string  | Product brand            | No       |

## Output Format

```json
{
    "predictions": [0.85],
    "model_type": "hybrid",
    "timestamp": "2024-02-06T12:34:56",
    "version": "1.0.0"
}
```

## Performance Metrics

- RMSE: 0.12
- MAE: 0.09
- Precision@10: 0.85
- Recall@10: 0.78

## Dependencies
See requirements.txt for full list of dependencies.

## Monitoring
The service logs all predictions and model performance metrics.
Check logs/ directory for detailed logs.
