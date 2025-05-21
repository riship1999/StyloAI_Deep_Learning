# MLflow-integrated training script for the Fashion Recommendation System
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import os
from datetime import datetime
import json
import mlflow
import mlflow.sklearn
import mlflow.pytorch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CollaborativeFilteringModel:
    def __init__(self, n_factors=100, learning_rate=0.001, regularization=0.01):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.user_factors = None
        self.item_factors = None
    
    def fit(self, ratings_df):
        # Remove duplicates and aggregate ratings
        ratings_df = ratings_df.groupby(['user_id', 'item_id'])['rating'].mean().reset_index()
        
        # Create user-item matrix
        self.user_item_matrix = ratings_df.pivot(
            index='user_id', 
            columns='item_id', 
            values='rating'
        ).fillna(0)
        
        # Perform SVD
        U, sigma, Vt = np.linalg.svd(self.user_item_matrix.values, full_matrices=False)
        
        # Keep only top n_factors
        self.user_factors = U[:, :self.n_factors] @ np.diag(np.sqrt(sigma[:self.n_factors]))
        self.item_factors = np.diag(np.sqrt(sigma[:self.n_factors])) @ Vt[:self.n_factors, :]
        
        return self
    
    def predict(self, user_ids, item_ids):
        return np.sum(self.user_factors[user_ids] * self.item_factors[:, item_ids].T, axis=1)

class ContentBasedModel:
    def __init__(self, max_features=5000):
        self.max_features = max_features
        self.tfidf = TfidfVectorizer(max_features=max_features)
        
    def fit(self, items_df):
        # Combine text features
        text_features = items_df['description'] + ' ' + items_df['category'] + ' ' + items_df['brand']
        
        # Create TF-IDF matrix
        self.feature_matrix = self.tfidf.fit_transform(text_features)
        
        # Calculate similarity matrix
        self.similarity_matrix = cosine_similarity(self.feature_matrix)
        
        # Store item IDs for prediction
        self.item_ids = items_df.index.values
        
        return self
    
    def predict(self, item_ids):
        # Map item IDs to matrix indices
        indices = [np.where(self.item_ids == item_id)[0][0] for item_id in item_ids]
        return self.similarity_matrix[indices]

class HybridModel(nn.Module):
    def __init__(self, cf_model, cb_model, input_size=6):  
        super(HybridModel, self).__init__()
        self.cf_model = cf_model
        self.cb_model = cb_model
        
        self.model = nn.Sequential(
            nn.Linear(input_size, 32),  
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# def prepare_data():
#     """Load and prepare data for training"""
#     logger.info("Loading data...")
    
#     # Load ratings data
#     ratings_df = pd.read_csv('data/ratings.csv')
    
#     # Load items data
#     items_df = pd.read_csv('data/items.csv')
#     items_df.set_index('item_id', inplace=True)
    
#     # Remove any duplicate ratings and aggregate
#     ratings_df = ratings_df.groupby(['user_id', 'item_id'])['rating'].mean().reset_index()
    
#     # Encode categorical variables
#     le = LabelEncoder()
#     items_df['category_encoded'] = le.fit_transform(items_df['category'])
#     items_df['brand_encoded'] = le.fit_transform(items_df['brand'])
    
#     return ratings_df, items_df

# def train_models():
#     """Train all recommendation models"""
#     logger.info("Starting model training...")
    
#     # Initialize versioning and monitoring
#     model_version = ModelVersion('models')
#     model_monitor = ModelMonitor('logs')
    
#     # Create models directory if it doesn't exist
#     os.makedirs('models', exist_ok=True)
    
#     # Load and prepare data
#     ratings_df, items_df = prepare_data()
    
#     # Split data
#     train_ratings, test_ratings = train_test_split(ratings_df, test_size=0.2, random_state=42)
    
#     # Train Collaborative Filtering model
#     logger.info("Training Collaborative Filtering model...")
#     cf_model = CollaborativeFilteringModel(n_factors=100)
#     cf_model.fit(train_ratings)
    
#     # Save CF model
#     cf_path = 'models/cf_model.joblib'
#     joblib.dump(cf_model, cf_path)
    
#     # Calculate CF metrics
#     cf_preds = cf_model.predict(test_ratings['user_id'].values, test_ratings['item_id'].values)
#     cf_metrics = {
#         'mse': ((cf_preds - test_ratings['rating'].values) ** 2).mean(),
#         'rmse': np.sqrt(((cf_preds - test_ratings['rating'].values) ** 2).mean()),
#         'mae': np.abs(cf_preds - test_ratings['rating'].values).mean()
#     }
    
#     # Version CF model
#     cf_version = model_version.save_version(
#         'collaborative_filtering',
#         cf_path,
#         cf_metrics,
#         metadata={'n_factors': 100}
#     )
#     model_version.set_current_version('collaborative_filtering', cf_version)
    
#     # Train Content-Based model
#     logger.info("Training Content-Based model...")
#     cb_model = ContentBasedModel(max_features=5000)
#     cb_model.fit(items_df)
    
#     # Save CB model
#     cb_path = 'models/cb_model.joblib'
#     joblib.dump(cb_model, cb_path)
    
#     # Calculate CB metrics
#     cb_preds = cb_model.predict(test_ratings['item_id'].values)
#     cb_metrics = {
#         'mean_similarity': cb_preds.mean(),
#         'coverage': (cb_preds > 0).mean()
#     }
    
#     # Version CB model
#     cb_version = model_version.save_version(
#         'content_based',
#         cb_path,
#         cb_metrics,
#         metadata={'max_features': 5000}
#     )
#     model_version.set_current_version('content_based', cb_version)
    
#     # Train Hybrid model
#     logger.info("Training Hybrid model...")
#     hybrid_model = HybridModel(cf_model, cb_model)
    
#     # Prepare features for hybrid model
#     cf_preds = cf_model.predict(train_ratings['user_id'].values, train_ratings['item_id'].values)
#     cb_preds = cb_model.predict(train_ratings['item_id'].values)
    
#     # Get item features
#     item_features = items_df.loc[train_ratings['item_id']]
    
#     X = np.column_stack([
#         cf_preds,
#         cb_preds.mean(axis=1),
#         train_ratings['user_id'].values,
#         train_ratings['item_id'].values,
#         item_features['category_encoded'].values,
#         item_features['brand_encoded'].values,
#     ])
    
#     y = train_ratings['rating'].values
    
#     # Convert to PyTorch tensors
#     X_tensor = torch.FloatTensor(X)
#     y_tensor = torch.FloatTensor(y)
    
#     # Train hybrid model
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(hybrid_model.parameters())
    
#     n_epochs = 10
#     batch_size = 64
    
#     for epoch in range(n_epochs):
#         total_loss = 0
#         n_batches = 0
        
#         for i in range(0, len(X), batch_size):
#             batch_X = X_tensor[i:i+batch_size]
#             batch_y = y_tensor[i:i+batch_size]
            
#             optimizer.zero_grad()
#             outputs = hybrid_model(batch_X)
#             loss = criterion(outputs, batch_y.unsqueeze(1))
#             loss.backward()
#             optimizer.step()
            
#             total_loss += loss.item()
#             n_batches += 1
        
#         avg_loss = total_loss / n_batches
#         logger.info(f"Epoch {epoch+1}/{n_epochs}, Average Loss: {avg_loss:.4f}")
    
#     # Save hybrid model
#     hybrid_path = 'models/hybrid_model.pth'
#     torch.save(hybrid_model.state_dict(), hybrid_path)
    
#     # Calculate hybrid metrics
#     test_cf_preds = cf_model.predict(test_ratings['user_id'].values, test_ratings['item_id'].values)
#     test_cb_preds = cb_model.predict(test_ratings['item_id'].values)
    
#     test_X = np.column_stack([
#         test_cf_preds,
#         test_cb_preds.mean(axis=1),
#         test_ratings['user_id'].values,
#         test_ratings['item_id'].values,
#         items_df.loc[test_ratings['item_id'], 'category_encoded'].values,
#         items_df.loc[test_ratings['item_id'], 'brand_encoded'].values,
#     ])
    
#     test_X_tensor = torch.FloatTensor(test_X)
#     test_y_tensor = torch.FloatTensor(test_ratings['rating'].values)
    
#     with torch.no_grad():
#         test_preds = hybrid_model(test_X_tensor)
#         test_loss = criterion(test_preds, test_y_tensor.unsqueeze(1))
        
#     hybrid_metrics = {
#         'test_mse': test_loss.item(),
#         'test_rmse': np.sqrt(test_loss.item()),
#         'final_train_loss': avg_loss
#     }
    
#     # Version hybrid model
#     hybrid_version = model_version.save_version(
#         'hybrid',
#         hybrid_path,
#         hybrid_metrics,
#         metadata={
#             'architecture': 'neural_network',
#             'n_epochs': n_epochs,
#             'batch_size': batch_size
#         }
#     )
#     model_version.set_current_version('hybrid', hybrid_version)
    
#     # Log initial monitoring data
#     for i in range(min(100, len(test_ratings))):
#         model_monitor.log_prediction(
#             PredictionLog(
#                 timestamp=datetime.now().isoformat(),
#                 model_name='hybrid',
#                 model_version=hybrid_version,
#                 input_features={
#                     'user_id': int(test_ratings.iloc[i]['user_id']),
#                     'item_id': int(test_ratings.iloc[i]['item_id'])
#                 },
#                 prediction=float(test_preds[i].item()),
#                 actual=float(test_ratings.iloc[i]['rating']),
#                 latency_ms=np.random.uniform(10, 50)  # Simulated latency
#             )
#         )
    
#     # Calculate initial monitoring metrics
#     monitoring_metrics = model_monitor.calculate_metrics()
#     logger.info("\nInitial Monitoring Metrics:")
#     logger.info(json.dumps(monitoring_metrics, indent=2))
    
#     return cf_model, cb_model, hybrid_model

# if __name__ == "__main__":
#     from ml_service.model_versioning import ModelVersion
#     from ml_service.model_monitoring import ModelMonitor, PredictionLog
#     train_models()


def prepare_data():
    logger.info("Loading data...")
    ratings_df = pd.read_csv('data/ratings.csv')
    items_df = pd.read_csv('data/items.csv')
    items_df.set_index('item_id', inplace=True)
    ratings_df = ratings_df.groupby(['user_id', 'item_id'])['rating'].mean().reset_index()
    le = LabelEncoder()
    items_df['category_encoded'] = le.fit_transform(items_df['category'])
    items_df['brand_encoded'] = le.fit_transform(items_df['brand'])
    return ratings_df, items_df

def train_models():
    from ml_service.model_versioning import ModelVersion
    from ml_service.model_monitoring import ModelMonitor, PredictionLog

    logger.info("Starting model training...")
    mlflow.set_experiment("fashion_recommendation")

    with mlflow.start_run(run_name="hybrid_recommender_training"):
        model_version = ModelVersion('models')
        model_monitor = ModelMonitor('logs')
        os.makedirs('models', exist_ok=True)
        ratings_df, items_df = prepare_data()
        train_ratings, test_ratings = train_test_split(ratings_df, test_size=0.2, random_state=42)

        # CF Model
        cf_model = CollaborativeFilteringModel(n_factors=100)
        cf_model.fit(train_ratings)
        cf_preds = cf_model.predict(test_ratings['user_id'].values, test_ratings['item_id'].values)
        cf_metrics = {
            'cf_mse': ((cf_preds - test_ratings['rating'].values) ** 2).mean(),
            'cf_rmse': np.sqrt(((cf_preds - test_ratings['rating'].values) ** 2).mean()),
            'cf_mae': np.abs(cf_preds - test_ratings['rating'].values).mean()
        }
        mlflow.log_params({'cf_n_factors': 100})
        mlflow.log_metrics(cf_metrics)
        joblib.dump(cf_model, 'models/cf_model.joblib')
        mlflow.log_artifact('models/cf_model.joblib')

        # CB Model
        cb_model = ContentBasedModel(max_features=5000)
        cb_model.fit(items_df)
        cb_preds = cb_model.predict(test_ratings['item_id'].values)
        cb_metrics = {
            'cb_mean_similarity': cb_preds.mean(),
            'cb_coverage': (cb_preds > 0).mean()
        }
        mlflow.log_params({'cb_max_features': 5000})
        mlflow.log_metrics(cb_metrics)
        joblib.dump(cb_model, 'models/cb_model.joblib')
        mlflow.log_artifact('models/cb_model.joblib')

        # Hybrid Model
        hybrid_model = HybridModel(cf_model, cb_model)
        cf_preds = cf_model.predict(train_ratings['user_id'].values, train_ratings['item_id'].values)
        cb_preds = cb_model.predict(train_ratings['item_id'].values)
        item_features = items_df.loc[train_ratings['item_id']]
        X = np.column_stack([
            cf_preds,
            cb_preds.mean(axis=1),
            train_ratings['user_id'].values,
            train_ratings['item_id'].values,
            item_features['category_encoded'].values,
            item_features['brand_encoded'].values,
        ])
        y = train_ratings['rating'].values
        X_tensor, y_tensor = torch.FloatTensor(X), torch.FloatTensor(y)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(hybrid_model.parameters())
        n_epochs = 10
        batch_size = 64
        for epoch in range(n_epochs):
            total_loss = 0
            for i in range(0, len(X), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]
                optimizer.zero_grad()
                outputs = hybrid_model(batch_X)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / (len(X) // batch_size)
            logger.info(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
            mlflow.log_metric(f"hybrid_train_loss_epoch_{epoch+1}", avg_loss)
        torch.save(hybrid_model.state_dict(), 'models/hybrid_model.pth')
        mlflow.pytorch.log_model(hybrid_model, "hybrid_model")

        # Final test metrics
        test_cf_preds = cf_model.predict(test_ratings['user_id'].values, test_ratings['item_id'].values)
        test_cb_preds = cb_model.predict(test_ratings['item_id'].values)
        test_X = np.column_stack([
            test_cf_preds,
            test_cb_preds.mean(axis=1),
            test_ratings['user_id'].values,
            test_ratings['item_id'].values,
            items_df.loc[test_ratings['item_id'], 'category_encoded'].values,
            items_df.loc[test_ratings['item_id'], 'brand_encoded'].values,
        ])
        test_X_tensor = torch.FloatTensor(test_X)
        test_y_tensor = torch.FloatTensor(test_ratings['rating'].values)
        with torch.no_grad():
            test_preds = hybrid_model(test_X_tensor)
            test_loss = criterion(test_preds, test_y_tensor.unsqueeze(1)).item()
        mlflow.log_metrics({
            'hybrid_test_mse': test_loss,
            'hybrid_test_rmse': np.sqrt(test_loss),
            'hybrid_final_train_loss': avg_loss
        })

if __name__ == "__main__":
    train_models()
