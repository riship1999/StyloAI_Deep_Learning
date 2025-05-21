import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
from dataclasses import dataclass
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PredictionLog:
    timestamp: str
    model_name: str
    model_version: str
    input_features: Dict
    prediction: float
    actual: Optional[float] = None
    latency_ms: Optional[float] = None

class ModelMonitor:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.predictions_file = os.path.join(log_dir, 'predictions.csv')
        self.metrics_file = os.path.join(log_dir, 'metrics.json')
        self._init_files()
    
    def _init_files(self):
        """Initialize log files if they don't exist"""
        if not os.path.exists(self.predictions_file):
            pd.DataFrame(columns=[
                'timestamp', 'model_name', 'model_version', 
                'input_features', 'prediction', 'actual', 'latency_ms'
            ]).to_csv(self.predictions_file, index=False)
        
        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'w') as f:
                json.dump({}, f)
    
    def log_prediction(self, log: PredictionLog):
        """Log a single prediction"""
        df = pd.DataFrame([{
            'timestamp': log.timestamp,
            'model_name': log.model_name,
            'model_version': log.model_version,
            'input_features': json.dumps(log.input_features),
            'prediction': log.prediction,
            'actual': log.actual,
            'latency_ms': log.latency_ms
        }])
        
        df.to_csv(self.predictions_file, mode='a', header=False, index=False)
    
    def calculate_metrics(self, 
                        start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None) -> Dict[str, Dict]:
        """Calculate monitoring metrics for the specified time period"""
        # Load predictions
        df = pd.read_csv(self.predictions_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter by time period
        if start_time:
            df = df[df['timestamp'] >= start_time]
        if end_time:
            df = df[df['timestamp'] <= end_time]
        
        metrics = {}
        for model_name in df['model_name'].unique():
            model_df = df[df['model_name'] == model_name]
            
            # Performance metrics
            performance = {
                'total_predictions': len(model_df),
                'avg_latency_ms': model_df['latency_ms'].mean(),
                'p95_latency_ms': model_df['latency_ms'].quantile(0.95),
                'p99_latency_ms': model_df['latency_ms'].quantile(0.99)
            }
            
            # Accuracy metrics (if actuals available)
            accuracy = {}
            if not model_df['actual'].isna().all():
                valid_preds = model_df.dropna(subset=['actual'])
                accuracy.update({
                    'mse': ((valid_preds['prediction'] - valid_preds['actual']) ** 2).mean(),
                    'mae': (valid_preds['prediction'] - valid_preds['actual']).abs().mean(),
                    'rmse': np.sqrt(((valid_preds['prediction'] - valid_preds['actual']) ** 2).mean())
                })
            
            metrics[model_name] = {
                'performance': performance,
                'accuracy': accuracy,
                'last_update': datetime.now().isoformat()
            }
        
        # Save metrics
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def generate_monitoring_dashboard(self, 
                                   model_name: str,
                                   last_n_hours: int = 24) -> Dict[str, go.Figure]:
        """Generate monitoring dashboard plots"""
        # Load data
        df = pd.read_csv(self.predictions_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter data
        start_time = datetime.now() - timedelta(hours=last_n_hours)
        df = df[
            (df['model_name'] == model_name) & 
            (df['timestamp'] >= start_time)
        ]
        
        # Create plots
        plots = {}
        
        # 1. Prediction Volume
        volume_df = df.set_index('timestamp').resample('1H').size()
        plots['prediction_volume'] = go.Figure(
            data=[go.Scatter(x=volume_df.index, y=volume_df.values)],
            layout=go.Layout(
                title="Prediction Volume (hourly)",
                xaxis_title="Time",
                yaxis_title="Number of Predictions"
            )
        )
        
        # 2. Latency Distribution
        plots['latency_distribution'] = go.Figure(
            data=[go.Histogram(x=df['latency_ms'], nbinsx=50)],
            layout=go.Layout(
                title="Latency Distribution",
                xaxis_title="Latency (ms)",
                yaxis_title="Count"
            )
        )
        
        # 3. Prediction vs Actual (if available)
        if not df['actual'].isna().all():
            plots['prediction_vs_actual'] = go.Figure(
                data=[go.Scatter(
                    x=df['actual'],
                    y=df['prediction'],
                    mode='markers',
                    marker=dict(size=6)
                )],
                layout=go.Layout(
                    title="Predictions vs Actuals",
                    xaxis_title="Actual Values",
                    yaxis_title="Predicted Values"
                )
            )
        
        return plots
    
    def detect_anomalies(self, 
                        model_name: str,
                        window_size: str = '1H',
                        threshold: float = 2.0) -> List[Dict]:
        """Detect anomalies in model predictions"""
        df = pd.read_csv(self.predictions_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        model_df = df[df['model_name'] == model_name]
        
        anomalies = []
        
        # 1. Latency anomalies
        latency_stats = model_df.set_index('timestamp')['latency_ms'].rolling(window_size)
        mean_latency = latency_stats.mean()
        std_latency = latency_stats.std()
        
        latency_anomalies = model_df[
            abs(model_df['latency_ms'] - mean_latency) > (threshold * std_latency)
        ]
        
        for _, row in latency_anomalies.iterrows():
            anomalies.append({
                'timestamp': row['timestamp'],
                'type': 'latency',
                'value': row['latency_ms'],
                'threshold': mean_latency + (threshold * std_latency),
                'severity': 'high' if row['latency_ms'] > mean_latency + (3 * std_latency) else 'medium'
            })
        
        # 2. Prediction distribution anomalies
        if not model_df['actual'].isna().all():
            error_stats = (model_df['prediction'] - model_df['actual']).abs().rolling(window_size)
            mean_error = error_stats.mean()
            std_error = error_stats.std()
            
            pred_anomalies = model_df[
                abs(model_df['prediction'] - model_df['actual']) > (threshold * std_error)
            ]
            
            for _, row in pred_anomalies.iterrows():
                anomalies.append({
                    'timestamp': row['timestamp'],
                    'type': 'prediction_error',
                    'value': abs(row['prediction'] - row['actual']),
                    'threshold': mean_error + (threshold * std_error),
                    'severity': 'high' if abs(row['prediction'] - row['actual']) > mean_error + (3 * std_error) else 'medium'
                })
        
        return anomalies
