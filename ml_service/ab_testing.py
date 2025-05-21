import numpy as np
from scipy import stats
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import json
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    name: str
    variant_a: str  # Control variant (current model)
    variant_b: str  # Test variant (new model)
    traffic_split: float  # Percentage of traffic to variant B (0-1)
    metrics: List[str]  # Metrics to track
    start_date: datetime
    end_date: datetime
    min_sample_size: int

class ABTesting:
    def __init__(self, experiments_dir: str):
        self.experiments_dir = experiments_dir
        os.makedirs(experiments_dir, exist_ok=True)
        self.experiments_file = os.path.join(experiments_dir, 'experiments.json')
        self.results_file = os.path.join(experiments_dir, 'results.csv')
        self._init_files()
        
    def _init_files(self):
        """Initialize experiment files"""
        if not os.path.exists(self.experiments_file):
            with open(self.experiments_file, 'w') as f:
                json.dump({}, f)
        
        if not os.path.exists(self.results_file):
            pd.DataFrame(columns=[
                'timestamp', 'experiment_name', 'variant',
                'user_id', 'prediction', 'actual', 'metrics'
            ]).to_csv(self.results_file, index=False)
    
    def create_experiment(self, config: ExperimentConfig) -> Dict:
        """Create a new A/B test experiment"""
        # Load existing experiments
        with open(self.experiments_file, 'r') as f:
            experiments = json.load(f)
        
        # Create experiment config
        experiment = {
            'name': config.name,
            'variant_a': config.variant_a,
            'variant_b': config.variant_b,
            'traffic_split': config.traffic_split,
            'metrics': config.metrics,
            'start_date': config.start_date.isoformat(),
            'end_date': config.end_date.isoformat(),
            'min_sample_size': config.min_sample_size,
            'status': 'active',
            'created_at': datetime.now().isoformat()
        }
        
        experiments[config.name] = experiment
        
        # Save updated experiments
        with open(self.experiments_file, 'w') as f:
            json.dump(experiments, f, indent=2)
        
        logger.info(f"Created experiment: {config.name}")
        return experiment
    
    def assign_variant(self, experiment_name: str, user_id: int) -> str:
        """Assign a user to a variant based on consistent hashing"""
        # Load experiment
        with open(self.experiments_file, 'r') as f:
            experiment = json.load(f)[experiment_name]
        
        # Use hash of user_id for consistent assignment
        hash_value = hash(f"{experiment_name}_{user_id}") % 100
        return experiment['variant_b'] if hash_value < (experiment['traffic_split'] * 100) else experiment['variant_a']
    
    def log_result(self, 
                  experiment_name: str,
                  variant: str,
                  user_id: int,
                  prediction: float,
                  actual: Optional[float],
                  metrics: Dict[str, float]):
        """Log an experiment result"""
        df = pd.DataFrame([{
            'timestamp': datetime.now().isoformat(),
            'experiment_name': experiment_name,
            'variant': variant,
            'user_id': user_id,
            'prediction': prediction,
            'actual': actual,
            'metrics': json.dumps(metrics)
        }])
        
        df.to_csv(self.results_file, mode='a', header=False, index=False)
    
    def analyze_experiment(self, experiment_name: str) -> Dict:
        """Analyze experiment results"""
        # Load experiment config
        with open(self.experiments_file, 'r') as f:
            experiment = json.load(f)[experiment_name]
        
        # Load results
        df = pd.read_csv(self.results_file)
        exp_df = df[df['experiment_name'] == experiment_name]
        
        if len(exp_df) < experiment['min_sample_size']:
            logger.warning(f"Insufficient samples for experiment {experiment_name}")
            return {'status': 'insufficient_data'}
        
        results = {
            'sample_sizes': {
                'A': len(exp_df[exp_df['variant'] == experiment['variant_a']]),
                'B': len(exp_df[exp_df['variant'] == experiment['variant_b']])
            }
        }
        
        # Analyze each metric
        for metric in experiment['metrics']:
            # Get metric values for each variant
            variant_a_values = self._extract_metric_values(exp_df, experiment['variant_a'], metric)
            variant_b_values = self._extract_metric_values(exp_df, experiment['variant_b'], metric)
            
            # Calculate statistics
            t_stat, p_value = stats.ttest_ind(variant_a_values, variant_b_values)
            effect_size = (np.mean(variant_b_values) - np.mean(variant_a_values)) / np.std(variant_a_values)
            
            results[metric] = {
                'mean_A': float(np.mean(variant_a_values)),
                'mean_B': float(np.mean(variant_b_values)),
                'std_A': float(np.std(variant_a_values)),
                'std_B': float(np.std(variant_b_values)),
                'p_value': float(p_value),
                'effect_size': float(effect_size),
                'significant': p_value < 0.05,
                'improvement': float(((np.mean(variant_b_values) - np.mean(variant_a_values)) 
                                   / np.mean(variant_a_values)) * 100)
            }
        
        return results
    
    def _extract_metric_values(self, df: pd.DataFrame, variant: str, metric: str) -> np.ndarray:
        """Extract metric values from results dataframe"""
        variant_df = df[df['variant'] == variant]
        return variant_df.apply(lambda row: json.loads(row['metrics'])[metric], axis=1).values
    
    def generate_experiment_dashboard(self, experiment_name: str) -> Dict[str, go.Figure]:
        """Generate dashboard plots for experiment analysis"""
        # Load results
        df = pd.read_csv(self.results_file)
        exp_df = df[df['experiment_name'] == experiment_name]
        
        plots = {}
        
        # 1. Sample Size Evolution
        sample_sizes = exp_df.groupby(['variant', pd.to_datetime(exp_df['timestamp']).dt.date]).size().unstack(0)
        plots['sample_size'] = go.Figure(
            data=[
                go.Scatter(x=sample_sizes.index, y=sample_sizes[col], name=col)
                for col in sample_sizes.columns
            ],
            layout=go.Layout(
                title="Sample Size Evolution",
                xaxis_title="Date",
                yaxis_title="Number of Samples"
            )
        )
        
        # 2. Metric Distributions
        metrics = json.loads(exp_df.iloc[0]['metrics']).keys()
        for metric in metrics:
            variant_a_values = self._extract_metric_values(exp_df, 'A', metric)
            variant_b_values = self._extract_metric_values(exp_df, 'B', metric)
            
            plots[f'distribution_{metric}'] = go.Figure(
                data=[
                    go.Histogram(x=variant_a_values, name='Variant A', opacity=0.75),
                    go.Histogram(x=variant_b_values, name='Variant B', opacity=0.75)
                ],
                layout=go.Layout(
                    title=f"{metric} Distribution by Variant",
                    barmode='overlay',
                    xaxis_title=metric,
                    yaxis_title="Count"
                )
            )
        
        return plots
    
    def get_experiment_status(self, experiment_name: str) -> Dict:
        """Get current status of an experiment"""
        # Load experiment config
        with open(self.experiments_file, 'r') as f:
            experiment = json.load(f)[experiment_name]
        
        # Load results
        df = pd.read_csv(self.results_file)
        exp_df = df[df['experiment_name'] == experiment_name]
        
        current_samples = len(exp_df)
        progress = (current_samples / experiment['min_sample_size']) * 100
        
        return {
            'name': experiment_name,
            'status': experiment['status'],
            'progress': min(100, progress),
            'current_samples': current_samples,
            'target_samples': experiment['min_sample_size'],
            'days_remaining': (
                datetime.fromisoformat(experiment['end_date']) - datetime.now()
            ).days
        }
