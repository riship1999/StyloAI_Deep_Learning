import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import os
from ml_service.model_monitoring import ModelMonitor
from ml_service.model_versioning import ModelVersion
from ml_service.ab_testing import ABTesting

st.set_page_config(page_title="ML Dashboard", page_icon="📊", layout="wide")

def load_model_metrics():
    """Load model metrics from the monitoring system"""
    monitor = ModelMonitor('logs')
    return monitor.calculate_metrics()

def load_model_versions():
    """Load model version history"""
    version_manager = ModelVersion('models')
    models = ['collaborative_filtering', 'content_based', 'hybrid']
    versions = {}
    for model in models:
        try:
            versions[model] = version_manager.list_versions(model)
        except:
            versions[model] = {}
    return versions

def load_ab_test_results():
    """Load A/B test results"""
    ab_testing = ABTesting('experiments')
    try:
        with open(os.path.join('experiments', 'experiments.json'), 'r') as f:
            experiments = json.load(f)
        return experiments
    except:
        return {}

def render_metrics_dashboard(metrics):
    """Render the metrics dashboard"""
    st.header("Model Performance Metrics")
    
    for model_name, model_metrics in metrics.items():
        st.subheader(f"{model_name.title()} Model")
        
        # Create columns for different metric categories
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Performance Metrics")
            metrics_df = pd.DataFrame({
                'Metric': ['Total Predictions', 'Avg Latency (ms)', 'P95 Latency (ms)', 'P99 Latency (ms)'],
                'Value': [
                    model_metrics['performance']['total_predictions'],
                    round(model_metrics['performance']['avg_latency_ms'], 2),
                    round(model_metrics['performance']['p95_latency_ms'], 2),
                    round(model_metrics['performance']['p99_latency_ms'], 2)
                ]
            })
            st.dataframe(metrics_df)
        
        with col2:
            if 'accuracy' in model_metrics and model_metrics['accuracy']:
                st.write("Accuracy Metrics")
                accuracy_df = pd.DataFrame({
                    'Metric': ['MSE', 'MAE', 'RMSE'],
                    'Value': [
                        round(model_metrics['accuracy']['mse'], 4),
                        round(model_metrics['accuracy']['mae'], 4),
                        round(model_metrics['accuracy']['rmse'], 4)
                    ]
                })
                st.dataframe(accuracy_df)

def render_version_history(versions):
    """Render model version history"""
    st.header("Model Version History")
    
    for model_name, model_versions in versions.items():
        st.subheader(f"{model_name.title()} Model Versions")
        
        if model_versions:
            # Create a DataFrame from version history
            version_data = []
            for version_id, info in model_versions.items():
                version_data.append({
                    'Version ID': version_id,
                    'Timestamp': info['timestamp'],
                    'MSE': round(info['metrics'].get('mse', 0), 4),
                    'RMSE': round(info['metrics'].get('rmse', 0), 4),
                    'MAE': round(info['metrics'].get('mae', 0), 4)
                })
            
            df = pd.DataFrame(version_data)
            st.dataframe(df)
            
            # Plot metrics evolution
            if len(df) > 1:
                fig = go.Figure()
                for metric in ['MSE', 'RMSE', 'MAE']:
                    fig.add_trace(go.Scatter(
                        x=df['Timestamp'],
                        y=df[metric],
                        name=metric,
                        mode='lines+markers'
                    ))
                fig.update_layout(
                    title=f"{model_name.title()} Model Metrics Evolution",
                    xaxis_title="Time",
                    yaxis_title="Value"
                )
                st.plotly_chart(fig)
        else:
            st.write("No version history available")

def render_ab_testing_dashboard(experiments):
    """Render A/B testing dashboard"""
    st.header("A/B Testing Dashboard")
    
    if experiments:
        for exp_name, exp_info in experiments.items():
            st.subheader(f"Experiment: {exp_name}")
            
            # Create columns for experiment details
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Experiment Configuration")
                config_df = pd.DataFrame({
                    'Parameter': [
                        'Start Date',
                        'End Date',
                        'Traffic Split',
                        'Min Sample Size',
                        'Status'
                    ],
                    'Value': [
                        exp_info['start_date'],
                        exp_info['end_date'],
                        f"{exp_info['traffic_split'] * 100}%",
                        exp_info['min_sample_size'],
                        exp_info['status']
                    ]
                })
                st.dataframe(config_df)
            
            with col2:
                st.write("Variants")
                variant_df = pd.DataFrame({
                    'Variant': ['A', 'B'],
                    'Model Version': [exp_info['variant_a'], exp_info['variant_b']]
                })
                st.dataframe(variant_df)
    else:
        st.write("No active experiments")

def main():
    st.title("ML System Dashboard")
    
    # Add tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Performance Metrics", "Version History", "A/B Testing"])
    
    # Load data
    metrics = load_model_metrics()
    versions = load_model_versions()
    experiments = load_ab_test_results()
    
    # Render different sections in tabs
    with tab1:
        render_metrics_dashboard(metrics)
    
    with tab2:
        render_version_history(versions)
    
    with tab3:
        render_ab_testing_dashboard(experiments)

if __name__ == "__main__":
    main()
