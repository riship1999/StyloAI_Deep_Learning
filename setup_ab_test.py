from ml_service.ab_testing import ABTesting, ExperimentConfig
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_ab_test():
    """Set up A/B test between model versions"""
    # Initialize AB testing
    ab_testing = ABTesting('experiments')
    
    # Create experiment config
    config = ExperimentConfig(
        name="hybrid_model_v2_test",
        variant_a="hybrid_20250206_v1",  # Current version
        variant_b="hybrid_20250206_v2",  # New version
        traffic_split=0.5,  # 50-50 split
        metrics=[
            'prediction_error',
            'latency_ms',
            'user_satisfaction'
        ],
        start_date=datetime.now(),
        end_date=datetime.now() + timedelta(days=14),  # 2-week test
        min_sample_size=1000
    )
    
    # Create experiment
    experiment = ab_testing.create_experiment(config)
    logger.info(f"Created experiment: {experiment['name']}")
    logger.info(f"Start date: {experiment['start_date']}")
    logger.info(f"End date: {experiment['end_date']}")
    
    # Example: Simulate some initial results
    import numpy as np
    for i in range(100):
        user_id = np.random.randint(1000)
        variant = ab_testing.assign_variant(config.name, user_id)
        
        # Simulate prediction and metrics
        prediction = np.random.normal(4, 0.5)
        actual = prediction + np.random.normal(0, 0.2)
        metrics = {
            'prediction_error': abs(prediction - actual),
            'latency_ms': np.random.uniform(10, 50),
            'user_satisfaction': np.random.uniform(3.5, 5)
        }
        
        # Log result
        ab_testing.log_result(
            experiment_name=config.name,
            variant=variant,
            user_id=user_id,
            prediction=prediction,
            actual=actual,
            metrics=metrics
        )
    
    # Get initial analysis
    analysis = ab_testing.analyze_experiment(config.name)
    logger.info("\nInitial A/B Test Analysis:")
    logger.info(f"Sample sizes: {analysis['sample_sizes']}")
    for metric in config.metrics:
        if metric in analysis:
            logger.info(f"\n{metric} results:")
            logger.info(f"Mean A: {analysis[metric]['mean_A']:.4f}")
            logger.info(f"Mean B: {analysis[metric]['mean_B']:.4f}")
            logger.info(f"P-value: {analysis[metric]['p_value']:.4f}")
            logger.info(f"Significant: {analysis[metric]['significant']}")
            logger.info(f"Improvement: {analysis[metric]['improvement']:.2f}%")

if __name__ == "__main__":
    setup_ab_test()
