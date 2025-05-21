import unittest
import numpy as np
import pandas as pd
from ml_service.model_versioning import ModelVersion
from ml_service.model_monitoring import ModelMonitor, PredictionLog
from ml_service.ab_testing import ABTesting, ExperimentConfig
from datetime import datetime, timedelta
import os
import tempfile
import shutil

class TestModelVersioning(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.model_version = ModelVersion(self.test_dir)
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_save_version(self):
        # Test saving a model version
        metrics = {'mse': 0.5, 'mae': 0.3}
        version_id = self.model_version.save_version(
            'test_model',
            'dummy_path.joblib',
            metrics,
            metadata={'test': True}
        )
        self.assertIsNotNone(version_id)
        
        # Test retrieving version
        versions = self.model_version.list_versions('test_model')
        self.assertIn(version_id, versions)
        
        # Test version info
        version_info = versions[version_id]
        self.assertEqual(version_info['metrics'], metrics)
        self.assertEqual(version_info['metadata']['test'], True)

class TestModelMonitoring(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.model_monitor = ModelMonitor(self.test_dir)
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_log_prediction(self):
        # Test logging a prediction
        log = PredictionLog(
            timestamp=datetime.now().isoformat(),
            model_name='test_model',
            model_version='v1',
            input_features={'feature1': 1},
            prediction=0.5,
            actual=0.6,
            latency_ms=10.0
        )
        self.model_monitor.log_prediction(log)
        
        # Test calculating metrics
        metrics = self.model_monitor.calculate_metrics()
        self.assertIn('test_model', metrics)
        self.assertIn('performance', metrics['test_model'])
        self.assertIn('accuracy', metrics['test_model'])

class TestABTesting(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.ab_testing = ABTesting(self.test_dir)
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_experiment_creation(self):
        # Test creating an experiment
        config = ExperimentConfig(
            name='test_experiment',
            variant_a='model_v1',
            variant_b='model_v2',
            traffic_split=0.5,
            metrics=['accuracy'],
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=7),
            min_sample_size=100
        )
        experiment = self.ab_testing.create_experiment(config)
        self.assertEqual(experiment['name'], config.name)
        
        # Test variant assignment
        variant = self.ab_testing.assign_variant(config.name, user_id=123)
        self.assertIn(variant, [config.variant_a, config.variant_b])

if __name__ == '__main__':
    unittest.main()
