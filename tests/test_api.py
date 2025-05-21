from fastapi.testclient import TestClient
from api.app import app
import unittest
import json

client = TestClient(app)

class TestAPI(unittest.TestCase):
    def test_health_check(self):
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "healthy"})
    
    def test_get_recommendations(self):
        # Test getting recommendations for a user
        response = client.get("/recommendations/1")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("recommendations", data)
        self.assertIsInstance(data["recommendations"], list)
    
    def test_model_metrics(self):
        # Test getting model metrics
        response = client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("metrics", data)
    
    def test_ab_test_assignment(self):
        # Test A/B test variant assignment
        response = client.get("/experiment/assign/1")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("variant", data)
    
    def test_invalid_user(self):
        # Test error handling for invalid user
        response = client.get("/recommendations/-1")
        self.assertEqual(response.status_code, 400)
    
    def test_model_training(self):
        # Test model training endpoint
        response = client.post("/train", json={
            "model_type": "hybrid",
            "parameters": {
                "epochs": 10,
                "batch_size": 32
            }
        })
        self.assertEqual(response.status_code, 202)
        data = response.json()
        self.assertIn("job_id", data)

if __name__ == '__main__':
    unittest.main()
