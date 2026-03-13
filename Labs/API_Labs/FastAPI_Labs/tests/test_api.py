"""
API endpoint tests
Run from project root: pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add src to path - navigate from tests/ to src/
current_dir = Path(__file__).parent
src_dir = current_dir.parent / 'src'
sys.path.insert(0, str(src_dir))

from main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint returns API info"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "version" in response.json()


def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "timestamp" in data


def test_predict_endpoint_valid():
    """Test prediction with valid data"""
    sample_data = {
        "mean_radius": 17.99,
        "mean_texture": 10.38,
        "mean_perimeter": 122.8,
        "mean_area": 1001.0,
        "mean_smoothness": 0.1184,
        "mean_compactness": 0.2776,
        "mean_concavity": 0.3001,
        "mean_concave_points": 0.1471,
        "mean_symmetry": 0.2419,
        "mean_fractal_dimension": 0.07871,
        "radius_error": 1.095,
        "texture_error": 0.9053,
        "perimeter_error": 8.589,
        "area_error": 153.4,
        "smoothness_error": 0.006399,
        "compactness_error": 0.04904,
        "concavity_error": 0.05373,
        "concave_points_error": 0.01587,
        "symmetry_error": 0.03003,
        "fractal_dimension_error": 0.006193,
        "worst_radius": 25.38,
        "worst_texture": 17.33,
        "worst_perimeter": 184.6,
        "worst_area": 2019.0,
        "worst_smoothness": 0.1622,
        "worst_compactness": 0.6656,
        "worst_concavity": 0.7119,
        "worst_concave_points": 0.2654,
        "worst_symmetry": 0.4601,
        "worst_fractal_dimension": 0.1189
    }
    
    response = client.post("/predict", json=sample_data)
    assert response.status_code == 200
    data = response.json()
    
    assert "prediction" in data
    assert "prediction_label" in data
    assert "confidence" in data
    assert "model_name" in data
    assert data['prediction'] in [0, 1]
    assert 0 <= data['confidence'] <= 1


def test_predict_endpoint_invalid_missing_field():
    """Test prediction with missing required field"""
    invalid_data = {
        "mean_radius": 17.99,
        # Missing other required fields
    }
    
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422  # Validation error


def test_predict_endpoint_invalid_negative_value():
    """Test prediction with invalid negative value"""
    invalid_data = {
        "mean_radius": -17.99,  # Should be > 0
        "mean_texture": 10.38,
        # ... other fields
    }
    
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422


def test_batch_predict_endpoint():
    """Test batch prediction endpoint"""
    # Create sample with 30 features each
    sample1 = [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 
               0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 
               0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 
               184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
    
    sample2 = [13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 
               0.1885, 0.05766, 0.2699, 0.7886, 2.058, 23.56, 0.008462, 
               0.0146, 0.02387, 0.01315, 0.0198, 0.0023, 15.11, 19.26, 
               99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259]
    
    request_data = {
        "samples": [sample1, sample2]
    }
    
    response = client.post("/predict/batch", json=request_data)
    assert response.status_code == 200
    data = response.json()
    
    assert "predictions" in data
    assert "count" in data
    assert data['count'] == 2
    assert len(data['predictions']) == 2


def test_batch_predict_invalid_dimensions():
    """Test batch prediction with wrong number of features"""
    invalid_samples = {
        "samples": [[1.0, 2.0, 3.0]]  # Only 3 features instead of 30
    }
    
    response = client.post("/predict/batch", json=invalid_samples)
    assert response.status_code == 422  # Validation error


def test_model_info_endpoint():
    """Test model info endpoint"""
    response = client.get("/model/info")
    
    if response.status_code == 200:
        data = response.json()
        assert "model_type" in data
        assert "accuracy" in data
        assert "training_date" in data
    else:
        # Model not trained yet
        assert response.status_code == 404


def test_model_metrics_endpoint():
    """Test model metrics endpoint"""
    response = client.get("/model/metrics")
    
    if response.status_code == 200:
        data = response.json()
        assert "metrics" in data
        assert "model_comparison" in data
        assert "selected_model" in data
    else:
        # Model not trained yet
        assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])