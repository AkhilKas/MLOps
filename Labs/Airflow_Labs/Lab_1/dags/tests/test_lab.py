"""
Unit tests for improved clustering pipeline
Run with: pytest tests/test_lab.py -v
"""

import pytest
import pandas as pd
import numpy as np
import base64
import pickle
import tempfile
import os
from unittest.mock import patch, MagicMock

# Import functions to test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.lab import (
    validate_data_schema,
    detect_outliers,
    load_data,
    data_preprocessing,
    find_optimal_clusters,
    build_save_model,
    load_model_predict
)


# ============================================================================
# Test Data Validation Functions
# ============================================================================

def test_validate_data_schema_success():
    """Test schema validation with valid data"""
    df = pd.DataFrame({
        'BALANCE': [100, 200, 300],
        'PURCHASES': [50, 75, 100],
        'CREDIT_LIMIT': [1000, 2000, 3000]
    })
    
    result = validate_data_schema(df, ['BALANCE', 'PURCHASES', 'CREDIT_LIMIT'])
    assert result == True


def test_validate_data_schema_missing_columns():
    """Test schema validation with missing columns"""
    df = pd.DataFrame({
        'BALANCE': [100, 200, 300],
        'PURCHASES': [50, 75, 100]
    })
    
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_data_schema(df, ['BALANCE', 'PURCHASES', 'CREDIT_LIMIT'])


def test_validate_data_schema_empty_dataframe():
    """Test schema validation with empty DataFrame"""
    df = pd.DataFrame()
    
    with pytest.raises(ValueError, match="DataFrame is empty"):
        validate_data_schema(df, ['BALANCE'])


def test_detect_outliers():
    """Test outlier detection"""
    df = pd.DataFrame({
        'BALANCE': [100, 200, 300, 10000],  # 10000 is outlier
        'PURCHASES': [50, 75, 100, 120]
    })
    
    result = detect_outliers(df, ['BALANCE', 'PURCHASES'], threshold=3.0)
    
    assert 'BALANCE' in result.index
    assert result.loc['BALANCE', 'outliers'] > 0


# ============================================================================
# Test Data Loading and Preprocessing
# ============================================================================

@pytest.fixture
def sample_csv_file():
    """Create a temporary CSV file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        f.write('BALANCE,PURCHASES,CREDIT_LIMIT,OTHER\n')
        f.write('100.5,50.2,1000,A\n')
        f.write('200.3,75.8,2000,B\n')
        f.write('300.1,100.5,3000,C\n')
        yield f.name
    os.unlink(f.name)


def test_data_preprocessing():
    """Test data preprocessing pipeline"""
    # Create sample DataFrame
    df = pd.DataFrame({
        'BALANCE': [100, 200, 300, np.nan],
        'PURCHASES': [50, 75, 100, 125],
        'CREDIT_LIMIT': [1000, 2000, 3000, 4000],
        'OTHER': ['A', 'B', 'C', 'D']
    })
    
    # Serialize
    serialized = pickle.dumps(df)
    encoded = base64.b64encode(serialized).decode("ascii")
    
    # Process
    result_encoded = data_preprocessing(encoded)
    
    # Decode result
    result_bytes = base64.b64decode(result_encoded)
    result_data = pickle.loads(result_bytes)
    
    # Assertions
    assert result_data.shape[0] == 3  # One row dropped due to NaN
    assert result_data.shape[1] == 3  # Three features
    assert result_data.min() >= 0.0  # MinMax scaled
    assert result_data.max() <= 1.0


# ============================================================================
# Test Model Training and Optimization
# ============================================================================

def test_find_optimal_clusters():
    """Test optimal cluster finding"""
    # Create simple clusterable data
    np.random.seed(42)
    data = np.vstack([
        np.random.randn(50, 3) + [0, 0, 0],
        np.random.randn(50, 3) + [5, 5, 5]
    ])
    
    optimal_k, metrics = find_optimal_clusters(data, max_k=10)
    
    # Assertions
    assert 2 <= optimal_k <= 10
    assert 'optimal_k' in metrics
    assert 'silhouette_score' in metrics
    assert 'davies_bouldin_score' in metrics
    assert metrics['silhouette_score'] > 0  # Should have some structure


def test_build_save_model(tmp_path):
    """Test model building and saving"""
    # Create sample data
    np.random.seed(42)
    data = np.random.randn(100, 3)
    
    # Serialize
    serialized = pickle.dumps(data)
    encoded = base64.b64encode(serialized).decode("ascii")
    
    # Build model
    with patch('src.lab.os.path.dirname') as mock_dirname:
        mock_dirname.return_value = str(tmp_path)
        
        metrics = build_save_model(encoded, "test_model.sav")
    
    # Assertions
    assert 'optimal_k' in metrics
    assert 'final_silhouette_score' in metrics
    assert 'cluster_distribution' in metrics
    assert metrics['optimal_k'] >= 2


# ============================================================================
# Test Prediction and Validation
# ============================================================================

def test_load_model_predict(tmp_path):
    """Test model loading and prediction"""
    # Create and save a simple model
    from sklearn.cluster import KMeans
    
    model = KMeans(n_clusters=3, random_state=42)
    test_data = np.random.randn(10, 3)
    model.fit(test_data)
    
    # Save model
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_path = model_dir / "test_model.sav"
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Create test CSV
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    test_csv = data_dir / "test.csv"
    
    pd.DataFrame({
        'BALANCE': [100, 200],
        'PURCHASES': [50, 75],
        'CREDIT_LIMIT': [1000, 2000]
    }).to_csv(test_csv, index=False)
    
    # Mock metrics
    metrics = {
        'optimal_k': 3,
        'inertia': 100.0
    }
    
    # Test prediction
    with patch('src.lab.os.path.dirname') as mock_dirname:
        mock_dirname.return_value = str(tmp_path)
        
        results = load_model_predict("test_model.sav", metrics)
    
    # Assertions
    assert 'predictions' in results
    assert 'n_samples' in results
    assert results['n_samples'] == 2
    assert len(results['predictions']) == 2


# ============================================================================
# Test Error Handling
# ============================================================================

def test_load_data_file_not_found():
    """Test error handling when file doesn't exist"""
    with patch('src.lab.os.path.exists', return_value=False):
        with pytest.raises(FileNotFoundError):
            load_data()


def test_data_preprocessing_invalid_input():
    """Test error handling for invalid input"""
    invalid_input = "not-base64-encoded"
    
    with pytest.raises(Exception):  # Should fail on base64 decode
        data_preprocessing(invalid_input)


# ============================================================================
# Test Edge Cases
# ============================================================================

def test_preprocessing_all_same_values():
    """Test preprocessing with constant values (edge case)"""
    df = pd.DataFrame({
        'BALANCE': [100, 100, 100],
        'PURCHASES': [50, 50, 50],
        'CREDIT_LIMIT': [1000, 1000, 1000]
    })
    
    serialized = pickle.dumps(df)
    encoded = base64.b64encode(serialized).decode("ascii")
    
    # Should handle gracefully (variance = 0 after scaling)
    result_encoded = data_preprocessing(encoded)
    
    result_bytes = base64.b64decode(result_encoded)
    result_data = pickle.loads(result_bytes)
    
    # All values should be same after scaling
    assert result_data.std() < 1e-10  # Essentially zero variance


def test_clustering_with_outliers():
    """Test that clustering handles outliers"""
    np.random.seed(42)
    # Normal data + outliers
    normal_data = np.random.randn(90, 3)
    outliers = np.random.randn(10, 3) * 10
    data = np.vstack([normal_data, outliers])
    
    optimal_k, metrics = find_optimal_clusters(data, max_k=10)
    
    # Should still find clusters despite outliers
    assert optimal_k >= 2
    assert metrics['silhouette_score'] is not None


# ============================================================================
# Integration Test
# ============================================================================

def test_full_pipeline_integration(tmp_path):
    """Integration test for complete pipeline"""
    # 1. Create sample data
    df = pd.DataFrame({
        'BALANCE': np.random.uniform(100, 5000, 100),
        'PURCHASES': np.random.uniform(50, 3000, 100),
        'CREDIT_LIMIT': np.random.uniform(1000, 10000, 100)
    })
    
    # 2. Serialize (simulating load_data output)
    serialized = pickle.dumps(df)
    encoded = base64.b64encode(serialized).decode("ascii")
    
    # 3. Preprocess
    preprocessed = data_preprocessing(encoded)
    
    # 4. Build model
    with patch('src.lab.os.path.dirname') as mock_dirname:
        mock_dirname.return_value = str(tmp_path)
        metrics = build_save_model(preprocessed, "integration_test.sav")
    
    # 5. Verify
    assert metrics['optimal_k'] >= 2
    assert 0 <= metrics['final_silhouette_score'] <= 1
    assert metrics['final_davies_bouldin_score'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])