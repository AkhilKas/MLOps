"""
Prediction logic with confidence scores
"""

import joblib
import json
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Model cache
_model = None
_metadata = None


def load_model():
    """
    Load model and metadata (cached)
    """
    global _model, _metadata
    
    if _model is None:
        model_path = Path(__file__).parent.parent / "model" / "breast_cancer_model.pkl"
        _model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
    
    if _metadata is None:
        metadata_path = Path(__file__).parent.parent / "model" / "model_metadata.json"
        with open(metadata_path, 'r') as f:
            _metadata = json.load(f)
        logger.info("Model metadata loaded")
    
    return _model, _metadata


def predict_single(X):
    """
    Predict class label for single sample with confidence
    
    Args:
        X (numpy.ndarray): Input features (shape: 1x30)
        
    Returns:
        dict: Prediction, label, confidence, model name
    """
    try:
        model, metadata = load_model()
        
        # Validate input shape
        if X.shape[1] != 30:
            raise ValueError(f"Expected 30 features, got {X.shape[1]}")
        
        # Make prediction
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        confidence = float(max(probabilities))
        
        # Map prediction to label
        label = "benign" if prediction == 1 else "malignant"
        
        return {
            'prediction': int(prediction),
            'prediction_label': label,
            'confidence': confidence,
            'model_name': metadata['model_type']
        }
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


def predict_batch(samples):
    """
    Predict class labels for multiple samples
    
    Args:
        samples (list): List of feature arrays
        
    Returns:
        list: List of prediction dictionaries
    """
    try:
        model, metadata = load_model()
        
        # Convert to numpy array
        X = np.array(samples)
        
        # Validate shape
        if X.shape[1] != 30:
            raise ValueError(f"Each sample must have 30 features, got {X.shape[1]}")
        
        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            confidence = float(max(prob))
            label = "benign" if pred == 1 else "malignant"
            
            results.append({
                'prediction': int(pred),
                'prediction_label': label,
                'confidence': confidence,
                'model_name': metadata['model_type']
            })
        
        return results
    
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise


def get_model_info():
    """
    Get model metadata information
    
    Returns:
        dict: Model metadata
    """
    try:
        _, metadata = load_model()
        
        return {
            'model_type': metadata['model_type'],
            'model_class': metadata['model_class'],
            'training_date': metadata['training_date'],
            'accuracy': metadata['metrics']['accuracy'],
            'f1_score': metadata['metrics']['f1_score'],
            'roc_auc': metadata['metrics']['roc_auc'],
            'dataset': metadata['dataset'],
            'features': metadata['features'],
            'classes': metadata['classes']
        }
    
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise