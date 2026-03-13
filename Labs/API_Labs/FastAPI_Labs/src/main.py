"""
Breast Cancer Classification API

Endpoints:
- GET  /health - Detailed health check
- POST /predict - Single prediction with confidence
- POST /predict/batch - Batch predictions
- GET  /model/info - Model metadata
- GET  /model/metrics - Training metrics
"""

from fastapi import FastAPI, status, HTTPException
from fastapi.responses import JSONResponse
import logging
import json
from datetime import datetime
from typing import List

from data import (
    BreastCancerData, 
    PredictionResponse, 
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfo,
    HealthResponse
)
from predict import predict_single, predict_batch, get_model_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Breast Cancer Classification API",
    description="ML model serving for breast cancer diagnosis (malignant vs benign)",
    version="2.0.0",
    contact={
        "name": "Akhilesh Kasturi",
        "email": "kasturi.a@northeastern.edu"
    }
)


@app.on_event("startup")
async def startup_event():
    """Log startup information"""
    logger.info("Starting Breast Cancer Classification API")
    try:
        info = get_model_info()
        logger.info(f"Model loaded: {info['model_type']}")
        logger.info(f"Model accuracy: {info['accuracy']:.4f}")
    except Exception as e:
        logger.error(f"Error loading model metadata: {e}")


@app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check():
    """
    Detailed health check endpoint
    Returns model status and metadata
    """
    try:
        info = get_model_info()
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_type=info['model_type'],
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            timestamp=datetime.now().isoformat()
        )


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict(features: BreastCancerData):
    """
    Make a single prediction
    
    Args:
        features: Breast cancer features (30 dimensions)
        
    Returns:
        Prediction with confidence score
    """
    try:
        logger.info("Received prediction request")
        
        # Convert to array
        feature_array = features.to_array()
        
        # Make prediction
        result = predict_single(feature_array)
        
        logger.info(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.4f})")
        
        return PredictionResponse(**result)
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, status_code=status.HTTP_200_OK)
async def predict_batch_endpoint(request: BatchPredictionRequest):
    """
    Make batch predictions
    
    Args:
        request: List of feature arrays
        
    Returns:
        List of predictions with confidence scores
    """
    try:
        logger.info(f"Received batch prediction request: {len(request.samples)} samples")
        
        # Make batch predictions
        results = predict_batch(request.samples)
        
        predictions = [PredictionResponse(**r) for r in results]
        
        logger.info(f"Batch prediction complete: {len(predictions)} predictions")
        
        return BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions)
        )
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/model/info", response_model=ModelInfo, status_code=status.HTTP_200_OK)
async def model_info():
    """
    Get model metadata and information
    
    Returns:
        Model type, training date, metrics, dataset info
    """
    try:
        info = get_model_info()
        return ModelInfo(**info)
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model metadata not found. Train model first.")
    
    except Exception as e:
        logger.error(f"Error retrieving model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/metrics", status_code=status.HTTP_200_OK)
async def model_metrics():
    """
    Get detailed model training metrics including comparison
    
    Returns:
        Complete metrics including model comparison results
    """
    try:
        with open('../model/model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        return {
            "metrics": metadata['metrics'],
            "model_comparison": metadata['model_comparison'],
            "selected_model": metadata['model_type']
        }
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model metrics not found. Train model first.")
    
    except Exception as e:
        logger.error(f"Error retrieving metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", status_code=status.HTTP_200_OK)
async def root():
    """
    Root endpoint - API information
    """
    return {
        "message": "Breast Cancer Classification API",
        "version": "2.0.0",
        "endpoints": {
            "/health": "Health check",
            "/predict": "Single prediction",
            "/predict/batch": "Batch predictions",
            "/model/info": "Model information",
            "/model/metrics": "Training metrics",
            "/docs": "API documentation"
        },
        "author": "Akhilesh Kasturi"
    }