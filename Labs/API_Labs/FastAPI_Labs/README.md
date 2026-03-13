# FastAPI Lab 1 - Breast Cancer Classification API

## Overview

This lab creates a REST API for breast cancer classification using FastAPI. The API serves a machine learning model that predicts whether a tumor is malignant or benign based on 30 cellular features.

**Dataset**: Breast Cancer Wisconsin (569 samples, 30 features, binary classification)  
**Models**: Random Forest vs Gradient Boosting (best model selected automatically)  
**Framework**: FastAPI with uvicorn server

---

## Improvements Made

### 1. Model Comparison
**Original**: Single Decision Tree model  
**Improved**: 
- Trains Random Forest and Gradient Boosting
- Compares models using accuracy, F1, ROC AUC
- Automatically serves best performing model

### 2. Enhanced Endpoints
**Original**: `/` and `/predict` only  
**Improved**: 
- `GET /health` - Detailed health check with model status
- `POST /predict` - Single prediction with confidence score
- `POST /predict/batch` - Batch predictions (up to 100 samples)
- `GET /model/info` - Model metadata and training info
- `GET /model/metrics` - Detailed metrics including comparison
- `GET /` - API information and endpoint list

### 3. Enhanced Error Handling
**Original**: Basic try-catch  
**Improved**:
- Input validation with Pydantic
- Proper HTTP status codes (400, 404, 500)
- Detailed error messages
- Logging for debugging

### 4. Model Caching
**Original**: Loads model on every request  
**Improved**:
- Caches model in memory after first load
- Faster response times
- Better resource utilization

### 5. Testing
**Original**: No tests  
**Improved**:
- pytest with TestClient
- Tests for all endpoints
- Validation testing
- Error case testing

## How to Run

### Step 1: Install Dependencies

```bash
cd API_Labs/FastAPI_Labs

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### Step 2: Train the Model

```bash
cd src
python train.py
```

**Generated files**:
- `model/breast_cancer_model.pkl`
- `model/model_metadata.json`

### Step 3: Start the API Server

```bash
# From src/ directory
uvicorn main:app --reload

# Or specify host and port:
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Server starts at**: http://localhost:8000

### Step 4: Test the API

**Option A: Interactive Documentation** (Recommended)

Go to http://localhost:8000/docs

You'll see Swagger UI with all endpoints. Click any endpoint to test it.

**Option B: Using curl**

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'

# Model info
curl http://localhost:8000/model/info

# Model metrics
curl http://localhost:8000/model/metrics
```

**Option C: Using Python requests**

```python
import requests

# Single prediction
url = "http://localhost:8000/predict"
data = {
    "mean_radius": 17.99,
    "mean_texture": 10.38,
    # ... (all 30 features)
}
response = requests.post(url, json=data)
print(response.json())

# Output:
# {
#   "prediction": 0,
#   "prediction_label": "malignant",
#   "confidence": 0.95,
#   "model_name": "GradientBoosting"
# }
```

---

## Running Tests

```bash
# Run all tests
pytest tests/test_api.py -v
```

## API Endpoints

### GET /health
Health check with model status

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "GradientBoosting",
  "timestamp": "2026-02-27T19:00:00.123456"
}
```

### POST /predict
Single prediction with confidence

**Request**:
```json
{
  "mean_radius": 17.99,
  "mean_texture": 10.38,
}
```

**Response**:
```json
{
  "prediction": 0,
  "prediction_label": "malignant",
  "confidence": 0.95,
  "model_name": "GradientBoosting"
}
```

### POST /predict/batch
Batch predictions (1-100 samples)

**Request**:
```json
{
  "samples": [
    [17.99, 10.38, 122.8, ...],  // 30 features
    [13.54, 14.36, 87.46, ...]   // 30 features
  ]
}
```

**Response**:
```json
{
  "predictions": [
    {
      "prediction": 0,
      "prediction_label": "malignant",
      "confidence": 0.95,
      "model_name": "GradientBoosting"
    },
    {
      "prediction": 1,
      "prediction_label": "benign",
      "confidence": 0.88,
      "model_name": "GradientBoosting"
    }
  ],
  "count": 2
}
```

### GET /model/info
Model metadata

**Response**:
```json
{
  "model_type": "GradientBoosting",
  "model_class": "GradientBoostingClassifier",
  "training_date": "2026-02-27T18:54:46.123456",
  "accuracy": 0.9737,
  "f1_score": 0.9736,
  "roc_auc": 0.9956,
  "dataset": "Breast Cancer Wisconsin",
  "features": 30,
  "classes": 2
}
```

### GET /model/metrics
Detailed training metrics with model comparison

**Response**:
```json
{
  "metrics": {
    "accuracy": 0.9737,
    "f1_score": 0.9736,
    "roc_auc": 0.9956
  },
  "model_comparison": {
    "RandomForest": {
      "accuracy": 0.9649,
      "f1_score": 0.9647,
      "roc_auc": 0.9932
    },
    "GradientBoosting": {
      "accuracy": 0.9737,
      "f1_score": 0.9736,
      "roc_auc": 0.9956
    }
  },
  "selected_model": "GradientBoosting"
}
```

## Troubleshooting

### Model not found error
```bash
# Train the model first
cd src
python train.py
```

### Port already in use
```bash
# Use different port
uvicorn main:app --port 8001
```

### Import errors
```bash
# Ensure you're in src/ directory
cd src
uvicorn main:app --reload
```