# Breast Cancer Classification with Model Comparison

## Overview

This lab containerizes a machine learning pipeline using Docker. The application trains and compares multiple classification models (Random Forest and Gradient Boosting) on the Breast Cancer Wisconsin dataset, automatically selecting the best performer based on evaluation metrics.

**Dataset**: Breast Cancer Wisconsin
**Models**: Random Forest vs Gradient Boosting  
**Output**: Best model + evaluation metrics + sample predictions

## Improvements Made

### 1. Dataset
**Original**: Iris dataset (150 samples, 4 features, 3 classes)  
**Improved**: Breast Cancer dataset (569 samples, 30 features, binary classification)
- More complex problem (30 features vs 4)
- Healthcare domain application
- Binary classification (malignant vs benign)

### 2. Model Comparison
**Original**: Single Random Forest model  
**Improved**: 
- Trains both Random Forest and Gradient Boosting
- Compares models using accuracy, F1 score, and ROC AUC
- Automatically selects best performing model
- Saves comparison results in metrics

### 3. Enhanced Evaluation Metrics
**Original**: No evaluation  
**Improved**: 
- Accuracy score
- F1 score (weighted)
- ROC AUC score (for binary classification)
- Classification report (precision, recall per class)
- Confusion matrix
- Model comparison table
- All metrics saved to JSON file

### 2. Data Validation
**Original**: No validation  
**Improved**:
- Checks for empty datasets
- Validates feature-label alignment
- Detects NaN and infinite values
- Logs data statistics

### 3. Structured Logging
**Original**: Single print statement  
**Improved**:
- Detailed logging at each pipeline stage
- INFO/WARNING/ERROR log levels
- Timestamps for all operations
- Model comparison results logged

### 4. Error Handling
**Original**: No error handling  
**Improved**:
- Try-catch blocks throughout
- Graceful failure with informative messages
- Proper exit codes (0=success, 1=failure)

### 5. Prediction Functionality
**Original**: No predictions  
**Improved**:
- Makes sample predictions on test set
- Saves predictions with actual labels
- Shows correct/incorrect predictions

### 6. Better Dockerfile
**Original**: Single-stage build  
**Improved**:
- Multi-stage build (smaller final image)
- Non-root user for security
- Better layer caching
- Health check included
- Metadata labels

### 7. .dockerignore
**New**: Reduces image size by excluding unnecessary files

## How to Run

### Prerequisites

```bash
# Ensure Docker is installed and running
docker --version

# Should show: Docker version 20.x or higher
```

### Step 1: Build the Docker Image

```bash
# Navigate to lab directory
cd Docker_Labs/Lab1

# Build the image
docker build -t breast-cancer-classifier:improved .

# Verify image created
docker images | grep breast-cancer-classifier
```

### Step 2: Run the Container

```bash
# Run the training pipeline
docker run --name cancer-training breast-cancer-classifier:improved

# View logs in real-time
docker logs -f cancer-training
```

**Note**: Before building Docker, test locally:
```bash
python tests/verify_locally.py
```

### Step 3: Extract Results

```bash
# Copy model file from container (name depends on which model won)
docker cp iris-training:/app/breast_cancer_gradientboosting.pkl ./
# OR
docker cp iris-training:/app/breast_cancer_randomforest.pkl ./

# Copy metrics file
docker cp iris-training:/app/model_metrics.json ./

# Copy predictions
docker cp iris-training:/app/sample_predictions.json ./
```

### Step 4: View Results

```bash
# View model metrics
cat model_metrics.json

# View sample predictions
cat sample_predictions.json
```

### Test the Container

```bash
# Run tests (if you add them)
docker run breast-cancer-classifier:improved python -m pytest tests/ -v

# Check health
docker inspect breast-cancer-classifier:improved | grep -i health
```

### Inspect Container

```bash
# View container filesystem
docker run -it breast-cancer-classifier:improved /bin/bash

# Inside container:
ls -la
cat model_metrics.json
exit
```

## Docker Commands Reference

### Build
```bash
docker build -t breast-cancer-classifier:improved .
```

### Run
```bash
# Basic run
docker run breast-cancer-classifier:improved

# Run with name
docker run --name cancer-training breast-cancer-classifier:improved

# Run with volume mount
docker run -v $(pwd)/output:/app breast-cancer-classifier:improved
```

### Manage
```bash
# List containers
docker ps -a

# View logs
docker logs cancer-training

## Troubleshooting

### Issue: "Cannot connect to Docker daemon"
```bash
# Solution: Start Docker Desktop
# Or start Docker service:
sudo systemctl start docker  # Linux
```

### Issue: "Permission denied" when running container
```bash
# Solution: Add user to docker group
sudo usermod -aG docker $USER
# Then logout and login again
```

### Issue: Build fails with "pip: command not found"
```bash
# Solution: Use correct Python base image
# Ensure dockerfile starts with: FROM python:3.10-slim
```

### Issue: "Module not found" when running
```bash
# Solution: Check requirements.txt is copied and installed
# Verify in dockerfile:
# COPY requirements.txt .
# RUN pip install -r requirements.txt
```