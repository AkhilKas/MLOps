# Iris Classification

## Overview

This lab containerizes a machine learning model training pipeline using Docker. The application trains a Random Forest classifier on the Iris dataset and generates comprehensive evaluation metrics.

**Dataset**: Iris (150 samples, 4 features, 3 classes)  
**Model**: Random Forest Classifier  
**Output**: Trained model + evaluation metrics + sample predictions

## Improvements Made

### 1. Model Evaluation Metrics
**Original**: Only saves model, no evaluation  
**Improved**: 
- Calculates accuracy, F1 score
- Generates classification report (precision, recall per class)
- Creates confusion matrix
- Saves all metrics to JSON file

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
- Training progress tracking

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
docker build -t iris-classifier:improved .

# Verify image created
docker images | grep iris-classifier
```

**Expected output**:
```
iris-classifier   improved   abc123def456   10 seconds ago   200MB
```

### Step 2: Run the Container

```bash
# Run the training pipeline
docker run --name iris-training iris-classifier:improved

# View logs in real-time
docker logs -f iris-training
```

**Note**: Before building Docker, test locally:
```bash
python tests/verify_locally.py
```

### Step 3: Extract Results

```bash
# Copy model file from container
docker cp iris-training:/app/iris_model.pkl ./

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

## Verification

### Test the Container

```bash
# Run tests (if you add them)
docker run iris-classifier:improved python -m pytest tests/ -v

# Check health
docker inspect iris-classifier:improved | grep -i health
```

### Inspect Container

```bash
# View container filesystem
docker run -it iris-classifier:improved /bin/bash

# Inside container:
ls -la
cat model_metrics.json
exit
```
---

## Docker Commands Reference

### Build
```bash
docker build -t iris-classifier:improved .
```

### Run
```bash
# Basic run
docker run iris-classifier:improved
```

### Manage
```bash
# List containers
docker ps -a

# View logs
docker logs iris-training

# Remove container
docker rm iris-training

# Remove image
docker rmi iris-classifier:improved
```

### Extract Files
```bash
# Copy files from container
docker cp iris-training:/app/iris_model.pkl ./
docker cp iris-training:/app/model_metrics.json ./
docker cp iris-training:/app/sample_predictions.json ./
```

---

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