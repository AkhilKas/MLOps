# Breast Cancer Classification with Experiment Tracking

## Overview

This lab demonstrates MLflow experiment tracking, model comparison, hyperparameter tuning, and model registry usage with a Breast Cancer classification problem.

**Dataset**: Breast Cancer Wisconsin (569 samples, 30 features, binary classification)  
**Models**: RandomForest, GradientBoosting, XGBoost  
**MLflow Features**: Tracking, Model Registry, Artifact Logging, Hyperparameter Tuning

## Improvements Made

### 1. Different Dataset
**Original**: Wine Quality and Diabetes datasets  
**Improved**: Breast Cancer dataset (569 samples, 30 features)
- Consistent with Docker and FastAPI labs
- Healthcare domain application
- Binary classification problem

### 2. Multiple Model Comparison
**Original**: Single model per script  
**Improved**: 
- Trains RandomForest, GradientBoosting in parallel
- Logs all experiments to MLflow
- Compares models side-by-side in UI
- Creates comparison visualization

### 3. Hyperparameter Tuning with Tracking
**Original**: Fixed hyperparameters  
**Improved**:
- Grid search across parameter space
- Logs all parameter combinations as nested runs
- Tracks CV scores for each configuration
- Selects and logs best parameters

### 4. Model Registry Integration
**Original**: Models logged but not registered  
**Improved**:
- Registers best model in Model Registry
- Versions models automatically
- Transitions model to Production stage
- Enables model lifecycle management

### 5. Artifact Logging
**Original**: Only model artifacts  
**Improved**:
- Confusion matrix heatmaps
- Model comparison bar charts
- Classification reports as text
- Feature importance (if applicable)

### 6. Experiment Organization
**Original**: Default experiment  
**Improved**:
- Named experiment: "breast_cancer_classification"
- Tags for filtering (model_type, stage, author)
- Nested runs for hyperparameter tuning
- Clear run naming convention

## How to Run

### Step 1: Install Dependencies

```bash
cd MLflow_Labs/Lab_1

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### Step 2: Run the Improved Script

```bash
python cancer_mlflow.py
```

### Step 3: View Results in MLflow UI

```bash
# Start MLflow UI
mlflow ui --port 5001
```

## Troubleshooting

### MLflow UI won't start
```bash
# Port 5000 might be used by macOS AirPlay
# Use port 5001 instead
mlflow ui --port 5001
```

### "No module named mlflow"
```bash
pip install mlflow
```

### Model Registry not showing
```bash
# Check tracking URI is not file-based
# Use SQLite backend:
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
python cancer_mlflow.py
```

### Plots not appearing
```bash
# Ensure matplotlib backend is set
pip install matplotlib seaborn
```

## Commands Reference

```bash
# Run improved script
python cancer_mlflow.py

# Start MLflow UI
mlflow ui --port 5001

# View specific run
mlflow runs describe --run-id <run_id>

# List experiments
mlflow experiments list

# Search runs
mlflow runs list --experiment-id 0

# Serve model from registry
mlflow models serve -m "models:/BreastCancerClassifier/Production" -p 5002
```