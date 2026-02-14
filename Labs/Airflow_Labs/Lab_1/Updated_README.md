# Airflow Lab 1 - Credit Card Customer Clustering
---

## Overview

This lab implements a K-Means clustering pipeline using Apache Airflow to segment credit card customers based on their balance, purchases, and credit limit.

**Dataset**: Credit card customers  
**Algorithm**: K-Means clustering  
**Pipeline**: Load Data → Validate → Preprocess → Train Model → Predict

---

## Improvements Made

### 1. Data Validation
- Added schema validation to check required columns exist
- Detect and log null values
- Identify outliers using Z-score method
- Check for duplicate rows

### 2. Optimal K Selection
**Original**: Used fixed K=49 (arbitrary)  
**Improved**: Automatically find optimal K (2-20) using:
- Elbow method
- Silhouette score
- Davies-Bouldin index
- k-means++ initialization

### 3. Model Evaluation Metrics
**Original**: Only SSE  
**Improved**: 
- Silhouette score (measures cluster quality)
- Davies-Bouldin score (measures cluster separation)
- Cluster size distribution
- Model convergence tracking

### 4. Quality Gates
Added automated quality checks with branching:
- Silhouette score must be > 0.3
- Davies-Bouldin score must be < 2.0
- All clusters must be populated
- Pipeline branches to "approved" or "rejected" based on results

### 6. Unit Tests
- Tests for validation, preprocessing, training, predictions
- Edge case handling

---

## How to Run

### Prerequisites

```bash
# Required software
Python 3.8+
pip
```

### Setup (First Time Only)

```bash
# 1. Navigate to lab directory
cd Labs/Airflow_Labs/Lab_1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variable
export AIRFLOW__CORE__ENABLE_XCOM_PICKLING=True

# 4. Initialize Airflow database
airflow db migrate

# 5. Create admin user
python -c "
from airflow import settings
from airflow.models import User
session = settings.Session()
user = User(username='admin', email=<email>,
            first_name='<f-name>', last_name='<l-name>',
            role='Admin', password='admin123')
session.add(user)
session.commit()
print('Admin user created!')
"
```

### Run the Pipeline

**Option A: Using Airflow UI**

```bash
# Terminal 1: Start scheduler
cd Labs/Airflow_Labs/Lab_1
export AIRFLOW__CORE__ENABLE_XCOM_PICKLING=True
airflow scheduler

# Terminal 2: Start webserver
cd Labs/Airflow_Labs/Lab_1
export AIRFLOW__CORE__ENABLE_XCOM_PICKLING=True
airflow webserver -p 8080

# Then:
# 1. Go to http://localhost:8080
# 2. Login (admin/admin123)
# 3. Find "improved_customer_clustering_pipeline"
# 4. Toggle it ON
# 5. Click the play button to run
```

**Option B: Direct Python Execution (Quick)**

```bash
cd Labs/Airflow_Labs/Lab_1/dags

# Run pipeline without Airflow UI
python -c "
import sys
sys.path.insert(0, 'src')
from lab import load_data, data_preprocessing, build_save_model, load_model_predict

data = load_data()
processed = data_preprocessing(data)
metrics = build_save_model(processed, 'customer_segments.sav')
results = load_model_predict('customer_segments.sav', metrics)

print(f'Optimal K: {metrics[\"optimal_k\"]}')
print(f'Silhouette Score: {metrics[\"final_silhouette_score\"]:.4f}')
print(f'Predictions: {results[\"predictions\"]}')
"
```

---

## Testing

Run unit tests to verify everything works:

```bash
cd Labs/Airflow_Labs/Lab_1/dags
pytest tests/test_lab.py -v
```

Expected: All 15+ tests should pass

---

## Expected Output

After running the pipeline, you should see:

### Model Metrics (in `model/customer_segments_metrics.json`):
```json
{
  "optimal_k": 5,
  "final_silhouette_score": 0.42,
  "final_davies_bouldin_score": 1.24,
  "cluster_distribution": {
    "cluster_0": 1823,
    "cluster_1": 2145,
    "cluster_2": 1654,
    "cluster_3": 2012,
    "cluster_4": 1316
  }
}
```

### Pipeline Logs:
```
✅ Data loaded successfully. Shape: (8950, 18)
✅ Preprocessing complete. Final shape: (8XXX, 3)
✅ Optimal K=5 (Silhouette: 0.4231)
✅ Model quality checks PASSED
✅ Predictions made: [2]
```

---

## Summary

This lab enhances the original K-Means clustering pipeline with:
- ✅ Data validation and quality checks
- ✅ Automated optimal K selection (instead of fixed K=49)
- ✅ Multiple evaluation metrics (Silhouette, Davies-Bouldin)
- ✅ Quality gates with branching logic
- ✅ Comprehensive error handling and logging
- ✅ Unit tests with pytest
 
**Output**: Trained model + metrics in `model/` directory
