# Distributed Training - CIFAR-10 Classification


## Overview

This lab demonstrates distributed training strategies in TensorFlow using CIFAR-10 image classification. It compares different distribution strategies and tracks performance metrics.

**Dataset**: CIFAR-10 (60,000 32x32 RGB images, 10 classes)  
**Model**: Convolutional Neural Network (CNN)  
**Strategies**: No Strategy (baseline), MirroredStrategy, MultiWorkerMirroredStrategy  
**Metrics**: Training time, throughput, accuracy

## Improvements Made

### 1. Different Dataset
**Original**: MNIST (28x28 grayscale, simple)  
**Improved**: CIFAR-10 (32x32 RGB, more complex)
- More realistic image classification problem
- Color images (3 channels vs 1)
- More complex features to learn
- Better demonstrates distributed training benefits

### 2. Strategy Comparison
**Original**: Only MultiWorkerMirroredStrategy  
**Improved**: 
- No Strategy (baseline for comparison)
- MirroredStrategy (single machine simulation)
- Side-by-side performance comparison
- Identifies best strategy for the workload

### 3. Performance Metrics
**Original**: No performance tracking  
**Improved**:
- Training time measurement
- Throughput calculation (samples/sec)
- Memory usage awareness
- Strategy comparison table

### 4. Model Evaluation
**Original**: Training only, no evaluation  
**Improved**:
- Evaluation on test set
- Test accuracy tracked
- Model comparison based on test performance
- Models saved for each strategy

### 5. Production-Ready Logging
**Original**: Minimal logging  
**Improved**:
- Structured logging throughout
- Strategy configuration logged
- Performance metrics logged
- Clear comparison output

## How to Run

### Step 1: Install Dependencies

```bash
cd Labs/Model_Development/Distributed_Training/Lab1

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### Step 2: Run Strategy Comparison

```bash
python main_improved.py
```

## Understanding the Results

### strategy_comparison.json

```json
{
  "experiment": "Distribution Strategy Comparison",
  "dataset": "CIFAR-10",
  "model": "CNN",
  "results": [
    {
      "strategy": "No Strategy",
      "batch_size": 64,
      "num_replicas": 1,
      "epochs": 3,
      "steps_per_epoch": 100,
      "training_time_seconds": 43.25,
      "final_train_loss": 1.6432,
      "final_train_accuracy": 0.4012,
      "test_loss": 1.5234,
      "test_accuracy": 0.4523,
      "throughput_samples_per_sec": 443.35
    },
    {
      "strategy": "MirroredStrategy",
      "batch_size": 64,
      "num_replicas": 1,
      "epochs": 3,
      "steps_per_epoch": 100,
      "training_time_seconds": 42.89,
      "final_train_loss": 1.6234,
      "final_train_accuracy": 0.4087,
      "test_loss": 1.5123,
      "test_accuracy": 0.4567,
      "throughput_samples_per_sec": 446.78
    }
  ],
  "winner": "MirroredStrategy"
}
```

## Running Original Multi-Worker Script

If you want to run the original multi-worker simulation:

```bash
# Terminal 1: Start worker 0
export TF_CONFIG='{"cluster": {"worker": ["localhost:12345", "localhost:23456"]}, "task": {"type": "worker", "index": 0}}'
python main.py &> job_0.log &

# Terminal 2: Start worker 1  
export TF_CONFIG='{"cluster": {"worker": ["localhost:12345", "localhost:23456"]}, "task": {"type": "worker", "index": 1}}'
python main.py &> job_1.log

# Check logs
tail -f job_0.log
tail -f job_1.log
```


## Key Concepts

### Distribution Strategies in TensorFlow

**1. No Strategy (Baseline)**
- Single device training
- No distribution overhead
- Good for small models/datasets

**2. MirroredStrategy**
- Single machine, multiple GPUs
- Synchronous all-reduce for gradients
- Good for: Multi-GPU workstations

**3. MultiWorkerMirroredStrategy** (original lab)
- Multiple machines
- Each worker has copy of model
- Good for: Large-scale training clusters

### When to Use Each

- **No Strategy**: Small models, single GPU sufficient
- **MirroredStrategy**: Multiple GPUs on one machine
- **MultiWorker**: Multiple machines in cluster, very large models


## Troubleshooting

### TensorFlow import errors
```bash
pip install tensorflow==2.15.0
```

### CUDA warnings
```bash
# Expected - we disabled GPU with:
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# This is for simulation purposes
```

### Out of memory
```bash
# Reduce batch size in cifar10.py
batch_size = 32  # Instead of 64
```

### Slow training
```bash
# Reduce epochs or steps_per_epoch in main_improved.py
epochs=2
steps_per_epoch=50
```