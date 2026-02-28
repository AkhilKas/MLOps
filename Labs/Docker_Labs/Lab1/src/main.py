"""
Breast Cancer Classification Model Training with Model Comparison

Improvements:
- Multiple model comparison (RandomForest vs GradientBoosting)
- Model evaluation with multiple metrics
- Data validation
- Structured logging
- Error handling
- Metrics persistence to JSON
- Prediction functionality
"""

import logging
import json
from datetime import datetime
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    f1_score,
    roc_auc_score
)
import joblib
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_data(X, y):
    """Validate input data quality"""
    try:
        assert X.shape[0] > 0, "Dataset is empty"
        assert X.shape[0] == len(y), "Feature and label counts don't match"
        assert not np.isnan(X).any(), "Dataset contains NaN values"
        assert not np.isinf(X).any(), "Dataset contains infinite values"
        
        logger.info(f"Data validation passed: {X.shape[0]} samples, {X.shape[1]} features")
        return True
    except AssertionError as e:
        logger.error(f"Data validation failed: {e}")
        raise


def train_model(X_train, y_train):
    """Train Random Forest model with logging"""
    try:
        logger.info("Training Random Forest model...")
        
        model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            max_depth=10,
            min_samples_split=5
        )
        
        model.fit(X_train, y_train)
        
        logger.info("Model trained successfully")
        logger.info(f"   - Number of trees: {model.n_estimators}")
        logger.info(f"   - Max depth: {model.max_depth}")
        
        return model
    
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise


def train_gradient_boosting(X_train, y_train):
    """Train Gradient Boosting model with logging"""
    try:
        logger.info("Training Gradient Boosting model...")
        
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        logger.info("Model trained successfully")
        logger.info(f"   - Number of estimators: {model.n_estimators}")
        logger.info(f"   - Learning rate: {model.learning_rate}")
        
        return model
    
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise


def compare_models(models_dict, X_test, y_test):
    """Compare multiple models and return best one"""
    try:
        logger.info("Comparing models...")
        
        comparison_results = {}
        
        for model_name, model in models_dict.items():
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            roc_auc = roc_auc_score(y_test, y_prob)
            
            comparison_results[model_name] = {
                'accuracy': float(accuracy),
                'f1_score': float(f1),
                'roc_auc': float(roc_auc)
            }
            
            logger.info(f"   {model_name}:")
            logger.info(f"      - Accuracy: {accuracy:.4f}")
            logger.info(f"      - F1 Score: {f1:.4f}")
            logger.info(f"      - ROC AUC: {roc_auc:.4f}")
        
        # Select best model based on F1 score
        best_model_name = max(comparison_results, key=lambda x: comparison_results[x]['f1_score'])
        best_model = models_dict[best_model_name]
        
        logger.info(f"\nBest model: {best_model_name}")
        
        return best_model, best_model_name, comparison_results
    
    except Exception as e:
        logger.error(f"Model comparison failed: {e}")
        raise


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate model and return metrics"""
    try:
        logger.info(f"Evaluating {model_name} on test set...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, y_prob)
        
        # Get classification report as dict
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        logger.info(f"Model evaluation complete:")
        logger.info(f"   - Accuracy: {accuracy:.4f}")
        logger.info(f"   - F1 Score: {f1:.4f}")
        logger.info(f"   - ROC AUC: {roc_auc:.4f}")
        logger.info(f"\nConfusion Matrix:\n{cm}")
        
        metrics = {
            'model_name': model_name,
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'test_samples': int(len(y_test)),
            'timestamp': datetime.now().isoformat()
        }
        
        return metrics
    
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        raise


def save_model_and_metrics(model, metrics, model_path='model.pkl', metrics_path='model_metrics.json'):
    """Save model and metrics to files"""
    try:
        # Save model
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save metrics
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Saving failed: {e}")
        raise


def make_sample_predictions(model, X_test, y_test, n_samples=5):
    """Make and display sample predictions"""
    try:
        logger.info(f"Making {n_samples} sample predictions...")
        
        # Get random samples
        indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
        
        predictions = []
        for idx in indices:
            pred = model.predict([X_test[idx]])[0]
            actual = y_test[idx]
            correct = "CORRECT" if pred == actual else "INCORRECT"
            
            predictions.append({
                'features': X_test[idx].tolist(),
                'predicted': int(pred),
                'actual': int(actual),
                'correct': bool(pred == actual)
            })
            
            logger.info(f"   {correct}: Predicted: {pred}, Actual: {actual}")
        
        return predictions
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


def main():
    """Main pipeline execution"""
    try:
        logger.info("=" * 60)
        logger.info("BREAST CANCER CLASSIFICATION MODEL TRAINING PIPELINE")
        logger.info("Author: Akhilesh Kasturi")
        logger.info("=" * 60)
        
        # Load dataset
        logger.info("Loading Breast Cancer dataset...")
        cancer = load_breast_cancer()
        X, y = cancer.data, cancer.target
        logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
        logger.info(f"Classes: 0=malignant, 1=benign")
        
        # Validate data
        validate_data(X, y)
        
        # Split data
        logger.info("Splitting data (80% train, 20% test)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        
        # Train multiple models
        logger.info("\nTraining multiple models for comparison...")
        rf_model = train_model(X_train, y_train)
        gb_model = train_gradient_boosting(X_train, y_train)
        
        # Compare models
        models_dict = {
            'RandomForest': rf_model,
            'GradientBoosting': gb_model
        }
        
        best_model, best_model_name, comparison_results = compare_models(
            models_dict, X_test, y_test
        )
        
        # Evaluate best model in detail
        metrics = evaluate_model(best_model, X_test, y_test, model_name=best_model_name)
        
        # Add comparison results to metrics
        metrics['model_comparison'] = comparison_results
        
        # Check quality threshold
        if metrics['accuracy'] < 0.90:
            logger.warning(f"Model accuracy ({metrics['accuracy']:.4f}) below threshold (0.90)")
        else:
            logger.info(f"Model meets quality threshold (accuracy > 0.90)")
        
        # Save best model and metrics
        save_model_and_metrics(best_model, metrics, 
                              model_path=f'breast_cancer_{best_model_name.lower()}.pkl',
                              metrics_path='model_metrics.json')
        
        # Make sample predictions
        predictions = make_sample_predictions(best_model, X_test, y_test, n_samples=5)
        
        # Save predictions
        with open('sample_predictions.json', 'w') as f:
            json.dump(predictions, f, indent=2)
        logger.info("Sample predictions saved to sample_predictions.json")
        
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"   - Best Model: {best_model_name}")
        logger.info(f"   - Model Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"   - F1 Score: {metrics['f1_score']:.4f}")
        logger.info(f"   - ROC AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"   - Files created: breast_cancer_{best_model_name.lower()}.pkl, model_metrics.json, sample_predictions.json")
        logger.info("=" * 60)
        
        return 0
    
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"PIPELINE FAILED: {e}")
        logger.error("=" * 60)
        return 1


if __name__ == '__main__':
    exit(main())