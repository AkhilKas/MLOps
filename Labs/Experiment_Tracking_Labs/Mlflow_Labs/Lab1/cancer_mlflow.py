"""
Breast Cancer Classification with MLflow Tracking

Features:
- Multiple model comparison
- Hyperparameter tuning with tracking
- Model registry integration
- Artifact logging (plots)
- Experiment organization
"""

import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# Set experiment
mlflow.set_experiment("breast_cancer_classification")


def load_and_prepare_data():
    """Load and split Breast Cancer dataset"""
    logger.info("Loading Breast Cancer dataset...")
    
    cancer = load_breast_cancer()
    X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y = cancer.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Data split: Train={len(X_train)}, Test={len(X_test)}")
    return X_train, X_test, y_train, y_test, cancer.target_names


def train_baseline_models(X_train, X_test, y_train, y_test):
    """Train baseline models and track with MLflow"""
    logger.info("Training baseline models...")
    
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for model_name, model in models.items():
        with mlflow.start_run(run_name=f"{model_name}_baseline"):
            # Log tags
            mlflow.set_tag("model_type", model_name)
            mlflow.set_tag("stage", "baseline")
            mlflow.set_tag("author", "Akhilesh Kasturi")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_prob)
            
            # Log parameters
            if hasattr(model, 'n_estimators'):
                mlflow.log_param("n_estimators", model.n_estimators)
            if hasattr(model, 'max_depth') and model.max_depth:
                mlflow.log_param("max_depth", model.max_depth)
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc_auc)
            
            # Log confusion matrix as artifact
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'{model_name} Confusion Matrix')
            plt.tight_layout()
            mlflow.log_figure(fig, f"confusion_matrix_{model_name}.png")
            plt.close()
            
            # Log model
            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(model, "model", signature=signature)
            
            results[model_name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'run_id': mlflow.active_run().info.run_id
            }
            
            logger.info(f"{model_name}: Accuracy={accuracy:.4f}, F1={f1:.4f}, ROC AUC={roc_auc:.4f}")
    
    return results


def hyperparameter_tuning(X_train, X_test, y_train, y_test):
    """Perform hyperparameter tuning with MLflow tracking"""
    logger.info("Starting hyperparameter tuning for Random Forest...")
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1', n_jobs=-1)
    
    with mlflow.start_run(run_name="RandomForest_GridSearch"):
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("stage", "hyperparameter_tuning")
        mlflow.set_tag("author", "Akhilesh Kasturi")
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Log all parameter combinations
        cv_results = pd.DataFrame(grid_search.cv_results_)
        for idx, row in cv_results.iterrows():
            with mlflow.start_run(run_name=f"RF_config_{idx}", nested=True):
                # Log parameters
                mlflow.log_params(row['params'])
                # Log CV score
                mlflow.log_metric("mean_cv_score", row['mean_test_score'])
                mlflow.log_metric("std_cv_score", row['std_test_score'])
        
        # Best model evaluation
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        # Log best parameters
        mlflow.log_params(grid_search.best_params_)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("best_cv_score", grid_search.best_score_)
        
        # Log model
        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(
            best_model, 
            "model", 
            signature=signature,
            registered_model_name="BreastCancerClassifier"
        )
        
        logger.info(f"Best params: {grid_search.best_params_}")
        logger.info(f"Best model: Accuracy={accuracy:.4f}, F1={f1:.4f}")
        
        return grid_search.best_estimator_, mlflow.active_run().info.run_id


def register_best_model(model_name, run_id):
    """Register model in MLflow Model Registry and transition to Production"""
    from mlflow.tracking import MlflowClient
    
    client = MlflowClient()
    
    # Get latest version
    latest_versions = client.get_latest_versions(model_name, stages=["None"])
    
    if latest_versions:
        version = latest_versions[0].version
        
        # Transition to Production
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )
        
        logger.info(f"Model {model_name} version {version} transitioned to Production")
        return version
    
    return None


def create_comparison_plot(results):
    """Create model comparison plot and log as artifact"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    models = list(results.keys())
    metrics = ['accuracy', 'f1_score', 'roc_auc']
    metric_labels = ['Accuracy', 'F1 Score', 'ROC AUC']
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [results[m][metric] for m in models]
        axes[idx].bar(models, values, color=['#1f77b4', '#ff7f0e'])
        axes[idx].set_ylabel(label)
        axes[idx].set_title(f'{label} Comparison')
        axes[idx].set_ylim([0.9, 1.0])
        
        # Add value labels on bars
        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.005, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    return fig


def main():
    """Main pipeline with MLflow tracking"""
    logger.info("=" * 60)
    logger.info("BREAST CANCER CLASSIFICATION WITH MLFLOW")
    logger.info("Author: Akhilesh Kasturi")
    logger.info("=" * 60)
    
    # Load data
    X_train, X_test, y_train, y_test, target_names = load_and_prepare_data()
    
    # Train baseline models
    baseline_results = train_baseline_models(X_train, X_test, y_train, y_test)
    
    # Create comparison plot
    with mlflow.start_run(run_name="Model_Comparison"):
        mlflow.set_tag("type", "comparison")
        mlflow.set_tag("author", "Akhilesh Kasturi")
        
        comparison_fig = create_comparison_plot(baseline_results)
        mlflow.log_figure(comparison_fig, "model_comparison.png")
        plt.close()
        
        # Log comparison as text
        comparison_text = "\n".join([
            f"{model}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}, AUC={metrics['roc_auc']:.4f}"
            for model, metrics in baseline_results.items()
        ])
        mlflow.log_text(comparison_text, "model_comparison.txt")
    
    # Hyperparameter tuning
    best_model, best_run_id = hyperparameter_tuning(X_train, X_test, y_train, y_test)
    
    # Register best model
    logger.info("Registering best model in Model Registry...")
    version = register_best_model("BreastCancerClassifier", best_run_id)
    
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("View results: mlflow ui --port 5001")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()