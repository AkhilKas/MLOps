"""
Model Training with Comparison and Metrics
"""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import joblib
import json
from datetime import datetime
from data import load_data, split_data


def train_models(X_train, y_train):
    """
    Train multiple models for comparison
    """
    models = {}
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    models['RandomForest'] = rf
    
    # Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    gb.fit(X_train, y_train)
    models['GradientBoosting'] = gb
    
    return models


def evaluate_models(models, X_test, y_test):
    """
    Evaluate all models and return comparison metrics
    """
    comparison = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        comparison[name] = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'f1_score': float(f1_score(y_test, y_pred, average='weighted')),
            'roc_auc': float(roc_auc_score(y_test, y_prob))
        }
    
    return comparison


def select_best_model(models, comparison):
    """
    Select best model based on F1 score
    """
    best_name = max(comparison, key=lambda x: comparison[x]['f1_score'])
    return models[best_name], best_name


def save_model_and_metrics(model, model_name, metrics, comparison):
    """
    Save model and comprehensive metrics
    """
    # Save model
    joblib.dump(model, "../model/breast_cancer_model.pkl")
    
    # Save metadata
    metadata = {
        'model_type': model_name,
        'model_class': str(type(model).__name__),
        'training_date': datetime.now().isoformat(),
        'metrics': metrics,
        'model_comparison': comparison,
        'dataset': 'Breast Cancer Wisconsin',
        'features': 30,
        'classes': 2
    }
    
    with open('../model/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved: {model_name}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    print("Loading data...")
    X, y = load_data()
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print("Training models...")
    models = train_models(X_train, y_train)
    
    print("Evaluating models...")
    comparison = evaluate_models(models, X_test, y_test)
    
    print("\nModel Comparison:")
    for name, metrics in comparison.items():
        print(f"  {name}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")
    
    print("\nSelecting best model...")
    best_model, best_name = select_best_model(models, comparison)
    
    print("Saving model and metrics...")
    save_model_and_metrics(best_model, best_name, comparison[best_name], comparison)
    
    print("\nTraining complete!")