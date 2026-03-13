"""
Data loading and Pydantic models
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from pydantic import BaseModel, Field, validator
from typing import List, Optional


def load_data():
    """
    Load the Breast Cancer dataset and return features and target values.
    Returns:
        X (numpy.ndarray): Features (30 dimensions)
        y (numpy.ndarray): Target values (0=malignant, 1=benign)
    """
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target
    return X, y


def split_data(X, y):
    """
    Split data into training and testing sets.
    Args:
        X (numpy.ndarray): Features
        y (numpy.ndarray): Target values
    Returns:
        X_train, X_test, y_train, y_test (tuple): Split dataset
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


# Pydantic Models for API

class BreastCancerData(BaseModel):
    """
    Request model for breast cancer prediction
    30 features from the dataset
    """
    mean_radius: float = Field(..., gt=0, description="Mean radius of cell nuclei")
    mean_texture: float = Field(..., gt=0, description="Mean texture")
    mean_perimeter: float = Field(..., gt=0, description="Mean perimeter")
    mean_area: float = Field(..., gt=0, description="Mean area")
    mean_smoothness: float = Field(..., ge=0, le=1, description="Mean smoothness")
    mean_compactness: float = Field(..., ge=0, description="Mean compactness")
    mean_concavity: float = Field(..., ge=0, description="Mean concavity")
    mean_concave_points: float = Field(..., ge=0, description="Mean concave points")
    mean_symmetry: float = Field(..., ge=0, le=1, description="Mean symmetry")
    mean_fractal_dimension: float = Field(..., ge=0, le=1, description="Mean fractal dimension")
    
    radius_error: float = Field(..., ge=0, description="Radius error")
    texture_error: float = Field(..., ge=0, description="Texture error")
    perimeter_error: float = Field(..., ge=0, description="Perimeter error")
    area_error: float = Field(..., ge=0, description="Area error")
    smoothness_error: float = Field(..., ge=0, description="Smoothness error")
    compactness_error: float = Field(..., ge=0, description="Compactness error")
    concavity_error: float = Field(..., ge=0, description="Concavity error")
    concave_points_error: float = Field(..., ge=0, description="Concave points error")
    symmetry_error: float = Field(..., ge=0, description="Symmetry error")
    fractal_dimension_error: float = Field(..., ge=0, description="Fractal dimension error")
    
    worst_radius: float = Field(..., gt=0, description="Worst radius")
    worst_texture: float = Field(..., gt=0, description="Worst texture")
    worst_perimeter: float = Field(..., gt=0, description="Worst perimeter")
    worst_area: float = Field(..., gt=0, description="Worst area")
    worst_smoothness: float = Field(..., ge=0, le=1, description="Worst smoothness")
    worst_compactness: float = Field(..., ge=0, description="Worst compactness")
    worst_concavity: float = Field(..., ge=0, description="Worst concavity")
    worst_concave_points: float = Field(..., ge=0, description="Worst concave points")
    worst_symmetry: float = Field(..., ge=0, le=1, description="Worst symmetry")
    worst_fractal_dimension: float = Field(..., ge=0, le=1, description="Worst fractal dimension")

    def to_array(self):
        """Convert to numpy array in correct order"""
        return np.array([[
            self.mean_radius, self.mean_texture, self.mean_perimeter, self.mean_area,
            self.mean_smoothness, self.mean_compactness, self.mean_concavity,
            self.mean_concave_points, self.mean_symmetry, self.mean_fractal_dimension,
            self.radius_error, self.texture_error, self.perimeter_error, self.area_error,
            self.smoothness_error, self.compactness_error, self.concavity_error,
            self.concave_points_error, self.symmetry_error, self.fractal_dimension_error,
            self.worst_radius, self.worst_texture, self.worst_perimeter, self.worst_area,
            self.worst_smoothness, self.worst_compactness, self.worst_concavity,
            self.worst_concave_points, self.worst_symmetry, self.worst_fractal_dimension
        ]])


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: int = Field(..., description="0=malignant, 1=benign")
    prediction_label: str = Field(..., description="Human-readable label")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    model_name: str = Field(..., description="Model used for prediction")


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    samples: List[List[float]] = Field(..., min_items=1, max_items=100, description="List of feature arrays")

    @validator('samples')
    def validate_sample_dimensions(cls, v):
        for sample in v:
            if len(sample) != 30:
                raise ValueError(f"Each sample must have exactly 30 features, got {len(sample)}")
        return v


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictionResponse]
    count: int


class ModelInfo(BaseModel):
    """Model metadata information"""
    model_type: str
    model_class: str
    training_date: str
    accuracy: float
    f1_score: float
    roc_auc: float
    dataset: str
    features: int
    classes: int


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_type: Optional[str] = None
    timestamp: str