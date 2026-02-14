import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from kneed import KneeLocator
import pickle
import os
import base64
import logging
import json
from typing import Tuple, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_data_schema(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Validate that DataFrame has required columns and proper data types.
    
    Args:
        df: Input DataFrame
        required_columns: List of required column names
        
    Returns:
        bool: True if validation passes
        
    Raises:
        ValueError: If validation fails
    """
    # Check required columns exist
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for empty dataframe
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate rows")
    
    logger.info(f"Schema validation passed. Shape: {df.shape}")
    return True


def detect_outliers(df: pd.DataFrame, columns: list, threshold: float = 3.0) -> pd.DataFrame:
    """
    Detect and log outliers using Z-score method.
    
    Args:
        df: Input DataFrame
        columns: Columns to check for outliers
        threshold: Z-score threshold for outlier detection
        
    Returns:
        DataFrame with outlier statistics
    """
    outlier_stats = {}
    
    for col in columns:
        if col in df.columns:
            z_scores = (df[col] - df[col].mean()) / df[col].std()
            outliers = (abs(z_scores) > threshold).sum()
            outlier_stats[col] = {
                'outliers': int(outliers),
                'percentage': round(outliers / len(df) * 100, 2)
            }
    
    logger.info(f"Outlier detection: {json.dumps(outlier_stats, indent=2)}")
    return pd.DataFrame(outlier_stats).T


def load_data() -> str:
    """
    Loads data from a CSV file with validation, serializes it, and returns the serialized data.
    
    Returns:
        str: Base64-encoded serialized data (JSON-safe).
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If data validation fails
    """
    try:
        file_path = os.path.join(os.path.dirname(__file__), "../data/file.csv")
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        
        # Validate schema
        required_columns = ["BALANCE", "PURCHASES", "CREDIT_LIMIT"]
        validate_data_schema(df, required_columns)
        
        # Check data quality
        null_counts = df[required_columns].isnull().sum()
        logger.info(f"Null values per column: {null_counts.to_dict()}")
        
        # Detect outliers
        detect_outliers(df, required_columns)
        
        # Serialize
        serialized_data = pickle.dumps(df)
        encoded_data = base64.b64encode(serialized_data).decode("ascii")
        
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return encoded_data
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def data_preprocessing(data_b64: str) -> str:
    """
    Deserializes base64-encoded pickled data, performs preprocessing,
    and returns base64-encoded pickled clustered data.
    
    Args:
        data_b64: Base64-encoded serialized DataFrame
        
    Returns:
        str: Base64-encoded preprocessed data
        
    Raises:
        ValueError: If preprocessing fails
    """
    try:
        # Decode data
        data_bytes = base64.b64decode(data_b64)
        df = pickle.loads(data_bytes)
        
        logger.info(f"Starting preprocessing. Initial shape: {df.shape}")
        
        # Drop missing values
        initial_rows = len(df)
        df = df.dropna()
        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0:
            logger.warning(f"Dropped {dropped_rows} rows with missing values")
        
        # Select clustering features
        clustering_data = df[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]]
        
        # Check for invalid values (negative balances, etc.)
        if (clustering_data < 0).any().any():
            logger.warning("Found negative values in data")
        
        # Scale data
        min_max_scaler = MinMaxScaler()
        clustering_data_minmax = min_max_scaler.fit_transform(clustering_data)
        
        # Log scaling statistics
        logger.info(f"Data scaled. Min: {clustering_data_minmax.min():.4f}, Max: {clustering_data_minmax.max():.4f}")
        
        # Serialize
        clustering_serialized_data = pickle.dumps(clustering_data_minmax)
        encoded_data = base64.b64encode(clustering_serialized_data).decode("ascii")
        
        logger.info(f"Preprocessing complete. Final shape: {clustering_data_minmax.shape}")
        return encoded_data
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise


def find_optimal_clusters(data, max_k: int = 20) -> Tuple[int, Dict]:
    """
    Find optimal number of clusters using elbow method and silhouette score.
    
    Args:
        data: Preprocessed data for clustering
        max_k: Maximum number of clusters to test
        
    Returns:
        Tuple of (optimal_k, metrics_dict)
    """
    kmeans_kwargs = {
        "init": "k-means++",  # Better initialization than random
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42
    }
    
    sse = []
    silhouette_scores = []
    davies_bouldin_scores = []
    
    logger.info(f"Testing K from 2 to {max_k}")
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        labels = kmeans.fit_predict(data)
        
        sse.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, labels))
        davies_bouldin_scores.append(davies_bouldin_score(data, labels))
    
    # Find elbow point
    kl = KneeLocator(
        range(2, max_k + 1), 
        sse, 
        curve="convex", 
        direction="decreasing"
    )
    optimal_k = kl.elbow if kl.elbow else 5  # Default to 5 if no elbow found
    
    # Get index for optimal k metrics
    optimal_idx = optimal_k - 2
    
    metrics = {
        'optimal_k': int(optimal_k),
        'sse_at_optimal_k': float(sse[optimal_idx]),
        'silhouette_score': float(silhouette_scores[optimal_idx]),
        'davies_bouldin_score': float(davies_bouldin_scores[optimal_idx]),
        'all_sse': sse,
        'all_silhouette_scores': silhouette_scores,
        'all_davies_bouldin_scores': davies_bouldin_scores
    }
    
    logger.info(f"Optimal K={optimal_k} (Silhouette: {metrics['silhouette_score']:.4f})")
    
    return optimal_k, metrics


def build_save_model(data_b64: str, filename: str) -> Dict:
    """
    Builds an optimized KMeans model and saves it with metadata.
    
    Args:
        data_b64: Base64-encoded preprocessed data
        filename: Output filename for model
        
    Returns:
        Dict: Model metrics and metadata (JSON-safe)
        
    Raises:
        Exception: If model training fails
    """
    try:
        # Decode data
        data_bytes = base64.b64decode(data_b64)
        data = pickle.loads(data_bytes)
        
        logger.info(f"Training model on data shape: {data.shape}")
        
        # Find optimal K
        optimal_k, metrics = find_optimal_clusters(data, max_k=20)
        
        # Train final model with optimal K
        kmeans_kwargs = {
            "init": "k-means++",
            "n_init": 10,
            "max_iter": 300,
            "random_state": 42
        }
        
        final_model = KMeans(n_clusters=optimal_k, **kmeans_kwargs)
        final_model.fit(data)
        
        # Calculate final metrics
        final_labels = final_model.predict(data)
        final_silhouette = silhouette_score(data, final_labels)
        final_db_score = davies_bouldin_score(data, final_labels)
        
        # Get cluster sizes
        unique, counts = pd.Series(final_labels).value_counts().sort_index().values, pd.Series(final_labels).value_counts().sort_index().index
        cluster_distribution = {f"cluster_{i}": int(count) for i, count in zip(unique, counts)}
        
        # Save model
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        
        with open(output_path, "wb") as f:
            pickle.dump(final_model, f)
        
        logger.info(f"Model saved to {output_path}")
        
        # Save metrics
        final_metrics = {
            'optimal_k': int(optimal_k),
            'final_silhouette_score': float(final_silhouette),
            'final_davies_bouldin_score': float(final_db_score),
            'inertia': float(final_model.inertia_),
            'n_iter': int(final_model.n_iter_),
            'cluster_distribution': cluster_distribution,
            'data_shape': list(data.shape),
            'all_sse': metrics['all_sse']
        }
        
        # Save metrics to JSON
        metrics_path = output_path.replace('.sav', '_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        logger.info(f"Model metrics saved to {metrics_path}")
        logger.info(f"Final model: K={optimal_k}, Silhouette={final_silhouette:.4f}")
        
        return final_metrics
        
    except Exception as e:
        logger.error(f"Error building model: {str(e)}")
        raise


def load_model_predict(filename: str, metrics: Dict) -> Dict:
    """
    Loads the saved model and makes predictions on test data with validation.
    
    Args:
        filename: Model filename
        metrics: Training metrics from previous task
        
    Returns:
        Dict: Prediction results and validation (JSON-safe)
        
    Raises:
        FileNotFoundError: If model or test file not found
        ValueError: If prediction validation fails
    """
    try:
        # Load model
        model_path = os.path.join(os.path.dirname(__file__), "../model", filename)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"Loading model from {model_path}")
        loaded_model = pickle.load(open(model_path, "rb"))
        
        # Load test data
        test_path = os.path.join(os.path.dirname(__file__), "../data/test.csv")
        
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test file not found: {test_path}")
        
        df_test = pd.read_csv(test_path)
        
        # Validate test data
        required_columns = ["BALANCE", "PURCHASES", "CREDIT_LIMIT"]
        validate_data_schema(df_test, required_columns)
        
        # Make predictions
        predictions = loaded_model.predict(df_test)
        
        # Validate predictions
        n_clusters = metrics.get('optimal_k', loaded_model.n_clusters)
        invalid_predictions = (predictions < 0) | (predictions >= n_clusters)
        
        if invalid_predictions.any():
            raise ValueError(f"Invalid cluster predictions detected")
        
        # Prepare results
        results = {
            'predictions': [int(p) for p in predictions],
            'n_samples': len(predictions),
            'cluster_counts': {f"cluster_{i}": int((predictions == i).sum()) 
                             for i in range(n_clusters)},
            'model_clusters': int(n_clusters),
            'model_inertia': float(metrics.get('inertia', 0))
        }
        
        logger.info(f"Predictions made: {results['predictions']}")
        logger.info(f"Cluster distribution: {results['cluster_counts']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise