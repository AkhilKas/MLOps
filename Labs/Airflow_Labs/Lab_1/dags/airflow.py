"""
This DAG implements an enhanced K-Means clustering pipeline with:
- Data validation and quality checks
- Hyperparameter optimization (optimal K selection)
- Comprehensive error handling and logging
- Model evaluation metrics
- Testing and validation
"""

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.standard.operators.bash import BashOperator
from datetime import datetime, timedelta
import logging

from src.lab import (
    load_data, 
    data_preprocessing, 
    build_save_model, 
    load_model_predict
)

# Configure logging
logger = logging.getLogger(__name__)

# Define default arguments
default_args = {
    'owner': 'akhilesh_shah',
    'start_date': datetime(2025, 2, 1),
    'retries': 2,  # Increased retries for robustness
    'retry_delay': timedelta(minutes=5),
    'retry_exponential_backoff': True,  # Exponential backoff
    'max_retry_delay': timedelta(minutes=30),
    'email_on_failure': False,  # Set to True in production
    'email_on_retry': False,
}

# Create DAG
with DAG(
    'improved_customer_clustering_pipeline',
    default_args=default_args,
    description='Enhanced K-Means clustering with validation and optimization',
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    tags=['mlops', 'clustering', 'credit-card', 'akhilesh-shah'],
    doc_md=__doc__,
) as dag:

    # Task 1: Load and Validate Data
    load_data_task = PythonOperator(
        task_id='load_and_validate_data',
        python_callable=load_data,
        doc_md="""
        ## Load and Validate Data
        
        **Purpose**: Load credit card customer data with validation
        
        **Validations**:
        - Schema validation (required columns present)
        - Missing value detection
        - Outlier detection using Z-scores
        - Duplicate row detection
        
        **Output**: Base64-encoded serialized DataFrame
        """,
    )

    # Task 2: Data Preprocessing
    data_preprocessing_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=data_preprocessing,
        op_args=[load_data_task.output],
        doc_md="""
        ## Data Preprocessing
        
        **Purpose**: Clean and normalize data for clustering
        
        **Steps**:
        - Drop rows with missing values
        - Select clustering features (BALANCE, PURCHASES, CREDIT_LIMIT)
        - Detect invalid values (negatives)
        - MinMax scaling to [0, 1]
        
        **Output**: Base64-encoded scaled data
        """,
    )

    # Task 3: Build and Save Optimized Model
    build_model_task = PythonOperator(
        task_id='build_optimized_model',
        python_callable=build_save_model,
        op_args=[data_preprocessing_task.output, "customer_segments.sav"],
        doc_md="""
        ## Build Optimized Model
        
        **Purpose**: Train K-Means with optimal cluster selection
        
        **Optimization**:
        - Test K from 2 to 20
        - Use elbow method to find optimal K
        - Evaluate with silhouette score and Davies-Bouldin index
        - Use k-means++ initialization
        
        **Metrics Saved**:
        - Optimal K
        - Silhouette score
        - Davies-Bouldin score
        - Cluster distribution
        - SSE curve
        
        **Output**: Model file and metrics JSON
        """,
    )

    # Task 4: Validate Model Quality
    def validate_model_quality(**context):
        """
        Check if model meets quality thresholds.
        
        Quality criteria:
        - Silhouette score > 0.3 (reasonable clustering)
        - Davies-Bouldin score < 2.0 (good separation)
        - No empty clusters
        """
        metrics = context['task_instance'].xcom_pull(task_ids='build_optimized_model')
        
        silhouette = metrics['final_silhouette_score']
        db_score = metrics['final_davies_bouldin_score']
        cluster_dist = metrics['cluster_distribution']
        
        # Check thresholds
        quality_checks = {
            'silhouette_score_ok': silhouette > 0.3,
            'davies_bouldin_ok': db_score < 2.0,
            'no_empty_clusters': min(cluster_dist.values()) > 0,
        }
        
        all_passed = all(quality_checks.values())
        
        logger.info(f"Quality checks: {quality_checks}")
        logger.info(f"Silhouette: {silhouette:.4f}, DB Score: {db_score:.4f}")
        
        if all_passed:
            logger.info("✅ Model quality checks PASSED")
            return 'model_approved'
        else:
            logger.warning("⚠️ Model quality checks FAILED")
            return 'model_rejected'

    quality_check_task = BranchPythonOperator(
        task_id='validate_model_quality',
        python_callable=validate_model_quality,
        doc_md="""
        ## Validate Model Quality
        
        **Purpose**: Ensure model meets quality standards
        
        **Checks**:
        - Silhouette score > 0.3
        - Davies-Bouldin score < 2.0
        - No empty clusters
        
        **Branches**:
        - Pass → model_approved
        - Fail → model_rejected
        """,
    )

    # Task 5a: Model Approved - Make Predictions
    predict_task = PythonOperator(
        task_id='model_approved',
        python_callable=load_model_predict,
        op_args=["customer_segments.sav", build_model_task.output],
        doc_md="""
        ## Make Predictions (Model Approved)
        
        **Purpose**: Use validated model for predictions
        
        **Steps**:
        - Load saved model
        - Validate test data schema
        - Make predictions
        - Validate prediction outputs
        
        **Output**: Prediction results with cluster distribution
        """,
    )

    # Task 5b: Model Rejected - Log Failure
    reject_task = BashOperator(
        task_id='model_rejected',
        bash_command='echo "⚠️ Model rejected due to quality issues. Check logs for details."',
        doc_md="""
        ## Model Rejected
        
        **Purpose**: Log model quality failure
        
        **Next Steps**:
        - Review silhouette and DB scores
        - Check cluster distribution
        - Consider different K values or preprocessing
        """,
    )

    # Task 6: Generate Report
    def generate_pipeline_report(**context):
        """Generate summary report of entire pipeline execution."""
        import json
        
        # Pull all task outputs
        ti = context['task_instance']
        metrics = ti.xcom_pull(task_ids='build_optimized_model')
        
        # Check if prediction ran
        try:
            predictions = ti.xcom_pull(task_ids='model_approved')
            prediction_status = "Completed"
        except:
            predictions = None
            prediction_status = "Skipped (Model Rejected)"
        
        report = {
            'pipeline': 'Customer Clustering',
            'execution_date': context['execution_date'].isoformat(),
            'model_metrics': {
                'optimal_k': metrics['optimal_k'],
                'silhouette_score': metrics['final_silhouette_score'],
                'davies_bouldin_score': metrics['final_davies_bouldin_score'],
                'cluster_distribution': metrics['cluster_distribution'],
            },
            'prediction_status': prediction_status,
            'predictions': predictions if predictions else 'N/A'
        }
        
        logger.info("=" * 60)
        logger.info("PIPELINE EXECUTION REPORT")
        logger.info("=" * 60)
        logger.info(json.dumps(report, indent=2))
        logger.info("=" * 60)
        
        return report

    report_task = PythonOperator(
        task_id='generate_pipeline_report',
        python_callable=generate_pipeline_report,
        trigger_rule='none_failed',  # Run even if predict_task was skipped
        doc_md="""
        ## Generate Pipeline Report
        
        **Purpose**: Summarize entire pipeline execution
        
        **Includes**:
        - Model metrics
        - Quality check results
        - Prediction status
        - Execution metadata
        """,
    )

    # Define task dependencies
    load_data_task >> data_preprocessing_task >> build_model_task >> quality_check_task
    quality_check_task >> [predict_task, reject_task]
    [predict_task, reject_task] >> report_task


# Test DAG structure
if __name__ == "__main__":
    dag.test()