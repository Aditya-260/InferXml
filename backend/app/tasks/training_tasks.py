"""
Training Tasks
Background tasks for model training using Celery
"""
import os
import tempfile
import traceback
from datetime import datetime
from typing import Dict, Any

import pandas as pd
import joblib

from app.celery_app import celery_app
from app import db
from app.models.experiment import Experiment, TrainingJob
from app.models.dataset import Dataset
from app.services.minio_service import get_minio_service
from app.services.data_profiler import DataProfiler
from app.services.problem_detector import ProblemDetector


@celery_app.task(bind=True, name='training.train_model')
def train_model_task(self, experiment_id: int) -> Dict[str, Any]:
    """
    Main training task - orchestrates the full training pipeline
    
    Args:
        experiment_id: ID of the experiment to train
        
    Returns:
        Training result dictionary
    """
    from app import create_app
    app = create_app()
    
    with app.app_context():
        experiment = Experiment.query.get(experiment_id)
        if not experiment:
            return {'status': 'error', 'message': 'Experiment not found'}
        
        try:
            # Update status
            experiment.status = 'training'
            db.session.commit()
            
            # Log GPU availability for this training job
            try:
                from ml_engine.utils.gpu_utils import log_gpu_info
                gpu_info = log_gpu_info()
            except Exception:
                gpu_info = None
                print("[Training] GPU detection skipped — ml_engine.utils not available", flush=True)
            
            # Get dataset
            dataset = Dataset.query.get(experiment.dataset_id)
            if not dataset:
                raise ValueError('Dataset not found')
            
            # Download dataset from MinIO
            minio = get_minio_service()
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
                tmp_path = tmp.name
            
            minio.download_file(minio.BUCKET_DATASETS, dataset.file_path, tmp_path)
            
            # Load data
            df = pd.read_csv(tmp_path)
            os.unlink(tmp_path)  # Clean up
            
            # Run training pipeline
            result = run_training_pipeline(
                experiment=experiment,
                df=df,
                task=self
            )
            
            # Update experiment with results
            experiment.status = 'completed'
            experiment.completed_at = datetime.utcnow()
            experiment.best_model_name = result.get('best_model')
            experiment.best_score = result.get('best_score')
            experiment.best_model_id = result.get('model_path')
            db.session.commit()
            
            # Send notification email if user opted in
            try:
                from app.models.user import User
                from app.services.notification_service import notify_training_completed
                user = User.query.get(experiment.user_id)
                if user:
                    notify_training_completed(user, experiment.name, result.get('best_model'))
            except Exception as notify_err:
                print(f'[notify] training_completed dispatch error: {notify_err}', flush=True)
            
            return {
                'status': 'success',
                'experiment_id': experiment_id,
                'best_model': result.get('best_model'),
                'best_score': result.get('best_score')
            }
            
        except Exception as e:
            experiment.status = 'failed'
            db.session.commit()
            
            # Send failure notification
            try:
                from app.models.user import User
                from app.services.notification_service import notify_training_failed
                user = User.query.get(experiment.user_id)
                if user:
                    notify_training_failed(user, experiment.name, str(e))
            except Exception as notify_err:
                print(f'[notify] training_failed dispatch error: {notify_err}', flush=True)
            
            return {
                'status': 'error',
                'message': str(e),
                'traceback': traceback.format_exc()
            }


def run_training_pipeline(
    experiment: Experiment,
    df: pd.DataFrame,
    task=None
) -> Dict[str, Any]:
    """
    Execute the training pipeline
    
    Args:
        experiment: Experiment database object
        df: Training DataFrame
        task: Celery task for progress updates
        
    Returns:
        Training result dictionary
    """
    from ml_engine.automl.tabular.classifier import TabularClassifier
    from ml_engine.automl.tabular.regressor import TabularRegressor
    from ml_engine.automl.tabular.clusterer import TabularClusterer
    from ml_engine.preprocessing.tabular_preprocessor import TabularPreprocessor
    from ml_engine.packaging.model_packager import ModelPackager, create_feature_schema
    from ml_engine.utils.visualizer import ModelVisualizer
    from sklearn.model_selection import train_test_split
    
    target_column = experiment.target_column
    problem_type = experiment.problem_type
    
    # Update progress
    if task:
        task.update_state(state='PROGRESS', meta={'step': 'preprocessing', 'progress': 10})
    
    # Preprocessing
    preprocessor = TabularPreprocessor()
    
    if target_column:
        # Industrial Best Practice: Drop rows where target is missing
        if target_column in df.columns:
            df = df.dropna(subset=[target_column])
            
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_processed = preprocessor.fit_transform(X, y)
        y_processed = preprocessor.transform_target(y) if hasattr(preprocessor, 'transform_target') else y
    else:
        X = df
        y = None
        X_processed = preprocessor.fit_transform(X)
        y_processed = None
    
    if task:
        task.update_state(state='PROGRESS', meta={'step': 'training', 'progress': 30})
    
    config = experiment.config or {}
    auto_balance = config.get('auto_balance', False)
    
    if auto_balance and target_column and 'classification' in problem_type:
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_processed, y_processed = smote.fit_resample(X_processed, y_processed)
            print(f"Applied SMOTE. New shape: {X_processed.shape}", flush=True)
        except ImportError:
            print("imbalanced-learn not installed. Skipping SMOTE auto-balance.", flush=True)
            pass
    
    # Select and train model based on problem type
    result = {}
    
    if problem_type in ['binary_classification', 'multiclass_classification', 'classification']:
        trainer = TabularClassifier()
        result = trainer.train(X_processed, y_processed, config=config)
        
    elif problem_type == 'regression':
        trainer = TabularRegressor()
        result = trainer.train(X_processed, y_processed, config=config)
        
    elif problem_type == 'clustering':
        trainer = TabularClusterer()
        result = trainer.train(X_processed)
        
    else:
        # Auto-detect problem type
        detector = ProblemDetector(df, target_column, goal_description=experiment.goal_description or '')
        detection = detector.detect()
        problem_type = detection['problem_type']
        
        if 'classification' in problem_type:
            trainer = TabularClassifier()
            result = trainer.train(X_processed, y_processed, config=config)
        elif 'regression' in problem_type:
            trainer = TabularRegressor()
            result = trainer.train(X_processed, y_processed, config=config)
        else:
            trainer = TabularClusterer()
            result = trainer.train(X_processed)
    
    if task:
        task.update_state(state='PROGRESS', meta={'step': 'packaging', 'progress': 80})
    
    # Recreate the exact test split to generate accurate evaluation graphs
    X_test = None
    y_test = None
    try:
        if 'classification' in problem_type:
            _, X_test, _, y_test = train_test_split(X_processed, y_processed, test_size=0.2, random_state=42, stratify=y_processed)
        elif 'regression' in problem_type:
            _, X_test, _, y_test = train_test_split(X_processed, y_processed, test_size=0.2, random_state=42)
    except Exception as e:
        print(f"Warning: Could not recreate test splits for graphs: {e}")
    
    # Package model
    with tempfile.TemporaryDirectory() as tmp_dir:
        packager = ModelPackager(tmp_dir)
        
        # Generate Graphs
        graphs_dir = os.path.join(tmp_dir, 'graphs')
        if X_test is not None and y_test is not None:
            if 'classification' in problem_type:
                ModelVisualizer.generate_classification_graphs(trainer.best_model, X_test, y_test, graphs_dir)
            elif 'regression' in problem_type:
                ModelVisualizer.generate_regression_graphs(trainer.best_model, X_test, y_test, graphs_dir)
        
        feature_schema = create_feature_schema(df, target_column or '')
        
        metadata = {
            'name': experiment.name,
            'experiment_id': experiment.id,
            'problem_type': problem_type,
            'target_column': target_column,
            'best_model': result.get('best_model'),
            'best_score': result.get('best_score'),
            'training_results': result.get('results', [])
        }
        
        package_dir = packager.package(
            model=trainer.best_model,
            preprocessor=preprocessor,
            feature_schema=feature_schema,
            metadata=metadata,
            graphs_dir=graphs_dir if os.path.exists(graphs_dir) else None
        )
        
        # Upload to MinIO
        minio = get_minio_service()
        model_path = minio.upload_model_package(
            user_id=experiment.user_id,
            experiment_id=experiment.id,
            package_dir=package_dir
        )
    
    if task:
        task.update_state(state='PROGRESS', meta={'step': 'complete', 'progress': 100})
    
    return {
        'best_model': result.get('best_model'),
        'best_score': result.get('best_score'),
        'model_path': model_path,
        'results': result.get('results', [])
    }


@celery_app.task(bind=True, name='training.profile_dataset')
def profile_dataset_task(self, dataset_id: int) -> Dict[str, Any]:
    """
    Profile a dataset in the background
    
    Args:
        dataset_id: ID of the dataset to profile
        
    Returns:
        Profiling result
    """
    from app import create_app
    app = create_app()
    
    with app.app_context():
        dataset = Dataset.query.get(dataset_id)
        if not dataset:
            return {'status': 'error', 'message': 'Dataset not found'}
        
        try:
            dataset.profile_status = 'processing'
            db.session.commit()
            
            # Download dataset from MinIO
            minio = get_minio_service()
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
                tmp_path = tmp.name
            
            minio.download_file(minio.BUCKET_DATASETS, dataset.file_path, tmp_path)
            
            # Load and profile
            df = pd.read_csv(tmp_path)
            os.unlink(tmp_path)
            
            profiler = DataProfiler(df)
            profile = profiler.profile_dataset()
            
            # Detect data type
            data_type = profiler.detect_data_type()
            
            # Update dataset
            dataset.profile_status = 'completed'
            dataset.profile_data = profile
            dataset.data_type = data_type
            dataset.num_rows = len(df)
            dataset.num_columns = len(df.columns)
            dataset.column_info = profile.get('column_profiles', {})
            db.session.commit()
            
            return {
                'status': 'success',
                'dataset_id': dataset_id,
                'data_type': data_type,
                'num_rows': len(df),
                'num_columns': len(df.columns)
            }
            
        except Exception as e:
            dataset.profile_status = 'failed'
            db.session.commit()
            
            return {
                'status': 'error',
                'message': str(e),
                'traceback': traceback.format_exc()
            }
