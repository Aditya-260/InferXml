
import sys
import os
import pandas as pd
import numpy as np

# Add backend directory to path
sys.path.append(os.path.abspath('backend'))

from app import create_app, db
from app.models.experiment import Experiment
from app.routes.training import _run_training_task

def trigger_training():
    app = create_app()
    with app.app_context():
        # Create a dummy experiment
        print("Creating dummy experiment...")
        experiment = Experiment(
            name="Test Model for Streamlit",
            target_column="target",
            problem_type="classification",
            status="training",
            user_id=1,  # Assuming user 1 exists
            dataset_id=1, # Dummy ID
            results={} # Initialize results
        )
        db.session.add(experiment)
        db.session.commit()
        print(f"Created Experiment ID: {experiment.id}")
        
        # Create dummy column info
        column_info = {
            'feature1': {'dtype': 'float64', 'semantic_type': 'numeric'},
            'feature2': {'dtype': 'object', 'semantic_type': 'categorical', 'categories': ['A', 'B']},
            'target': {'dtype': 'object', 'semantic_type': 'categorical'}
        }
        
        # We need to bypass the MinIO download part in _run_training_task or handle it.
        # _run_training_task calls get_minio_service() and download_bytes.
        # If we provide a bogus path, it will hit the exception handler:
        # "⚠️ MinIO download failed... Switching to fallback dummy data..."
        # This is EXACTLY what we want!
        
        file_path = "dummy/path.csv"
        
        print("Starting training task...")
        # This will use fallback dummy data, train a model, and package it using the NEW ModelPackager code
        _run_training_task(experiment.id, file_path, "target", column_info)
        
        # Verify status
        db.session.refresh(experiment)
        print(f"Final Status: {experiment.status}")
        if experiment.status == 'completed':
            print(f"Model generated successfully. ID: {experiment.id}")
            print(f"Results: {experiment.results}")
        else:
            print(f"Training failed: {experiment.error_message}")

if __name__ == "__main__":
    trigger_training()
