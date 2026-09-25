"""
Experiment and Training Job Models
"""
from datetime import datetime
from app import db


class Experiment(db.Model):
    """Experiment/Project model"""
    
    __tablename__ = 'experiments'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    
    # Problem definition
    problem_type = db.Column(db.String(50))  # classification, regression, clustering, timeseries, image_classification
    target_column = db.Column(db.String(255))
    goal_description = db.Column(db.Text)  # Natural language goal
    
    # Configuration
    config = db.Column(db.JSON)  # Training configuration
    
    # Status
    status = db.Column(db.String(20), default='created')  # created, training, completed, failed
    
    # Best model info
    best_model_id = db.Column(db.String(255))  # MinIO path to best model
    best_score = db.Column(db.Float)
    best_model_name = db.Column(db.String(100))
    
    # Training results (includes model_package_path)
    results = db.Column(db.JSON, default=dict)
    
    # Live thinking/status logs for real-time UI updates
    thinking_logs = db.Column(db.Text, default='')
    
    # Rich explanation data for human-friendly UI
    explanation_data = db.Column(db.JSON, default=dict)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = db.Column(db.DateTime)
    
    # Foreign keys
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=False)
    
    # Relationships
    training_jobs = db.relationship('TrainingJob', backref='experiment', lazy='dynamic')
    
    def to_dict(self):
        """Serialize to dictionary"""
        results = self.results or {}
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'problem_type': self.problem_type,
            'target_column': self.target_column,
            'goal_description': self.goal_description,
            'status': self.status,
            'best_model_name': self.best_model_name,
            'best_score': self.best_score,
            'dataset_id': self.dataset_id,
            'thinking_logs': self.thinking_logs or '',
            'explanation_data': self.explanation_data or {},
            'created_at': self.created_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'results': results,
            'has_package': bool(results.get('model_package_path'))
        }
    
    def __repr__(self):
        return f'<Experiment {self.name}>'


class TrainingJob(db.Model):
    """Individual model training job"""
    
    __tablename__ = 'training_jobs'
    
    id = db.Column(db.Integer, primary_key=True)
    
    # Model info
    model_name = db.Column(db.String(100), nullable=False)  # e.g., RandomForest, XGBoost
    model_params = db.Column(db.JSON)  # Hyperparameters
    
    # Status
    status = db.Column(db.String(20), default='pending')  # pending, running, completed, failed
    progress = db.Column(db.Float, default=0.0)  # 0-100
    
    # Results
    metrics = db.Column(db.JSON)  # accuracy, f1, rmse, etc.
    model_path = db.Column(db.String(512))  # MinIO path
    
    # Logs
    logs = db.Column(db.Text)
    error_message = db.Column(db.Text)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    started_at = db.Column(db.DateTime)
    completed_at = db.Column(db.DateTime)
    
    # Foreign keys
    experiment_id = db.Column(db.Integer, db.ForeignKey('experiments.id'), nullable=False)
    
    def to_dict(self):
        """Serialize to dictionary"""
        return {
            'id': self.id,
            'model_name': self.model_name,
            'model_params': self.model_params,
            'status': self.status,
            'progress': self.progress,
            'metrics': self.metrics,
            'created_at': self.created_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }
    
    def __repr__(self):
        return f'<TrainingJob {self.model_name}>'
