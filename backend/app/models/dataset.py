"""
Dataset Model
"""
from datetime import datetime
from app import db


class Dataset(db.Model):
    """Dataset metadata model"""
    
    __tablename__ = 'datasets'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    
    # File info
    file_path = db.Column(db.String(512), nullable=False)  # MinIO path
    file_type = db.Column(db.String(20), nullable=False)   # csv, xlsx, image_zip
    file_size = db.Column(db.BigInteger)
    
    # Data info
    data_type = db.Column(db.String(20))  # tabular, timeseries, image
    num_rows = db.Column(db.Integer)
    num_columns = db.Column(db.Integer)
    column_info = db.Column(db.JSON)  # Column names, types, stats
    
    # Profile info
    profile_status = db.Column(db.String(20), default='pending')  # pending, completed, failed
    profile_data = db.Column(db.JSON)  # Detailed profiling results
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Foreign keys
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Relationships
    experiments = db.relationship('Experiment', backref='dataset', lazy='dynamic')
    
    def to_dict(self):
        """Serialize to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'file_type': self.file_type,
            'file_size': self.file_size,
            'data_type': self.data_type,
            'num_rows': self.num_rows,
            'num_columns': self.num_columns,
            'column_info': self.column_info,
            'profile_status': self.profile_status,
            'profile_data': self.profile_data,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    def __repr__(self):
        return f'<Dataset {self.name}>'
