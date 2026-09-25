"""
User Model
"""
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from app import db


class User(db.Model):
    """User model for authentication"""
    
    __tablename__ = 'users'
    __table_args__ = (
        db.UniqueConstraint('oauth_provider', 'oauth_provider_id', name='uq_oauth_identity'),
    )
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    email_verified = db.Column(db.Boolean, default=False, nullable=False)
    verification_code_hash = db.Column(db.String(256), nullable=True)
    verification_code_expires_at = db.Column(db.DateTime, nullable=True)
    verification_sent_at = db.Column(db.DateTime, nullable=True)
    password_reset_code_hash = db.Column(db.String(256), nullable=True)
    password_reset_code_expires_at = db.Column(db.DateTime, nullable=True)
    password_reset_sent_at = db.Column(db.DateTime, nullable=True)
    last_login_at = db.Column(db.DateTime, nullable=True)

    # OAuth
    oauth_provider = db.Column(db.String(20), nullable=True)
    oauth_provider_id = db.Column(db.String(256), nullable=True)
    avatar_url = db.Column(db.String(512), nullable=True)
    
    # Billing & Plan
    plan_type = db.Column(db.String(20), server_default='free', nullable=False)
    plan_expires_at = db.Column(db.DateTime, nullable=True)
    razorpay_customer_id = db.Column(db.String(100), nullable=True)
    razorpay_subscription_id = db.Column(db.String(100), nullable=True)
    
    # Relationships
    datasets = db.relationship('Dataset', backref='owner', lazy='dynamic')
    experiments = db.relationship('Experiment', backref='owner', lazy='dynamic')
    
    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check password against hash"""
        if not self.password_hash:
            return False
        return check_password_hash(self.password_hash, password)
    
    @property
    def is_pro_active(self):
        """Check if user has an active paid subscription (Pro or Advance)."""
        if self.plan_type not in ('pro', 'advance'):
            return False
        if self.plan_expires_at and self.plan_expires_at < datetime.utcnow():
            return False
        return True
    
    def to_dict(self):
        """Serialize to dictionary"""
        return {
            'id': self.id,
            'email': self.email,
            'username': self.username,
            'created_at': self.created_at.isoformat(),
            'is_active': self.is_active,
            'email_verified': self.email_verified,
            'last_login_at': self.last_login_at.isoformat() if self.last_login_at else None,
            'plan_type': self.plan_type,
            'plan_expires_at': self.plan_expires_at.isoformat() if self.plan_expires_at else None,
            'is_pro_active': self.is_pro_active,
            'oauth_provider': self.oauth_provider,
            'avatar_url': self.avatar_url,
        }
    
    def __repr__(self):
        return f'<User {self.username}>'
