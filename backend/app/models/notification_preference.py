"""
Notification Preferences Model
"""
from app import db


class NotificationPreference(db.Model):
    """Per-user notification toggle states."""

    __tablename__ = 'notification_preferences'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), unique=True, nullable=False, index=True)

    training_completed = db.Column(db.Boolean, default=True, nullable=False)
    training_failed = db.Column(db.Boolean, default=True, nullable=False)
    dataset_uploaded = db.Column(db.Boolean, default=False, nullable=False)
    api_rate_limit = db.Column(db.Boolean, default=True, nullable=False)
    weekly_report = db.Column(db.Boolean, default=False, nullable=False)
    security_alerts = db.Column(db.Boolean, default=True, nullable=False)
    product_updates = db.Column(db.Boolean, default=False, nullable=False)
    billing_updates = db.Column(db.Boolean, default=True, nullable=False)

    # Relationship
    user = db.relationship('User', backref=db.backref('notification_prefs', uselist=False, lazy='joined'))

    # All toggles in order, matching the frontend keys
    FIELDS = [
        'training_completed',
        'training_failed',
        'dataset_uploaded',
        'api_rate_limit',
        'weekly_report',
        'security_alerts',
        'product_updates',
        'billing_updates',
    ]

    def to_dict(self):
        return {f: getattr(self, f) for f in self.FIELDS}

    @classmethod
    def get_or_create(cls, user_id):
        """Return existing prefs or create defaults."""
        prefs = cls.query.filter_by(user_id=user_id).first()
        if not prefs:
            prefs = cls(user_id=user_id)
            db.session.add(prefs)
            db.session.commit()
        return prefs

    def __repr__(self):
        return f'<NotificationPreference user_id={self.user_id}>'
