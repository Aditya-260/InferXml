"""
Payment Model — Stores all payment transactions for audit trail and billing history.
"""
from datetime import datetime
from app import db


class Payment(db.Model):
    """Records every payment event (from client verify or Razorpay webhook)."""

    __tablename__ = 'payments'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    razorpay_order_id = db.Column(db.String(100), nullable=False, index=True)
    razorpay_payment_id = db.Column(db.String(100), nullable=True, unique=True)
    razorpay_signature = db.Column(db.String(256), nullable=True)
    amount = db.Column(db.Integer, nullable=False)            # paise
    currency = db.Column(db.String(10), default='INR')
    status = db.Column(db.String(30), nullable=False, default='created')
    # status enum: created | captured | failed | refunded
    provider = db.Column(db.String(30), default='razorpay')
    source = db.Column(db.String(20), default='client')       # 'client' or 'webhook'
    plan_granted = db.Column(db.String(20), nullable=True)     # e.g. 'pro'
    notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship
    user = db.relationship('User', backref=db.backref('payments', lazy='dynamic'))

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'razorpay_order_id': self.razorpay_order_id,
            'razorpay_payment_id': self.razorpay_payment_id,
            'amount': self.amount,
            'currency': self.currency,
            'status': self.status,
            'source': self.source,
            'plan_granted': self.plan_granted,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self):
        return f'<Payment {self.razorpay_payment_id} - {self.status}>'
