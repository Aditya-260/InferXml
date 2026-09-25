"""
Cron Script: Downgrade Expired Pro Plans
Runs periodically (e.g., every hour) to find users whose Pro subscription has expired.
"""
import sys
import os
from datetime import datetime

# Add the parent directory to sys.path to allow importing 'app'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import create_app, db
from app.models.user import User
from app.services.notification_service import notify_subscription_expired

def downgrade_expired_plans():
    app = create_app()
    with app.app_context():
        now = datetime.utcnow()
        # Find Pro users whose plan has expired
        expired_users = User.query.filter(
            User.plan_type.in_(['pro', 'advance']),
            User.plan_expires_at < now
        ).all()

        if not expired_users:
            print(f"[{now}] No expired plans found.")
            return

        for user in expired_users:
            print(f"[{now}] Downgrading user {user.username} (ID: {user.id}). Expired at {user.plan_expires_at}")
            
            # Send notification about expiry
            try:
                notify_subscription_expired(user, user.plan_expires_at.strftime('%d %b %Y'))
            except Exception as e:
                print(f"  [!] Failed to notify user {user.id}: {e}")
            
            # Reset to free plan
            user.plan_type = 'free'
            
        try:
            db.session.commit()
            print(f"[{now}] Successfully downgraded {len(expired_users)} users.")
        except Exception as e:
            db.session.rollback()
            print(f"[{now}] ERROR: Failed to commit downgrades: {e}")

if __name__ == '__main__':
    downgrade_expired_plans()
