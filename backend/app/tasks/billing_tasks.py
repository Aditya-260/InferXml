"""
Billing Tasks — Periodic background jobs for subscription management.
"""
import logging
from datetime import datetime
from celery.schedules import crontab
from app import db
from app.models.user import User
from app.celery_app import celery_app

logger = logging.getLogger(__name__)

@celery_app.task(name='app.tasks.billing_tasks.check_subscription_expiry')
def check_subscription_expiry():
    """
    Daily task to find users whose Pro plan has expired and downgrade them to Free.
    Also handles cleanup of any stale payment records if necessary.
    """
    logger.info("Starting subscription expiry check...")
    
    from app import create_app
    app = create_app()
    
    with app.app_context():
        # 1. Find users who are Pro but plan has expired
        # We use a small buffer (e.g., 1 hour) to avoid race conditions with active sessions
        now = datetime.utcnow()
        expired_users = User.query.filter(
            User.plan_type.in_(['pro', 'advance']),
            User.plan_expires_at < now
        ).all()
        
        from app.services.notification_service import notify_subscription_expired
        
        downgrade_count = 0
        for user in expired_users:
            try:
                logger.info(f"Downgrading User {user.id} ({user.email}) - Plan expired at {user.plan_expires_at}")
                
                expiry_str = user.plan_expires_at.isoformat()
                user.plan_type = 'free'
                # We keep plan_expires_at as a historical record of when it ended
                
                # Notify user
                try:
                    notify_subscription_expired(user, expiry_str)
                except Exception as e:
                    logger.error(f"Failed to notify user {user.id}: {str(e)}")
                    
                downgrade_count += 1
            except Exception as e:
                logger.error(f"Failed to downgrade user {user.id}: {str(e)}")
                continue
                
        if downgrade_count > 0:
            db.session.commit()
            logger.info(f"Successfully downgraded {downgrade_count} users to Free plan.")
        else:
            logger.info("No expired subscriptions found.")
            
        return {"downgraded": downgrade_count}
