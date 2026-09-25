"""
Celery Application Configuration
Background task processing for model training
"""
from celery import Celery
from celery.schedules import crontab
from app.config import Config

# Create Celery app
celery_app = Celery(
    'inferx_ml',
    broker=Config.CELERY_BROKER_URL,
    backend=Config.CELERY_RESULT_BACKEND,
    include=['app.tasks.training_tasks', 'app.tasks.billing_tasks']
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Beat Schedule (Cron Jobs)
    beat_schedule={
        'check-subscriptions-daily': {
            'task': 'app.tasks.billing_tasks.check_subscription_expiry',
            'schedule': crontab(hour=0, minute=0),  # Every midnight
        },
    },
    
    # Task tracking
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max per task
    task_soft_time_limit=3300,  # 55 min soft limit
    
    # Result settings
    result_expires=86400,  # Results expire after 1 day
    
    # Worker settings
    worker_prefetch_multiplier=1,
    worker_concurrency=2,
)


# Optional: Import tasks after configuration
def init_celery(app):
    """Initialize Celery with Flask app context"""
    celery_app.conf.update(app.config)
    
    class ContextTask(celery_app.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)
    
    celery_app.Task = ContextTask
    return celery_app
