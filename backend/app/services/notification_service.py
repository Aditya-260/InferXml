"""
Notification Service
Central dispatch for sending email notifications based on user preferences.
Checks user's NotificationPreference before sending via the mailer service.
"""
import os
from datetime import datetime
from app.models.notification_preference import NotificationPreference
from app.services.mailer_service import mailer_service, MailerError


def _should_notify(user_id, pref_field):
    """Check if the user has enabled a particular notification type."""
    try:
        prefs = NotificationPreference.query.filter_by(user_id=user_id).first()
        if not prefs:
            # Use class-level defaults when no row exists yet
            default = NotificationPreference.__table__.columns[pref_field].default
            return default.arg if default else False
        return getattr(prefs, pref_field, False)
    except Exception:
        return False


def notify_training_completed(user, experiment_name, model_name=None):
    """Send email when training finishes successfully."""
    if not _should_notify(user.id, 'training_completed'):
        return
    try:
        mailer_service.send_template_email(
            'training_completed',
            user.email,
            {
                'username': user.username,
                'experiment_name': experiment_name,
                'model_name': model_name or experiment_name,
            }
        )
    except MailerError as e:
        print(f"[notify] training_completed email failed for user {user.id}: {e}")


def notify_training_failed(user, experiment_name, error_message=''):
    """Send email when a training job fails."""
    if not _should_notify(user.id, 'training_failed'):
        return
    try:
        mailer_service.send_template_email(
            'training_failed',
            user.email,
            {
                'username': user.username,
                'experiment_name': experiment_name,
                'error_message': error_message or 'An unexpected error occurred.',
            }
        )
    except MailerError as e:
        print(f"[notify] training_failed email failed for user {user.id}: {e}")


def notify_dataset_uploaded(user, dataset_name):
    """Send email when a dataset is uploaded successfully."""
    if not _should_notify(user.id, 'dataset_uploaded'):
        return
    try:
        mailer_service.send_template_email(
            'dataset_uploaded',
            user.email,
            {
                'username': user.username,
                'dataset_name': dataset_name,
            }
        )
    except MailerError as e:
        print(f"[notify] dataset_uploaded email failed for user {user.id}: {e}")


def notify_rate_limit_warning(user, current_usage, limit):
    """Send email when approaching API rate limit."""
    if not _should_notify(user.id, 'api_rate_limit'):
        return
    try:
        mailer_service.send_template_email(
            'rate_limit_warning',
            user.email,
            {
                'username': user.username,
                'current_usage': current_usage,
                'limit': limit,
            }
        )
    except MailerError as e:
        print(f"[notify] rate_limit_warning email failed for user {user.id}: {e}")


def notify_security_alert(user, event_type, details=''):
    """Send email for security events (new login, password change)."""
    if not _should_notify(user.id, 'security_alerts'):
        return
    try:
        mailer_service.send_template_email(
            'security_alert',
            user.email,
            {
                'username': user.username,
                'event_type': event_type,
                'details': details,
                'timestamp': datetime.utcnow().isoformat(),
            }
        )
    except MailerError as e:
        print(f"[notify] security_alert email failed for user {user.id}: {e}")


def notify_subscription_expired(user, expiry_date):
    """Send email when Pro subscription expires."""
    if not _should_notify(user.id, 'billing_updates'):
        return
    try:
        mailer_service.send_template_email(
            'subscription_expired',
            user.email,
            {
                'username': user.username,
                'expiry_date': expiry_date,
            }
        )
    except MailerError as e:
        print(f"[notify] subscription_expired email failed for user {user.id}: {e}")


def notify_subscription_renewed(user, new_expiry_date, plan_name='pro'):
    """Send email when Pro subscription is successfully renewed/extended."""
    if not _should_notify(user.id, 'billing_updates'):
        return
    try:
        mailer_service.send_template_email(
            'subscription_renewed',
            user.email,
            {
                'username': user.username,
                'new_expiry_date': new_expiry_date,
                'plan_name': plan_name,
            }
        )
    except MailerError as e:
        print(f"[notify] subscription_renewed email failed for user {user.id}: {e}")
