import click
from datetime import datetime
from flask.cli import with_appcontext


@click.command('check-subscriptions')
@with_appcontext
def check_subscriptions():
    """
    Check for expired subscriptions and downgrade users to free plan.
    This should be run daily via cron.
    """
    from app import db
    from app.models.user import User
    from app.services.notification_service import notify_subscription_expired

    click.echo(f"[{datetime.utcnow().isoformat()}] Checking expired subscriptions...")
    
    now = datetime.utcnow()
    # Find users whose Pro plan has expired
    expired_users = User.query.filter(
        User.plan_type.in_(['pro', 'advance']),
        User.plan_expires_at < now
    ).all()
    
    downgraded_count = 0
    for user in expired_users:
        click.echo(f"  Downgrading user {user.username} (ID: {user.id}) - Expired at {user.plan_expires_at}")
        user.plan_type = 'free'
        # We keep plan_expires_at for history or re-activation check
        
        # Notify user
        try:
            notify_subscription_expired(user, user.plan_expires_at.isoformat())
        except Exception as e:
            click.echo(f"    Failed to notify {user.email}: {e}")
            
        downgraded_count += 1
    
    db.session.commit()
    click.echo(f"Successfully downgraded {downgraded_count} users.")

def register_commands(app):
    app.cli.add_command(check_subscriptions)
