"""
InferX-ML Backend Application Factory
"""
import os
from flask import Flask, request as flask_request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from sqlalchemy import text

from .config import Config
from .commands import register_commands

db = SQLAlchemy()
migrate = Migrate()
jwt = JWTManager()

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address


def _get_real_ip():
    """
    Resolve the real client IP.
    Trusts the ProxyFix middleware to handle proxy headers securely.
    """
    return get_remote_address()


def get_user_rate_limit():
    try:
        from flask_jwt_extended import verify_jwt_in_request, get_jwt_identity
        from .models.user import User
        # Check if a valid JWT is present without enforcing it
        verify_jwt_in_request(optional=True)
        user_id = get_jwt_identity()
        if user_id is not None:
            user = User.query.get(int(user_id))
            if user and user.plan_type in ('pro', 'advance'):
                return "200 per minute"
    except Exception:
        pass
    return "20 per minute"

limiter = Limiter(
    key_func=_get_real_ip,
    default_limits=[get_user_rate_limit],
    storage_uri=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
    headers_enabled=True,  # Fix P1-7: expose X-RateLimit-* headers
    strategy="moving-window"  # Fix P2-12: smoother enforcement
)


@limiter.request_filter
def header_whitelist():
    internal_key = os.environ.get("INTERNAL_API_KEY")
    return bool(internal_key) and flask_request.headers.get("X-Internal-Service") == internal_key


def _sync_billing_schema():
    """Backfill billing columns for existing installs without full migrations."""
    if db.engine.dialect.name != 'postgresql':
        return

    statements = [
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS plan_type VARCHAR(20) NOT NULL DEFAULT 'free'",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS plan_expires_at TIMESTAMP",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS razorpay_customer_id VARCHAR(100)",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS razorpay_subscription_id VARCHAR(100)",
    ]
    for statement in statements:
        db.session.execute(text(statement))

    # Create the payments table if it doesn't exist
    db.session.execute(text("""
        CREATE TABLE IF NOT EXISTS payments (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id),
            razorpay_order_id VARCHAR(100) NOT NULL,
            razorpay_payment_id VARCHAR(100) UNIQUE,
            razorpay_signature VARCHAR(256),
            amount INTEGER NOT NULL,
            currency VARCHAR(10) DEFAULT 'INR',
            status VARCHAR(30) NOT NULL DEFAULT 'created',
            provider VARCHAR(30) DEFAULT 'razorpay',
            source VARCHAR(20) DEFAULT 'client',
            plan_granted VARCHAR(20),
            notes TEXT,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        )
    """))
    db.session.execute(text(
        "CREATE INDEX IF NOT EXISTS idx_payments_user_id ON payments(user_id)"
    ))
    db.session.execute(text(
        "CREATE INDEX IF NOT EXISTS idx_payments_order_id ON payments(razorpay_order_id)"
    ))
    db.session.commit()


def _sync_auth_schema():
    """Backfill auth columns for existing installs without full migrations."""
    if db.engine.dialect.name != 'postgresql':
        return

    statements = [
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS email_verified BOOLEAN NOT NULL DEFAULT FALSE",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS verification_code_hash VARCHAR(256)",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS verification_code_expires_at TIMESTAMP",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS verification_sent_at TIMESTAMP",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS password_reset_code_hash VARCHAR(256)",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS password_reset_code_expires_at TIMESTAMP",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS password_reset_sent_at TIMESTAMP",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS last_login_at TIMESTAMP",
    ]
    for statement in statements:
        db.session.execute(text(statement))
    db.session.commit()


def _sync_oauth_schema():
    """Add OAuth columns for existing installs."""
    if db.engine.dialect.name != 'postgresql':
        return

    statements = [
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS oauth_provider VARCHAR(20)",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS oauth_provider_id VARCHAR(256)",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS avatar_url VARCHAR(512)",
        "ALTER TABLE users ALTER COLUMN password_hash DROP NOT NULL",
    ]
    for statement in statements:
        db.session.execute(text(statement))
    # Add unique constraint for OAuth identity (ignore if exists)
    db.session.execute(text("""
        DO $$ BEGIN
            ALTER TABLE users ADD CONSTRAINT uq_oauth_identity
                UNIQUE (oauth_provider, oauth_provider_id);
        EXCEPTION WHEN duplicate_table THEN NULL;
        END $$;
    """))
    db.session.commit()


def _register_rate_limit_error_handler(app):
    """Return JSON instead of plain text on 429 errors (P0-4 adjacent)."""
    @app.errorhandler(429)
    def ratelimit_handler(e):
        # P1-11: Notify user when rate limit is hit
        try:
            from flask_jwt_extended import verify_jwt_in_request, get_jwt_identity
            from .models.user import User
            from .services.notification_service import notify_rate_limit_warning

            verify_jwt_in_request(optional=True)
            user_id = get_jwt_identity()
            if user_id is not None:
                user = User.query.get(int(user_id))
                if user:
                    notify_rate_limit_warning(user, "Exceeded", str(e.description))
        except Exception:
            pass

        return jsonify({
            'error': 'Rate limit exceeded',
            'message': str(e.description),
            'upgrade_url': '/pricing',
        }), 429


def create_app(config_class=Config):
    """Application factory pattern"""
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
    
    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    jwt.init_app(app)
    CORS(app)
    limiter.init_app(app)
    register_commands(app)
    
    # Register custom error handlers
    _register_rate_limit_error_handler(app)
    
    # Register blueprints
    from .routes.auth import auth_bp
    from .routes.datasets import datasets_bp
    from .routes.training import training_bp
    from .routes.predictions import predictions_bp
    from .routes.models import models_bp
    from .routes.system import system_bp
    from .routes.billing import billing_bp
    from .routes.notifications import notifications_bp
    from .routes.oauth import oauth_bp, init_oauth
    from .models.notification_preference import NotificationPreference  # ensure table creation
    from .models.payment import Payment  # ensure payments table creation
    
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(oauth_bp, url_prefix='/api/auth')
    app.register_blueprint(datasets_bp, url_prefix='/api/datasets')
    app.register_blueprint(training_bp, url_prefix='/api/training')
    app.register_blueprint(predictions_bp, url_prefix='/api/predict')
    app.register_blueprint(models_bp, url_prefix='/api/models')
    app.register_blueprint(system_bp, url_prefix='/api/system')
    app.register_blueprint(billing_bp, url_prefix='/api/billing')
    app.register_blueprint(notifications_bp, url_prefix='/api/notifications')
    init_oauth(app)
    
    # Health check endpoint (exempt from rate limiting)
    @app.route('/api/health')
    @limiter.exempt
    def health_check():
        return {'status': 'healthy', 'service': 'InferX-ML API'}

    with app.app_context():
        db.create_all()
        _sync_auth_schema()
        _sync_oauth_schema()
        _sync_billing_schema()
    
    return app
