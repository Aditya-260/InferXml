"""
OAuth Routes — Google & GitHub
"""
import json
import logging
import requests as http_requests
from urllib.parse import urlencode
from datetime import datetime
from flask import Blueprint, current_app, redirect, request, url_for
from flask_jwt_extended import create_access_token
from authlib.integrations.flask_client import OAuth
from app import db
from app.models.user import User

logger = logging.getLogger(__name__)

oauth_bp = Blueprint('oauth', __name__)
oauth = OAuth()


def init_oauth(app):
    """Called from app factory to configure OAuth providers."""
    oauth.init_app(app)

    oauth.register(
        name='google',
        client_id=app.config['GOOGLE_CLIENT_ID'],
        client_secret=app.config['GOOGLE_CLIENT_SECRET'],
        server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
        client_kwargs={'scope': 'openid email profile'},
    )

    oauth.register(
        name='github',
        client_id=app.config['GITHUB_CLIENT_ID'],
        client_secret=app.config['GITHUB_CLIENT_SECRET'],
        access_token_url='https://github.com/login/oauth/access_token',
        authorize_url='https://github.com/login/oauth/authorize',
        api_base_url='https://api.github.com/',
        client_kwargs={'scope': 'user:email'},
    )


def _find_or_create_user(provider, provider_id, email, username, avatar_url):
    """Find existing OAuth user or create a new one.

    Resolution order:
    1. Exact OAuth identity match → log in
    2. Email matches existing account → link OAuth to it
    3. No match → create new user
    """
    # 1. Check for existing OAuth link
    user = User.query.filter_by(
        oauth_provider=provider,
        oauth_provider_id=provider_id,
    ).first()
    if user:
        user.last_login_at = datetime.utcnow()
        user.avatar_url = avatar_url
        db.session.commit()
        return user

    # 2. Check if email already exists (e.g. previously registered with password)
    user = User.query.filter_by(email=email).first()
    if user:
        user.oauth_provider = provider
        user.oauth_provider_id = provider_id
        user.avatar_url = avatar_url
        user.email_verified = True
        user.last_login_at = datetime.utcnow()
        db.session.commit()
        return user

    # 3. Create new user — handle username collisions
    base_username = username or email.split('@')[0]
    final_username = base_username[:32]  # respect model max length
    counter = 1
    while User.query.filter_by(username=final_username).first():
        suffix = f'_{counter}'
        final_username = f'{base_username[:32 - len(suffix)]}{suffix}'
        counter += 1

    user = User(
        email=email,
        username=final_username,
        oauth_provider=provider,
        oauth_provider_id=provider_id,
        avatar_url=avatar_url,
        email_verified=True,
        is_active=True,
    )
    db.session.add(user)
    db.session.commit()
    return user


def _issue_jwt_and_redirect(user):
    """Create a JWT and redirect to frontend with token + user info."""
    access_token = create_access_token(identity=str(user.id))
    frontend_url = current_app.config['FRONTEND_URL']

    params = urlencode({
        'token': access_token,
        'user': json.dumps(user.to_dict()),
    })
    return redirect(f'{frontend_url}/auth/callback?{params}')


# ──────────────── Google ────────────────

@oauth_bp.route('/google')
def google_login():
    redirect_uri = url_for('oauth.google_callback', _external=True)
    return oauth.google.authorize_redirect(redirect_uri)


@oauth_bp.route('/google/callback')
def google_callback():
    try:
        token = oauth.google.authorize_access_token()
        user_info = token.get('userinfo') or oauth.google.userinfo()

        user = _find_or_create_user(
            provider='google',
            provider_id=user_info['sub'],
            email=user_info['email'],
            username=user_info.get('name', '').replace(' ', '_').lower(),
            avatar_url=user_info.get('picture'),
        )
        return _issue_jwt_and_redirect(user)
    except Exception as exc:
        logger.error('Google OAuth error: %s', exc, exc_info=True)
        frontend_url = current_app.config['FRONTEND_URL']
        params = urlencode({'error': str(exc)})
        return redirect(f'{frontend_url}/login?{params}')


# ──────────────── GitHub ────────────────

@oauth_bp.route('/github')
def github_login():
    redirect_uri = url_for('oauth.github_callback', _external=True)
    return oauth.github.authorize_redirect(redirect_uri)


@oauth_bp.route('/github/callback')
def github_callback():
    try:
        code = request.args.get('code')
        logger.info('GitHub callback received code: %s', code[:8] if code else None)

        # Manual token exchange — more reliable than Authlib for GitHub
        token_resp = http_requests.post(
            'https://github.com/login/oauth/access_token',
            headers={'Accept': 'application/json'},
            data={
                'client_id': current_app.config['GITHUB_CLIENT_ID'],
                'client_secret': current_app.config['GITHUB_CLIENT_SECRET'],
                'code': code,
            },
            timeout=15,
        )
        token_data = token_resp.json()
        logger.info('GitHub token response keys: %s', list(token_data.keys()))

        if 'error' in token_data:
            raise ValueError(f"GitHub token error: {token_data['error_description']}")

        access_token = token_data['access_token']

        # Fetch user profile
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Accept': 'application/json',
        }
        profile_resp = http_requests.get('https://api.github.com/user', headers=headers, timeout=15)
        profile = profile_resp.json()
        logger.info('GitHub user: %s (id=%s)', profile.get('login'), profile.get('id'))

        # Fetch email (may not be in profile)
        email = profile.get('email')
        if not email:
            emails_resp = http_requests.get('https://api.github.com/user/emails', headers=headers, timeout=15)
            emails = emails_resp.json()
            logger.info('GitHub emails response type: %s, value: %s', type(emails).__name__, emails)

            if isinstance(emails, list) and len(emails) > 0:
                if isinstance(emails[0], dict):
                    # OAuth App format: [{"email": "...", "primary": true, "verified": true}, ...]
                    primary = next(
                        (e for e in emails if e.get('primary') and e.get('verified')),
                        None,
                    )
                    email = primary['email'] if primary else emails[0]['email']
                else:
                    # GitHub App format: plain list of strings ["user@example.com", ...]
                    email = emails[0]

        if not email:
            # Last resort: construct from username
            email = f"{profile.get('login', 'user')}@users.noreply.github.com"
            logger.warning('Using noreply email fallback: %s', email)

        user = _find_or_create_user(
            provider='github',
            provider_id=str(profile['id']),
            email=email,
            username=profile.get('login', ''),
            avatar_url=profile.get('avatar_url'),
        )
        return _issue_jwt_and_redirect(user)
    except Exception as exc:
        logger.error('GitHub OAuth error: %s', exc, exc_info=True)
        frontend_url = current_app.config['FRONTEND_URL']
        params = urlencode({'error': str(exc)})
        return redirect(f'{frontend_url}/login?{params}')

