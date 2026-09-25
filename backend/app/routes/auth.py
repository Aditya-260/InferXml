"""
Authentication Routes
"""
import re
import secrets
from datetime import datetime, timedelta
from flask import Blueprint, current_app, request, jsonify
from flask_jwt_extended import (
    create_access_token, 
    create_refresh_token,
    jwt_required, 
    get_jwt_identity
)
from werkzeug.security import check_password_hash, generate_password_hash
from app import db, limiter
from app.models.user import User
from app.services.mailer_service import MailerError, mailer_service

auth_bp = Blueprint('auth', __name__)
PASSWORD_PATTERN = re.compile(r'^(?=.*[A-Za-z])(?=.*\d).{8,}$')
USERNAME_PATTERN = re.compile(r'^[A-Za-z0-9_]{3,32}$')


def _normalize_email(value):
    return (value or '').strip().lower()


def _password_error(password):
    if not PASSWORD_PATTERN.match(password or ''):
        return 'Password must be at least 8 characters and include letters and numbers'
    return None


def _issue_otp(user, purpose):
    now = datetime.utcnow()
    cooldown = timedelta(seconds=current_app.config.get('OTP_RESEND_COOLDOWN_SECONDS', 60))
    expiry = timedelta(minutes=current_app.config.get('OTP_EXPIRY_MINUTES', 10))

    sent_at_field = 'verification_sent_at' if purpose == 'verification' else 'password_reset_sent_at'
    code_hash_field = 'verification_code_hash' if purpose == 'verification' else 'password_reset_code_hash'
    expires_at_field = 'verification_code_expires_at' if purpose == 'verification' else 'password_reset_code_expires_at'

    last_sent_at = getattr(user, sent_at_field)
    if last_sent_at and (now - last_sent_at) < cooldown:
        remaining = int((cooldown - (now - last_sent_at)).total_seconds())
        return None, f'Please wait {remaining}s before requesting another code'

    otp = f'{secrets.randbelow(1000000):06d}'
    setattr(user, code_hash_field, generate_password_hash(otp))
    setattr(user, expires_at_field, now + expiry)
    setattr(user, sent_at_field, now)
    db.session.commit()
    return otp, None


def _send_verification_email(user):
    otp, error_message = _issue_otp(user, 'verification')
    if error_message:
        return error_message
    try:
        mailer_service.send_template_email(
            'verification',
            user.email,
            {
                'username': user.username,
                'otp': otp,
                'expiresInMinutes': current_app.config.get('OTP_EXPIRY_MINUTES', 10),
            }
        )
    except MailerError as exc:
        return str(exc)
    return None


def _send_password_reset_email(user):
    otp, error_message = _issue_otp(user, 'reset')
    if error_message:
        return error_message
    try:
        mailer_service.send_template_email(
            'password_reset',
            user.email,
            {
                'username': user.username,
                'otp': otp,
                'expiresInMinutes': current_app.config.get('OTP_EXPIRY_MINUTES', 10),
            }
        )
    except MailerError as exc:
        return str(exc)
    return None


def _issue_auth_payload(user):
    access_token = create_access_token(identity=str(user.id))
    refresh_token = create_refresh_token(identity=str(user.id))
    user.last_login_at = datetime.utcnow()
    db.session.commit()
    return {
        'access_token': access_token,
        'refresh_token': refresh_token,
        'user': user.to_dict()
    }


@auth_bp.route('/register', methods=['POST'])
@limiter.limit("3 per minute")  # Stricter for registrations

def register():
    """Register a new user"""
    data = request.get_json()
    
    # Validate input
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    email = _normalize_email(data.get('email'))
    username = (data.get('username') or '').strip()
    password = data.get('password')
    
    if not all([email, username, password]):
        return jsonify({'error': 'Email, username, and password are required'}), 400

    if not USERNAME_PATTERN.match(username):
        return jsonify({'error': 'Username must be 3-32 characters and use only letters, numbers, or underscores'}), 400

    password_error = _password_error(password)
    if password_error:
        return jsonify({'error': password_error}), 400
    
    # Check if user exists
    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email already registered'}), 409
    
    if User.query.filter_by(username=username).first():
        return jsonify({'error': 'Username already taken'}), 409
    
    # Create new user
    user = User(email=email, username=username)
    user.set_password(password)
    
    db.session.add(user)
    db.session.commit()

    mail_error = _send_verification_email(user)
    if mail_error:
        db.session.delete(user)
        db.session.commit()
        return jsonify({'error': f'Unable to send verification email. {mail_error}'}), 503

    return jsonify({
        'message': 'Account created. Verify your email to continue.',
        'requires_verification': True,
        'email': user.email
    }), 201


@auth_bp.route('/login', methods=['POST'])
@limiter.limit("5 per minute")
def login():
    """Login user and return tokens"""
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    email = _normalize_email(data.get('email'))
    password = data.get('password')
    
    if not all([email, password]):
        return jsonify({'error': 'Email and password are required'}), 400
    
    # Find user
    user = User.query.filter_by(email=email).first()
    
    if not user or not user.check_password(password):
        return jsonify({'error': 'Invalid email or password'}), 401
    
    if not user.is_active:
        return jsonify({'error': 'Account is deactivated'}), 403

    if not user.email_verified:
        return jsonify({
            'error': 'Verify your email before signing in',
            'code': 'EMAIL_NOT_VERIFIED',
            'requires_verification': True,
            'email': user.email
        }), 403

    return jsonify(_issue_auth_payload(user)), 200


@auth_bp.route('/verify-email', methods=['POST'])
@limiter.limit("10 per minute")

def verify_email():
    data = request.get_json() or {}
    email = _normalize_email(data.get('email'))
    otp = (data.get('otp') or '').strip()

    if not email or not otp:
        return jsonify({'error': 'Email and verification code are required'}), 400

    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({'error': 'Verification session not found'}), 404

    if user.email_verified:
        return jsonify(_issue_auth_payload(user)), 200

    if not user.verification_code_hash or not user.verification_code_expires_at:
        return jsonify({'error': 'Verification code not found. Request a new one.'}), 400

    if user.verification_code_expires_at < datetime.utcnow():
        return jsonify({'error': 'Verification code expired. Request a new one.'}), 400

    if not check_password_hash(user.verification_code_hash, otp):
        return jsonify({'error': 'Invalid verification code'}), 400

    user.email_verified = True
    user.verification_code_hash = None
    user.verification_code_expires_at = None
    user.verification_sent_at = None
    db.session.commit()

    return jsonify(_issue_auth_payload(user)), 200


@auth_bp.route('/resend-verification', methods=['POST'])
@limiter.limit("3 per minute")
def resend_verification():
    data = request.get_json() or {}
    email = _normalize_email(data.get('email'))

    if not email:
        return jsonify({'error': 'Email is required'}), 400

    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({'error': 'Account not found'}), 404

    if user.email_verified:
        return jsonify({'message': 'Email is already verified'}), 200

    mail_error = _send_verification_email(user)
    if mail_error:
        status = 429 if mail_error.startswith('Please wait') else 503
        return jsonify({'error': mail_error}), status

    return jsonify({'message': 'Verification code sent'}), 200


@auth_bp.route('/forgot-password', methods=['POST'])
@limiter.limit("3 per minute")  # Protect against email enumeration / spam

def forgot_password():
    data = request.get_json() or {}
    email = _normalize_email(data.get('email'))
    generic_message = 'If an account exists for that email, a reset code has been sent'

    if not email:
        return jsonify({'message': generic_message}), 200

    user = User.query.filter_by(email=email).first()
    if not user or not user.is_active or not user.email_verified:
        return jsonify({'message': generic_message}), 200

    mail_error = _send_password_reset_email(user)
    if mail_error and mail_error.startswith('Please wait'):
        return jsonify({'error': mail_error}), 429
    if mail_error:
        return jsonify({'error': mail_error}), 503

    return jsonify({'message': generic_message}), 200


@auth_bp.route('/reset-password', methods=['POST'])
@limiter.limit("3 per minute")

def reset_password():
    data = request.get_json() or {}
    email = _normalize_email(data.get('email'))
    otp = (data.get('otp') or '').strip()
    password = data.get('password')

    if not all([email, otp, password]):
        return jsonify({'error': 'Email, reset code, and new password are required'}), 400

    password_error = _password_error(password)
    if password_error:
        return jsonify({'error': password_error}), 400

    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({'error': 'Reset session not found'}), 404

    if not user.password_reset_code_hash or not user.password_reset_code_expires_at:
        return jsonify({'error': 'Reset code not found. Request a new one.'}), 400

    if user.password_reset_code_expires_at < datetime.utcnow():
        return jsonify({'error': 'Reset code expired. Request a new one.'}), 400

    if not check_password_hash(user.password_reset_code_hash, otp):
        return jsonify({'error': 'Invalid reset code'}), 400

    user.set_password(password)
    user.password_reset_code_hash = None
    user.password_reset_code_expires_at = None
    user.password_reset_sent_at = None
    db.session.commit()

    return jsonify({'message': 'Password updated successfully'}), 200


@auth_bp.route('/change-password', methods=['POST'])
@jwt_required()
@limiter.limit("5 per minute")

def change_password():
    """Change password for the currently authenticated user."""
    current_user_id = int(get_jwt_identity())
    user = User.query.get(current_user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404

    data = request.get_json() or {}
    current_password = data.get('current_password')
    new_password = data.get('new_password')

    if not all([current_password, new_password]):
        return jsonify({'error': 'Current password and new password are required'}), 400

    if not user.check_password(current_password):
        return jsonify({'error': 'Current password is incorrect'}), 401

    password_error = _password_error(new_password)
    if password_error:
        return jsonify({'error': password_error}), 400

    user.set_password(new_password)
    db.session.commit()

    return jsonify({'message': 'Password changed successfully'}), 200


@auth_bp.route('/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh():
    """Refresh access token"""
    current_user_id = int(get_jwt_identity())
    access_token = create_access_token(identity=str(current_user_id))
    
    return jsonify({'access_token': access_token}), 200


@auth_bp.route('/me', methods=['GET'])
@jwt_required()
def get_current_user():
    """Get current user info"""
    current_user_id = int(get_jwt_identity())
    user = User.query.get(current_user_id)
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    return jsonify({'user': user.to_dict()}), 200


@auth_bp.route('/update-username', methods=['PUT'])
@jwt_required()
@limiter.limit("5 per minute")
def update_username():
    """Update the username for the currently authenticated user."""
    current_user_id = int(get_jwt_identity())
    user = User.query.get(current_user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404

    data = request.get_json() or {}
    new_username = (data.get('username') or '').strip()

    if not new_username:
        return jsonify({'error': 'Username is required'}), 400

    if not USERNAME_PATTERN.match(new_username):
        return jsonify({'error': 'Username must be 3-32 characters and use only letters, numbers, or underscores'}), 400

    if new_username == user.username:
        return jsonify({'user': user.to_dict()}), 200

    existing = User.query.filter_by(username=new_username).first()
    if existing:
        return jsonify({'error': 'Username already taken'}), 409

    user.username = new_username
    db.session.commit()

    return jsonify({'user': user.to_dict(), 'message': 'Username updated successfully'}), 200


@auth_bp.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    """Logout user (client should discard tokens)"""
    # In a production app, you'd want to blacklist the token
    return jsonify({'message': 'Logged out successfully'}), 200

