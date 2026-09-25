"""
Notification Preferences API
GET  /api/notifications/preferences   — fetch current toggles
PUT  /api/notifications/preferences   — update toggles
"""
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity

from app import db
from app.models.user import User
from app.models.notification_preference import NotificationPreference

notifications_bp = Blueprint('notifications', __name__)


@notifications_bp.route('/preferences', methods=['GET'])
@jwt_required()
def get_preferences():
    """Return the authenticated user's notification toggles."""
    user_id = get_jwt_identity()
    prefs = NotificationPreference.get_or_create(user_id)
    return jsonify(prefs.to_dict())


@notifications_bp.route('/preferences', methods=['PUT'])
@jwt_required()
def update_preferences():
    """Bulk-update notification toggles.

    Accepts a JSON body like:
        { "training_completed": true, "weekly_report": false, ... }
    Only keys present in NotificationPreference.FIELDS are accepted;
    unknown keys are silently ignored.
    """
    user_id = get_jwt_identity()
    data = request.get_json(silent=True) or {}

    prefs = NotificationPreference.get_or_create(user_id)

    changed = False
    for field in NotificationPreference.FIELDS:
        if field in data and isinstance(data[field], bool):
            setattr(prefs, field, data[field])
            changed = True

    if changed:
        db.session.commit()

    return jsonify(prefs.to_dict())
