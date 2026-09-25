"""
Billing Routes — Razorpay integration with webhook support.

P0-1: Webhook handler with signature verification
P0-2: All transactions recorded in payments table
P0-4: Per-route rate limits on sensitive billing endpoints

Uses real Razorpay by default. Set MOCK_PAYMENTS=true in .env
only for local development without Razorpay credentials.
"""
import os
import hmac
import hashlib
import logging
import uuid
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity

from app import db, limiter
from app.models.user import User
from app.models.payment import Payment

logger = logging.getLogger(__name__)

billing_bp = Blueprint('billing', __name__)

# ──────────────────────────────────────────────
# Mock mode flag — defaults to false (real Razorpay)
# Set MOCK_PAYMENTS=true in .env for local dev without Razorpay keys
# ──────────────────────────────────────────────
MOCK_PAYMENTS = os.environ.get('MOCK_PAYMENTS', 'false').lower() in ('true', '1', 'yes')

# ──────────────────────────────────────────────
# Razorpay client (lazy singleton)
# ──────────────────────────────────────────────
_razorpay_client = None


def get_razorpay_client():
    global _razorpay_client
    if MOCK_PAYMENTS:
        return None          # never touch real Razorpay in mock mode
    if not _razorpay_client:
        try:
            import razorpay
            key_id = os.environ.get('RAZORPAY_KEY_ID')
            key_secret = os.environ.get('RAZORPAY_KEY_SECRET')
            if key_id and key_secret:
                _razorpay_client = razorpay.Client(auth=(key_id, key_secret))
        except ImportError:
            logger.warning("razorpay package not installed")
    return _razorpay_client


def _get_razorpay_secret():
    return os.environ.get('RAZORPAY_KEY_SECRET', '')


def _get_webhook_secret():
    """Razorpay webhook secret (configured in Razorpay dashboard)."""
    return os.environ.get('RAZORPAY_WEBHOOK_SECRET', _get_razorpay_secret())


PRO_PLAN_AMOUNT = int(os.environ.get('PRO_PLAN_AMOUNT_PAISE', '49900'))  # P1-10: configurable
PRO_PLAN_DURATION_DAYS = int(os.environ.get('PRO_PLAN_DURATION_DAYS', '30'))

ADVANCE_PLAN_AMOUNT = int(os.environ.get('ADVANCE_PLAN_AMOUNT_PAISE', '499900'))
ADVANCE_PLAN_DURATION_DAYS = int(os.environ.get('ADVANCE_PLAN_DURATION_DAYS', '365'))


# ──────────────────────────────────────────────
# GET /api/billing/config — tells frontend which mode is active
# ──────────────────────────────────────────────
@billing_bp.route('/config', methods=['GET'])
def billing_config():
    return jsonify({
        'mock_payments': MOCK_PAYMENTS,
        'pro_plan_amount': PRO_PLAN_AMOUNT,
        'advance_plan_amount': ADVANCE_PLAN_AMOUNT,
        'currency': 'INR',
    })


# ──────────────────────────────────────────────
# POST /api/billing/create-order
# ──────────────────────────────────────────────
@billing_bp.route('/create-order', methods=['POST'])
@jwt_required()
@limiter.limit("10 per minute")  # P0-4: stricter limit on payment creation
def create_order():
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        data = request.get_json() if request.is_json else {}
        plan = data.get('plan', 'pro')
        if plan not in ('pro', 'advance'):
            return jsonify({'error': 'Invalid plan selected'}), 400

        amount = ADVANCE_PLAN_AMOUNT if plan == 'advance' else PRO_PLAN_AMOUNT

        # ── Mock mode ──
        if MOCK_PAYMENTS:
            mock_order_id = f"order_mock_{uuid.uuid4().hex[:16]}"
            payment = Payment(
                user_id=int(user_id),
                razorpay_order_id=mock_order_id,
                amount=amount,
                currency='INR',
                status='created',
                source='mock',
                plan_granted=plan,
            )
            db.session.add(payment)
            db.session.commit()
            return jsonify({
                'order_id': mock_order_id,
                'amount': amount,
                'currency': 'INR',
                'key_id': 'mock_key',
                'mock': True,
            })

        # ── Real Razorpay ──
        client = get_razorpay_client()
        if not client:
            return jsonify({'error': 'Payment gateway not configured'}), 503

        order_data = {
            "amount": amount,
            "currency": "INR",
            "receipt": f"receipt_user_{user_id}_{int(datetime.utcnow().timestamp())}",
            "notes": {
                "user_id": str(user_id),
                "email": user.email,
                "plan": plan,
            }
        }

        order = client.order.create(data=order_data)

        # Record the order in payments table (status=created)
        payment = Payment(
            user_id=int(user_id),
            razorpay_order_id=order['id'],
            amount=amount,
            currency='INR',
            status='created',
            source='client',
            plan_granted=plan,
        )
        db.session.add(payment)
        db.session.commit()

        return jsonify({
            'order_id': order['id'],
            'amount': order['amount'],
            'currency': order['currency'],
            'key_id': os.environ.get('RAZORPAY_KEY_ID')
        })

    except Exception as e:
        logger.exception("Error creating order")
        return jsonify({'error': 'Failed to create payment order'}), 500


# ──────────────────────────────────────────────
# POST /api/billing/verify-payment (client-side)
# ──────────────────────────────────────────────
@billing_bp.route('/verify-payment', methods=['POST'])
@jwt_required()
@limiter.limit("10 per minute")
def verify_payment():
    try:
        data = request.get_json()
        razorpay_payment_id = data.get('razorpay_payment_id')
        razorpay_order_id = data.get('razorpay_order_id')
        razorpay_signature = data.get('razorpay_signature')

        if not all([razorpay_payment_id, razorpay_order_id]):
            return jsonify({'error': 'Missing payment verification details'}), 400

        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # ── Mock mode: skip signature verification ──
        if MOCK_PAYMENTS:
            logger.info(f"Mock payment verified for user {user_id}")
        else:
            # Real Razorpay: require signature + verify
            if not razorpay_signature:
                return jsonify({'error': 'Missing payment signature'}), 400

            client = get_razorpay_client()
            if not client:
                return jsonify({'error': 'Payment gateway not configured'}), 503

            try:
                client.utility.verify_payment_signature({
                    'razorpay_order_id': razorpay_order_id,
                    'razorpay_payment_id': razorpay_payment_id,
                    'razorpay_signature': razorpay_signature
                })
            except Exception:
                return jsonify({'error': 'Payment verification failed'}), 400

        # Idempotency check: skip if already captured for this payment_id
        existing = Payment.query.filter_by(razorpay_payment_id=razorpay_payment_id).first()
        if existing and existing.status == 'captured':
            return jsonify({'message': 'Payment already verified.'}), 200

        # Update or create payment record
        payment = Payment.query.filter_by(razorpay_order_id=razorpay_order_id).first()
        plan_to_grant = 'pro'
        if payment:
            payment.razorpay_payment_id = razorpay_payment_id
            payment.razorpay_signature = razorpay_signature
            payment.status = 'captured'
            payment.source = 'client'
            if not payment.plan_granted:
                payment.plan_granted = plan_to_grant
            plan_to_grant = payment.plan_granted
            payment.updated_at = datetime.utcnow()
        else:
            payment = Payment(
                user_id=int(user_id),
                razorpay_order_id=razorpay_order_id,
                razorpay_payment_id=razorpay_payment_id,
                razorpay_signature=razorpay_signature,
                amount=PRO_PLAN_AMOUNT,  # Fallback
                currency='INR',
                status='captured',
                source='client',
                plan_granted=plan_to_grant,
            )
            db.session.add(payment)

        # Upgrade the user
        now = datetime.utcnow()
        base_date = user.plan_expires_at if user.plan_expires_at and user.plan_expires_at > now else now
        
        user.plan_type = plan_to_grant
        duration = ADVANCE_PLAN_DURATION_DAYS if plan_to_grant == 'advance' else PRO_PLAN_DURATION_DAYS
        user.plan_expires_at = base_date + timedelta(days=duration)
        db.session.commit()

        # P1-3: Notify user of renewal/extension
        from app.services.notification_service import notify_subscription_renewed
        notify_subscription_renewed(user, user.plan_expires_at.strftime('%d %b %Y'))

        return jsonify({
            'message': 'Payment successful. User upgraded to Pro.',
            'plan_expires_at': user.plan_expires_at.isoformat(),
        }), 200

    except Exception as e:
        logger.exception("Error verifying payment")
        db.session.rollback()
        return jsonify({'error': 'Failed to verify payment'}), 500


# ──────────────────────────────────────────────
# POST /api/billing/webhook  (P0-1)
# Razorpay server-to-server webhook — the single source of truth.
# ──────────────────────────────────────────────
@billing_bp.route('/webhook', methods=['POST'])
@limiter.exempt  # Webhooks come from Razorpay servers, not end-users
def razorpay_webhook():
    """
    Handles Razorpay webhook events.
    Verifies signature using RAZORPAY_WEBHOOK_SECRET, then processes:
      - payment.captured  → upgrade user to Pro
      - payment.failed    → mark payment as failed
    """
    # 1. Verify webhook signature
    webhook_secret = _get_webhook_secret()
    if not webhook_secret:
        logger.error("RAZORPAY_WEBHOOK_SECRET not configured")
        return jsonify({'error': 'Webhook not configured'}), 503

    webhook_signature = request.headers.get('X-Razorpay-Signature', '')
    webhook_body = request.get_data()

    expected_signature = hmac.new(
        webhook_secret.encode('utf-8'),
        webhook_body,
        hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(expected_signature, webhook_signature):
        logger.warning("Webhook signature mismatch")
        return jsonify({'error': 'Invalid signature'}), 400

    # 2. Parse the event
    try:
        event_data = request.get_json(force=True)
    except Exception:
        return jsonify({'error': 'Invalid JSON'}), 400

    event = event_data.get('event', '')
    payload = event_data.get('payload', {})
    payment_entity = payload.get('payment', {}).get('entity', {})

    razorpay_payment_id = payment_entity.get('id')
    razorpay_order_id = payment_entity.get('order_id')
    amount = payment_entity.get('amount', 0)
    currency = payment_entity.get('currency', 'INR')
    notes = payment_entity.get('notes', {})
    user_id_str = notes.get('user_id')

    plan_granted = notes.get('plan', 'pro')

    logger.info(f"Webhook event={event} payment_id={razorpay_payment_id} order_id={razorpay_order_id} plan={plan_granted}")

    if not razorpay_order_id:
        # Not a payment event we care about
        return jsonify({'status': 'ignored'}), 200

    # 3. Process by event type
    if event == 'payment.captured':
        # Idempotency: check if already processed
        existing = Payment.query.filter_by(razorpay_payment_id=razorpay_payment_id).first()
        if existing and existing.status == 'captured':
            return jsonify({'status': 'already_processed'}), 200

        # Find or create payment record
        payment = Payment.query.filter_by(razorpay_order_id=razorpay_order_id).first()
        if payment:
            payment.razorpay_payment_id = razorpay_payment_id
            payment.status = 'captured'
            payment.source = 'webhook'
            if not payment.plan_granted:
                payment.plan_granted = plan_granted
            plan_granted = payment.plan_granted
            payment.amount = amount
            payment.currency = currency
            payment.updated_at = datetime.utcnow()
        else:
            payment = Payment(
                user_id=int(user_id_str) if user_id_str else 0,
                razorpay_order_id=razorpay_order_id,
                razorpay_payment_id=razorpay_payment_id,
                amount=amount,
                currency=currency,
                status='captured',
                source='webhook',
                plan_granted=plan_granted,
                notes=str(notes),
            )
            db.session.add(payment)

        # Upgrade the user
        if user_id_str:
            user = User.query.get(int(user_id_str))
            if user:
                now = datetime.utcnow()
                base_date = user.plan_expires_at if user.plan_expires_at and user.plan_expires_at > now else now
                
                user.plan_type = plan_granted
                duration = ADVANCE_PLAN_DURATION_DAYS if plan_granted == 'advance' else PRO_PLAN_DURATION_DAYS
                user.plan_expires_at = base_date + timedelta(days=duration)

        db.session.commit()

        # P1-3: Notify user of renewal/extension
        if user_id_str and 'user' in locals() and user:
            from app.services.notification_service import notify_subscription_renewed
            notify_subscription_renewed(user, user.plan_expires_at.strftime('%d %b %Y'))

        return jsonify({'status': 'captured'}), 200

    elif event == 'payment.failed':
        payment = Payment.query.filter_by(razorpay_order_id=razorpay_order_id).first()
        if payment:
            payment.razorpay_payment_id = razorpay_payment_id
            payment.status = 'failed'
            payment.source = 'webhook'
            payment.updated_at = datetime.utcnow()
            db.session.commit()

        return jsonify({'status': 'failed_recorded'}), 200

    else:
        # Unhandled event — acknowledge to prevent retries
        return jsonify({'status': 'ignored', 'event': event}), 200


# ──────────────────────────────────────────────
# GET /api/billing/history — enriched payment history for current user
# ──────────────────────────────────────────────
@billing_bp.route('/history', methods=['GET'])
@jwt_required()
def payment_history():
    user_id = get_jwt_identity()
    user = User.query.get(int(user_id))
    if not user:
        return jsonify({'error': 'User not found'}), 404

    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    per_page = min(per_page, 50)  # cap

    query = Payment.query.filter_by(user_id=int(user_id)) \
        .order_by(Payment.created_at.desc())

    total = query.count()
    payments = query.offset((page - 1) * per_page).limit(per_page).all()

    # Summary stats
    from sqlalchemy import func
    total_spent = db.session.query(func.sum(Payment.amount)) \
        .filter_by(user_id=int(user_id), status='captured').scalar() or 0

    captured_count = Payment.query.filter_by(
        user_id=int(user_id), status='captured').count()

    return jsonify({
        'payments': [p.to_dict() for p in payments],
        'pagination': {
            'page': page,
            'per_page': per_page,
            'total': total,
            'pages': (total + per_page - 1) // per_page,
        },
        'summary': {
            'total_spent': total_spent,          # paise
            'total_spent_display': f"₹{total_spent / 100:,.2f}",
            'successful_payments': captured_count,
            'current_plan': user.plan_type,
            'plan_expires_at': user.plan_expires_at.isoformat() if user.plan_expires_at else None,
            'is_pro_active': user.is_pro_active,
        }
    }), 200


# ──────────────────────────────────────────────
# GET /api/billing/invoice/<payment_id> — download invoice PDF
# ──────────────────────────────────────────────
@billing_bp.route('/invoice/<int:payment_id>', methods=['GET'])
@jwt_required()
def download_invoice(payment_id):
    user_id = get_jwt_identity()
    payment = Payment.query.get(payment_id)

    if not payment:
        return jsonify({'error': 'Payment not found'}), 404

    if payment.user_id != int(user_id):
        return jsonify({'error': 'Unauthorized'}), 403

    if payment.status != 'captured':
        return jsonify({'error': 'Invoice available only for successful payments'}), 400

    user = User.query.get(int(user_id))
    if not user:
        return jsonify({'error': 'User not found'}), 404

    try:
        from app.services.invoice_service import generate_invoice_pdf
        pdf_bytes = generate_invoice_pdf(payment, user)

        from flask import Response
        invoice_filename = f"InferX_Invoice_{payment.id:06d}.pdf"
        return Response(
            pdf_bytes,
            mimetype='application/pdf',
            headers={
                'Content-Disposition': f'attachment; filename="{invoice_filename}"',
                'Content-Length': str(len(pdf_bytes)),
            }
        )
    except Exception as e:
        logger.exception("Error generating invoice PDF")
        return jsonify({'error': 'Failed to generate invoice'}), 500

