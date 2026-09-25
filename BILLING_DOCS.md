# InferX-ML Billing & Subscription Architecture

This document outlines the complete end-to-end architecture of the subscription and billing system in InferX-ML. 

## 1. Plan Structure
The platform utilizes a dynamic 3-tier subscription model:

- **Free Tier (Default):**
  - **Cost:** ₹0
  - **Limits:** 20 API requests/minute.
  - **Features:** 
    - Train up to **1 model** per 24 hours.
    - Predict API limited to **100 calls/day**.
    - Standard CPU training (no GPU support).
    - Datasets:
      - Synthetic Data Generation: **Locked**
      - Kaggle Dataset Search: **Limited to 5 results**
    - Maximum dataset upload size: **50 MB**.
    - Access to basic ML algorithms (4 algorithms). Advanced models (like YOLOv8, XGBoost, LightGBM) are disabled.
  
- **Pro Tier (Monthly):**
  - **Cost:** ₹499 per 30 days (`PRO_PLAN_AMOUNT_PAISE=49900`).
  - **Limits:** 200 API requests/minute.
  - **Features:** Unlocks all 10+ algorithms, including YOLOv8 Deep Learning and XGBoost. Unlimited training jobs.

- **Advance Tier (Yearly):**
  - **Cost:** ₹4999 per 365 days (`ADVANCE_PLAN_AMOUNT_PAISE=499900`).
  - **Limits:** 200 API requests/minute (priority queues).
  - **Features:** 
    - **Unlimited** model training.
    - **Unlimited** Predict API calls.
    - Priority access to **GPU acceleration**.
    - Datasets:
      - Synthetic Data Generation: **Unlocked**
      - Kaggle Dataset Search: **Unlimited results**
    - Maximum dataset upload size: **2 GB**.
    - Best value. Includes everything in Pro, plus early access features, custom integrations, and dedicated support.

---

## 2. Razorpay Payment Workflow

The application uses **Razorpay** as the payment gateway. The integration runs in two phases: the Client-side Checkout and the Server-side Webhook.

### Client-Side Checkout (`SettingsPage.jsx` & `PricingPage.jsx`)
1. **Initiation:** When the user clicks "Upgrade to Pro" or "Advance", the frontend calls `POST /api/billing/create-order` passing the `plan` name.
2. **Order Creation:** The backend calculates the cost, requests an Order ID from Razorpay, creates a `Payment` record in the DB with `status='created'`, and returns the `key_id` and `order_id`.
3. **Modal:** The frontend opens the Razorpay modal (`checkout.js`). The user completes the payment.
4. **Verification:** Upon success, Razorpay returns a `razorpay_signature`. The frontend sends this to `POST /api/billing/verify-payment`. The backend verifies the signature using `hmac` and your secret key. If valid, the user's `plan_type` and `plan_expires_at` are immediately updated.

### Server-Side Webhook (`POST /api/billing/webhook`)
Webhooks act as the single source of truth to handle edge cases (like a user closing the browser before verification).
- Razorpay fires `payment.captured` or `payment.failed` to the webhook URL.
- The webhook verifies the `X-Razorpay-Signature` using `RAZORPAY_WEBHOOK_SECRET`.
- If `payment.captured`, it idempotently upgrades the user's plan and logs the payment in the DB.

---

## 3. Rate Limiting Logic (`app/__init__.py`)
Rate limiting is enforced globally using `Flask-Limiter` backed by a Redis datastore.

- **Smart IP Resolution:** The limiter parses `X-Forwarded-For` and `X-Real-Ip` to ensure users behind proxies (like Nginx) are tracked individually.
- **Dynamic JWT Rules:** A function checks the JWT token on every request. If `user.plan_type` is `pro` or `advance`, the global limit dynamically expands from `20 per minute` to `200 per minute`.
- **Strict Endpoints:** Sensitive endpoints like `/create-order` and `/verify-payment` are strictly limited to `10 per minute` to prevent abuse.
- **Alerts:** If a user hits a rate limit, the API returns a `429 Too Many Requests` error and automatically dispatches an in-app notification warning the user of the throttling.

---

## 4. Background Tasks & Plan Expiration

### Celery Beat Task (`app/tasks/billing_tasks.py`)
- **Daily Downgrades:** A periodic Celery beat task runs automatically at midnight (`crontab(minute=0, hour=0)`).
- **Execution:** It scans the `users` table for anyone where `plan_expires_at < now()` and `plan_type IN ('pro', 'advance')`.
- **Action:** It forcefully downgrades them to the `free` plan and dispatches an in-app notification informing them their subscription has expired.

### CLI Cron Job (`cron/downgrade_expired_plans.py`)
For setups not utilizing Celery, a raw Python script exists. This can be scheduled via traditional Linux `crontab`. It performs the exact same expiry checks and database downgrades as the Celery task.

---

## 5. Invoice Generation (`app/services/invoice_service.py`)
Whenever a payment status reaches `captured`:
- The user can navigate to the **Settings Page -> Billing tab** to view their history.
- Clicking **Download Invoice** hits `GET /api/billing/invoice/<payment_id>`.
- The backend utilizes the `fpdf2` library to dynamically generate a PDF receipt detailing the transaction ID, Date, Amount (in INR), and Plan details. The PDF bytes are streamed directly to the browser for download.

---

## 6. Environment Variables Setup
To run the billing system in production, your `.env` must contain:

```env
# MOCK_PAYMENTS=true bypasses Razorpay entirely for local dev
MOCK_PAYMENTS=false

# Plan Configurations (amount in Paise, meaning 49900 = ₹499)
PRO_PLAN_AMOUNT_PAISE=49900
PRO_PLAN_DURATION_DAYS=30
ADVANCE_PLAN_AMOUNT_PAISE=499900
ADVANCE_PLAN_DURATION_DAYS=365

# Razorpay Keys
RAZORPAY_KEY_ID=your_test_key_id
RAZORPAY_KEY_SECRET=your_test_key_secret
RAZORPAY_WEBHOOK_SECRET=your_webhook_secret
```

---

## 7. Model Execution Guards (`app/routes/training.py`)
Before a model training job begins, the backend enforces two strict checks based on the user's plan:

### 1. Daily Training Limits
The backend queries the `experiments` database to count the user's jobs over the last 24 hours:
```python
if user and user.plan_type == 'free':
    yesterday = datetime.utcnow() - timedelta(days=1)
    recent_trainings = Experiment.query.filter(
        Experiment.user_id == user_id,
        Experiment.created_at >= yesterday
    ).count()
    if recent_trainings >= 1:
        return jsonify({'error': 'Free plan limit reached (1 training per day max). Upgrade to Pro for unlimited training.'}), 403
```
- **Free Tier:** Locked to **1 training job per rolling 24-hour period**.
- **Pro & Advance Tier:** This check is bypassed completely, enabling **unlimited training**.

### 2. Algorithm Restrictions
Even if the Free tier user is within their 1-job-per-day limit, compute-heavy algorithms are restricted:
```python
is_pro = user.plan_type in ('pro', 'advance')
if algorithm in ['xgboost', 'yolov8'] and not is_pro:
    return jsonify({'error': 'Pro or Advance plan required'}), 403
```
This ensures strict enforcement of billing constraints at the exact moment a compute-heavy task is requested.
