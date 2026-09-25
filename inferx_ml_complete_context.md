# InferX-ML — Complete Project Context Document

> **Purpose**: This document captures every technical detail of the InferX-ML codebase so that any AI (or developer) reading it can fully understand the entire system without needing to read the source code.

---

## 1. What is InferX-ML?

InferX-ML is a **no-code, self-hosted machine learning platform** that automates the entire ML lifecycle for **structured (tabular), time-series, and image data**. A user uploads a dataset, describes their goal in plain English, and the system automatically:

1. Profiles the data (detects types, distributions, missing values, class imbalance)
2. Detects the ML problem type (classification, regression, clustering, timeseries, image)
3. Preprocesses (imputation, encoding, scaling, SMOTE balancing)
4. Trains 8–13 models simultaneously (a live leaderboard shows models competing)
5. Evaluates with cross-validation, generates confusion matrices / ROC curves / residual plots
6. Explains predictions with SHAP values
7. Packages the best model with a preprocessor, JSON-based UI schema, and a loader script
8. Auto-generates a Streamlit prediction interface from the UI schema — zero code needed

Think of it as an **open-source alternative to Google AutoML / AWS SageMaker Canvas**, optimized for education, rapid prototyping, and small-to-medium analytics.

---

## 2. Tech Stack

| Layer | Technology | Role |
|-------|-----------|------|
| **Frontend (Main App)** | React 18 + Vite | Dashboard, dataset management, training config, models, predictions, billing, settings |
| **Frontend (ML UI)** | Streamlit | Live training progress, interactive predictions, charts, SHAP explanations |
| **Backend API** | Flask (Python 3.10+) | REST API, orchestration, auth, billing |
| **Task Queue** | Celery + Redis | Background model training, dataset profiling, plan expiry cron |
| **Database** | PostgreSQL 15 | Users, datasets, experiments, training jobs, payments, notification preferences |
| **Object Storage** | MinIO (S3-compatible) | Raw datasets, processed data, trained models, pipelines, metrics, artifacts |
| **ML Libraries** | scikit-learn, XGBoost, LightGBM, Prophet, statsmodels (ARIMA), TensorFlow/Keras (LSTM, MobileNet, EfficientNet), SHAP | Core ML algorithms |
| **AI Services** | Groq (llama-3.3-70b) → Gemini (gemini-2.5-flash) → Ollama (llama3.2 local) | Natural language goal analysis, target column suggestion, synthetic data generation |
| **Email** | Node.js Express + Nodemailer | Transactional emails (OTP verification, password reset, training notifications) |
| **Payments** | Razorpay | Subscription billing (Pro/Advance plans) |
| **Auth** | JWT (Flask-JWT-Extended) + OAuth (Google, GitHub) | Authentication & authorization |
| **Rate Limiting** | Flask-Limiter + Redis | Per-user dynamic rate limits based on plan tier |
| **Containerization** | Docker + Docker Compose | Full-stack orchestration (8 services) |

---

## 3. Project Directory Structure

```
InferX-ML/
├── backend/                          # Flask API Server
│   ├── app/
│   │   ├── __init__.py               # App factory, extensions (db, jwt, limiter), schema sync
│   │   ├── celery_app.py             # Celery configuration
│   │   ├── commands.py               # Flask CLI commands (seed, init)
│   │   ├── config.py                 # Config class (DB, JWT, MinIO, Redis, OAuth, upload limits)
│   │   ├── models/
│   │   │   ├── user.py               # User model (auth, billing, OAuth fields)
│   │   │   ├── dataset.py            # Dataset metadata model
│   │   │   ├── experiment.py         # Experiment + TrainingJob models
│   │   │   ├── payment.py            # Razorpay payment records
│   │   │   └── notification_preference.py  # Per-user notification toggles
│   │   ├── routes/
│   │   │   ├── auth.py               # Register, login, OTP verify, password reset
│   │   │   ├── oauth.py              # Google/GitHub OAuth flows
│   │   │   ├── datasets.py           # Upload, list, delete, profile, preview, Kaggle search/download, synthetic generation, AI schema
│   │   │   ├── training.py           # Start training, analyze-prompt (AI), status, logs, thinking logs, cancel, ensemble
│   │   │   ├── models.py             # List models, download package, get schema, get graphs
│   │   │   ├── predictions.py        # Single predict, batch predict, explain
│   │   │   ├── billing.py            # Create Razorpay order, verify payment, webhook, invoice PDF download, plan status
│   │   │   ├── notifications.py      # Get/update notification preferences
│   │   │   └── system.py             # Health check
│   │   ├── services/
│   │   │   ├── gemini_service.py      # AI service (Groq → Gemini → Ollama fallback chain)
│   │   │   ├── minio_service.py       # MinIO storage operations (upload/download files, bytes, JSON, model packages)
│   │   │   ├── data_profiler.py       # Auto data profiling (types, distributions, missing values, stats)
│   │   │   ├── problem_detector.py    # Rule-based ML problem type detection
│   │   │   ├── explanation_engine.py  # AI-powered human-readable model explanations
│   │   │   ├── ai_insight_generator.py # AI insight/report generation
│   │   │   ├── invoice_service.py     # PDF invoice generation (fpdf2)
│   │   │   ├── mailer_service.py      # HTTP client to mailer microservice
│   │   │   └── notification_service.py # Dispatches email notifications (training complete/failed, rate limit)
│   │   ├── tasks/
│   │   │   ├── training_tasks.py      # Celery task: train_model_task, profile_dataset_task
│   │   │   └── billing_tasks.py       # Celery beat: daily plan expiry downgrade
│   │   ├── utils/                     # Backend utilities
│   │   └── static/                    # Static assets
│   ├── cron/                          # Standalone cron scripts (alternative to Celery beat)
│   ├── migrations/                    # DB migrations
│   ├── migrations.py                  # Migration CLI (init, seed, etc.)
│   ├── requirements.txt               # Python dependencies
│   └── run.py                         # Entry point (runs Flask dev server on port 5000)
│
├── frontend/                          # React Dashboard (Vite)
│   ├── src/
│   │   ├── App.jsx                    # Router: public routes (landing, login, auth callback) + protected routes (all others)
│   │   ├── main.jsx                   # React entry point (BrowserRouter, QueryClientProvider)
│   │   ├── index.css                  # Global styles
│   │   ├── queryClient.js             # React Query client config
│   │   ├── pages/
│   │   │   ├── LandingPage.jsx/css    # Public marketing landing page
│   │   │   ├── Login.jsx/css          # Login + Register + OTP Verify + Forgot Password (multi-step form)
│   │   │   ├── AuthCallback.jsx       # OAuth callback handler (Google/GitHub)
│   │   │   ├── Dashboard.jsx/css      # Overview dashboard (stats, recent experiments, quick actions)
│   │   │   ├── Datasets.jsx/css       # Dataset management (upload, list, preview, Kaggle search, synthetic generation)
│   │   │   ├── DatasetViewer.jsx/css  # Detailed dataset viewer (column stats, profiling)
│   │   │   ├── Training.jsx/css       # Training config + live training monitor + model leaderboard + ensemble
│   │   │   ├── Models.jsx/css         # Model library (list, view metrics, download, graphs)
│   │   │   ├── Predictions.jsx/css    # Prediction interface (single + batch)
│   │   │   ├── PricingPage.jsx/css    # Pricing tiers (Free/Pro/Advance) + Razorpay checkout
│   │   │   └── SettingsPage.jsx/css   # User settings (profile, billing history, notifications, security)
│   │   ├── components/
│   │   │   ├── Layout.jsx/css         # Sidebar nav + header + main content layout
│   │   │   ├── LearningTimeline.jsx/css # Visual training progress timeline
│   │   │   ├── TrainingMonitor.jsx/css  # Live training status component
│   │   │   ├── MockPaymentModal.jsx/css # Dev-mode payment simulation
│   │   │   └── ui/                    # Reusable UI primitives
│   │   ├── services/
│   │   │   └── api.js                 # Axios instance + all API methods (datasets, training, models, predictions, orders)
│   │   └── store/
│   │       ├── authStore.js           # Zustand auth store (login, register, verify, OAuth, logout) — persisted to localStorage
│   │       └── trainingStore.js       # Zustand training state store
│   ├── package.json                   # Dependencies: react, react-router-dom, axios, zustand, @tanstack/react-query, recharts, etc.
│   └── vite.config.js                 # Vite config (proxy /api → localhost:5000)
│
├── ml_engine/                         # Core ML Engine (Python package)
│   ├── __init__.py
│   ├── automl/
│   │   ├── tabular/
│   │   │   ├── classifier.py          # 10 classifiers: LogReg, RF, GB, SVM, KNN, DecisionTree, AdaBoost, ExtraTrees, XGBoost, LightGBM
│   │   │   ├── regressor.py           # 13 regressors: Linear, Ridge, Lasso, ElasticNet, RF, GB, SVR, KNN, DecisionTree, AdaBoost, ExtraTrees, XGBoost, LightGBM
│   │   │   └── clusterer.py           # K-Means (k=2-10) + DBSCAN
│   │   ├── timeseries/
│   │   │   ├── arima.py               # ARIMA forecasting
│   │   │   ├── prophet.py             # Facebook Prophet forecasting
│   │   │   └── lstm.py                # LSTM deep learning forecasting
│   │   └── vision/
│   │       ├── classifier.py          # Image classification (MobileNet, EfficientNet via transfer learning)
│   │       └── detector.py            # Object detection (future: YOLOv8)
│   ├── preprocessing/
│   │   ├── tabular_preprocessor.py    # Auto preprocessing: type detection, imputation (median/most_frequent), scaling (RobustScaler), encoding (OneHot/Ordinal), variance threshold, missing indicator columns
│   │   ├── timeseries_preprocessor.py # Time-series specific preprocessing
│   │   ├── image_preprocessor.py      # Image preprocessing (resize, normalize, augment)
│   │   └── detection_preprocessor.py  # Object detection preprocessing
│   ├── explainability/
│   │   ├── shap_explainer.py          # SHAP explainer (Tree/Linear/Kernel/Deep), feature importance, waterfall plots, summary plots, text explanations
│   │   └── __init__.py
│   ├── packaging/
│   │   ├── model_packager.py          # Packages model + preprocessor + feature_schema.json + ui_schema.json + metadata.json + target_classes.json + model_info.json + loader.py + streamlit_app.py + graphs/
│   │   └── __init__.py
│   ├── utils/
│   │   ├── gpu_utils.py               # GPU auto-detection for XGBoost/LightGBM CUDA params
│   │   └── visualizer.py              # Generates classification/regression evaluation graphs (confusion matrix, ROC, residuals, feature importance)
│   └── requirements.txt
│
├── streamlit_app/                     # Streamlit Prediction UI
│   ├── app.py                         # Main app (~46KB): connects to Flask API, renders prediction forms from ui_schema.json, displays training results, charts, SHAP explanations
│   ├── components/
│   │   └── __init__.py
│   ├── pages/
│   │   └── __init__.py
│   └── requirements.txt
│
├── mailer/                            # Email Microservice (Node.js)
│   ├── index.js                       # Express server on port 4000 with Nodemailer. Templates: verification OTP, password reset OTP, training_completed, training_failed, rate_limit_warning, billing_updated
│   └── package.json                   # Dependencies: express, nodemailer
│
├── docker/                            # Dockerfiles
│   ├── Dockerfile.backend             # Python 3.10 + pip install requirements
│   ├── Dockerfile.frontend            # Node 18 + npm install + npm run dev
│   ├── Dockerfile.mailer              # Node 18 + npm install + node index.js
│   └── Dockerfile.streamlit           # Python 3.10 + pip install + streamlit run
│
├── tests/                             # Test Suite
│   ├── unit/                          # Unit tests
│   └── integration/                   # Integration tests
│
├── docker-compose.yml                 # 8 services: postgres, redis, minio, backend, celery, celery-beat, mailer, frontend, streamlit
├── docker-compose.gpu.yml             # GPU override (NVIDIA runtime for backend + celery)
├── .env.example                       # All environment variables
├── plan.md                            # Full project plan with phases, API spec, model package spec
├── BILLING_DOCS.md                    # Billing architecture documentation
├── PREPROCESSING.md                   # Preprocessing pipeline documentation
├── pytest.ini                         # Pytest config
└── README.md                          # Setup & usage guide
```

---

## 4. Database Schema (PostgreSQL)

### 4.1 `users` Table

| Column | Type | Notes |
|--------|------|-------|
| `id` | SERIAL PRIMARY KEY | |
| `email` | VARCHAR(120) UNIQUE NOT NULL | Indexed |
| `username` | VARCHAR(80) UNIQUE NOT NULL | |
| `password_hash` | VARCHAR(256) NULLABLE | Null for OAuth-only users |
| `is_active` | BOOLEAN DEFAULT TRUE | |
| `email_verified` | BOOLEAN DEFAULT FALSE | Must verify OTP after registration |
| `verification_code_hash` | VARCHAR(256) | Hashed 6-digit OTP |
| `verification_code_expires_at` | TIMESTAMP | Default: 10 minutes from generation |
| `verification_sent_at` | TIMESTAMP | Cooldown tracking (60s between resends) |
| `password_reset_code_hash` | VARCHAR(256) | Hashed reset OTP |
| `password_reset_code_expires_at` | TIMESTAMP | |
| `password_reset_sent_at` | TIMESTAMP | |
| `last_login_at` | TIMESTAMP | |
| `oauth_provider` | VARCHAR(20) | 'google' or 'github' |
| `oauth_provider_id` | VARCHAR(256) | Provider's user ID |
| `avatar_url` | VARCHAR(512) | From OAuth provider |
| `plan_type` | VARCHAR(20) DEFAULT 'free' | 'free', 'pro', 'advance' |
| `plan_expires_at` | TIMESTAMP | Pro: +30 days, Advance: +365 days |
| `razorpay_customer_id` | VARCHAR(100) | |
| `razorpay_subscription_id` | VARCHAR(100) | |
| `created_at` | TIMESTAMP | |
| `updated_at` | TIMESTAMP | |

**Relationships**: `datasets` (1:many), `experiments` (1:many), `payments` (1:many), `notification_prefs` (1:1)

**Unique Constraint**: `(oauth_provider, oauth_provider_id)`

**Property**: `is_pro_active` → returns True if `plan_type` is pro/advance AND `plan_expires_at > now()`

### 4.2 `datasets` Table

| Column | Type | Notes |
|--------|------|-------|
| `id` | SERIAL PRIMARY KEY | |
| `name` | VARCHAR(255) NOT NULL | User-provided name |
| `description` | TEXT | |
| `file_path` | VARCHAR(512) NOT NULL | MinIO path: `user_{id}/dataset_{id}/{filename}` |
| `file_type` | VARCHAR(20) NOT NULL | 'csv', 'xlsx', 'xls', 'image_zip' |
| `file_size` | BIGINT | Bytes |
| `data_type` | VARCHAR(20) | 'tabular', 'timeseries', 'image' (auto-detected) |
| `num_rows` | INTEGER | |
| `num_columns` | INTEGER | |
| `column_info` | JSON | Column names, types, stats |
| `profile_status` | VARCHAR(20) DEFAULT 'pending' | 'pending', 'processing', 'completed', 'failed' |
| `profile_data` | JSON | Full profiling results |
| `user_id` | INTEGER FK → users(id) | |
| `created_at` / `updated_at` | TIMESTAMP | |

### 4.3 `experiments` Table

| Column | Type | Notes |
|--------|------|-------|
| `id` | SERIAL PRIMARY KEY | |
| `name` | VARCHAR(255) NOT NULL | |
| `description` | TEXT | |
| `problem_type` | VARCHAR(50) | 'binary_classification', 'multiclass_classification', 'regression', 'clustering', 'timeseries', 'image_classification' |
| `target_column` | VARCHAR(255) | Null for clustering |
| `goal_description` | TEXT | Natural language goal from user |
| `config` | JSON | Training config: `auto_balance`, `n_estimators`, `max_depth`, `epochs`, etc. |
| `status` | VARCHAR(20) DEFAULT 'created' | 'created', 'training', 'completed', 'failed' |
| `best_model_id` | VARCHAR(255) | MinIO path to best model |
| `best_score` | FLOAT | Best CV score |
| `best_model_name` | VARCHAR(100) | e.g., 'random_forest', 'xgboost' |
| `results` | JSON | Full training results including `model_package_path` |
| `thinking_logs` | TEXT | Live timestamped logs for real-time UI: `[HH:MM:SS] message\n` |
| `explanation_data` | JSON | Rich human-friendly explanation data |
| `user_id` | INTEGER FK → users(id) | |
| `dataset_id` | INTEGER FK → datasets(id) | |
| `created_at` / `updated_at` / `completed_at` | TIMESTAMP | |

### 4.4 `training_jobs` Table

| Column | Type | Notes |
|--------|------|-------|
| `id` | SERIAL PRIMARY KEY | |
| `model_name` | VARCHAR(100) NOT NULL | e.g., 'RandomForest', 'XGBoost' |
| `model_params` | JSON | Hyperparameters used |
| `status` | VARCHAR(20) DEFAULT 'pending' | 'pending', 'running', 'completed', 'failed' |
| `progress` | FLOAT DEFAULT 0.0 | 0–100 |
| `metrics` | JSON | accuracy, f1, rmse, r2, etc. |
| `model_path` | VARCHAR(512) | MinIO path |
| `logs` | TEXT | |
| `error_message` | TEXT | |
| `experiment_id` | INTEGER FK → experiments(id) | |
| `created_at` / `started_at` / `completed_at` | TIMESTAMP | |

### 4.5 `payments` Table

| Column | Type | Notes |
|--------|------|-------|
| `id` | SERIAL PRIMARY KEY | |
| `user_id` | INTEGER FK → users(id) | Indexed |
| `razorpay_order_id` | VARCHAR(100) NOT NULL | Indexed |
| `razorpay_payment_id` | VARCHAR(100) UNIQUE | |
| `razorpay_signature` | VARCHAR(256) | |
| `amount` | INTEGER NOT NULL | In paise (49900 = ₹499) |
| `currency` | VARCHAR(10) DEFAULT 'INR' | |
| `status` | VARCHAR(30) DEFAULT 'created' | 'created', 'captured', 'failed', 'refunded' |
| `provider` | VARCHAR(30) DEFAULT 'razorpay' | |
| `source` | VARCHAR(20) DEFAULT 'client' | 'client' or 'webhook' |
| `plan_granted` | VARCHAR(20) | 'pro' or 'advance' |
| `notes` | TEXT | |
| `created_at` / `updated_at` | TIMESTAMP | |

### 4.6 `notification_preferences` Table

| Column | Type | Default |
|--------|------|---------|
| `id` | SERIAL PRIMARY KEY | |
| `user_id` | INTEGER FK → users(id) UNIQUE | |
| `training_completed` | BOOLEAN | TRUE |
| `training_failed` | BOOLEAN | TRUE |
| `dataset_uploaded` | BOOLEAN | FALSE |
| `api_rate_limit` | BOOLEAN | TRUE |
| `weekly_report` | BOOLEAN | FALSE |
| `security_alerts` | BOOLEAN | TRUE |
| `product_updates` | BOOLEAN | FALSE |
| `billing_updates` | BOOLEAN | TRUE |

---

## 5. API Endpoints (Complete)

**Base URL**: `http://localhost:5000/api`
**Auth**: JWT Bearer token in `Authorization` header (except public routes)

### 5.1 Auth (`/api/auth`)

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/register` | Register (email, username, password) → sends OTP email | No |
| POST | `/login` | Login → returns JWT + user (blocks if unverified) | No |
| POST | `/verify-email` | Verify OTP (email, otp) → returns JWT + user | No |
| POST | `/resend-verification` | Resend OTP (60s cooldown) | No |
| POST | `/forgot-password` | Send password reset OTP | No |
| POST | `/reset-password` | Reset with OTP (email, otp, password) | No |
| GET | `/me` | Get current user info | Yes |
| GET | `/google/login` | Initiate Google OAuth | No |
| GET | `/google/callback` | Google OAuth callback → redirect with JWT | No |
| GET | `/github/login` | Initiate GitHub OAuth | No |
| GET | `/github/callback` | GitHub OAuth callback → redirect with JWT | No |

### 5.2 Datasets (`/api/datasets`)

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/` | List all user's datasets | Yes |
| POST | `/upload` | Upload file (multipart: csv/xlsx/zip) → triggers background profiling | Yes |
| GET | `/:id` | Get dataset details + profile | Yes |
| DELETE | `/:id` | Delete dataset + MinIO files | Yes |
| GET | `/:id/profile` | Get detailed profiling data | Yes |
| GET | `/:id/preview` | Get first N rows of data | Yes |
| GET | `/:id/detect-headers` | AI-detect column headers/types | Yes |
| POST | `/:id/set-headers` | Set custom column headers | Yes |
| POST | `/use-sample` | Use a built-in sample dataset | Yes |
| GET | `/kaggle/search` | Search Kaggle datasets (limited by plan) | Yes |
| POST | `/kaggle/download` | Download Kaggle dataset to MinIO | Yes |
| POST | `/generate-synthetic` | Generate synthetic tabular data | Yes |
| POST | `/ai/schema` | AI generates dataset schema from prompt | Yes |
| POST | `/ai/generate` | AI generates full synthetic dataset | Yes |

### 5.3 Training (`/api/training`)

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/start` | Start training job → Celery task (enforces plan limits) | Yes |
| POST | `/analyze-prompt` | AI analyzes dataset with natural language goal → suggests target column, problem type | Yes |
| GET | `/:id/status` | Get experiment status + results | Yes |
| GET | `/:id/logs` | Get training logs | Yes |
| GET | `/:id/thinking` | Get live thinking logs (real-time UI) | Yes |
| POST | `/:id/cancel` | Cancel running training | Yes |
| POST | `/:id/ensemble` | Create ensemble of top 3 models (Pro feature) | Yes |

### 5.4 Models (`/api/models`)

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/` | List all trained models (completed experiments) | Yes |
| GET | `/:id` | Get model details + metadata | Yes |
| DELETE | `/:id` | Delete model + MinIO package | Yes |
| GET | `/:id/download` | Download model package as ZIP | Yes |
| GET | `/:id/schema` | Get UI schema (for prediction form) | Yes |
| GET | `/:id/graphs` | Get evaluation graphs (confusion matrix, ROC, etc.) | Yes |

### 5.5 Predictions (`/api/predict`)

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/:modelId` | Single prediction (JSON input) | Yes |
| POST | `/:modelId/batch` | Batch prediction (file upload) | Yes |
| POST | `/:modelId/explain` | Single prediction + SHAP explanation | Yes |

### 5.6 Billing (`/api/billing`)

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/plan` | Get current plan status | Yes |
| POST | `/create-order` | Create Razorpay order for Pro/Advance | Yes |
| POST | `/verify-payment` | Verify Razorpay payment signature → upgrade plan | Yes |
| POST | `/webhook` | Razorpay webhook (payment.captured / payment.failed) | No (signature verified) |
| GET | `/history` | Get payment history | Yes |
| GET | `/invoice/:paymentId` | Download PDF invoice | Yes |

### 5.7 Notifications (`/api/notifications`)

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/preferences` | Get notification toggle states | Yes |
| PUT | `/preferences` | Update notification toggles | Yes |

### 5.8 System (`/api/system`)

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/health` | Health check (exempt from rate limiting) | No |

---

## 6. ML Pipeline — Detailed Flow

### 6.1 Problem Detection (`ProblemDetector`)

When a user starts training, the system determines the problem type via rule-based logic:

1. **No target column** → `clustering` (K-Means, DBSCAN)
2. **Boolean target** → `binary_classification`
3. **Target with 2 unique values** → `binary_classification` (confidence: 0.95)
4. **Target with ≤20 unique values AND low unique ratio (≤0.2)** → `multiclass_classification`
5. **Non-numeric target with ≤100 unique values** → `multiclass_classification`
6. **Numeric target with many unique continuous values** → `regression`
7. **Integer target with ≤50 unique values and ≤10% unique ratio** → `multiclass_classification` (coded classes)
8. **Date column detected AND goal contains forecast terms** → `timeseries`
9. **Identifier-like target** (high unique ratio + name ends with `_id`) → warning
10. **Fallback** → `regression` with low confidence

The system also checks for **column name hints** (e.g., 'churn', 'fraud', 'status', 'category', 'label') to boost classification confidence.

### 6.2 AI Goal Analysis (`AIService` / `gemini_service.py`)

Users can describe their goal in natural language. The system uses a 3-provider fallback chain:

1. **Groq** (cloud, fast) — model: `llama-3.3-70b-versatile`
2. **Gemini** (cloud, Google) — model: `gemini-2.5-flash`
3. **Ollama** (local, offline) — model: `llama3.2`

Fallback triggers on: rate limits, quota exhaustion, timeouts, connection errors, invalid/missing API keys.

The AI analyzes the dataset columns + user's natural language prompt and returns:
- Suggested target column
- Problem type (classification/regression/timeseries)
- Preprocessing recommendations
- Reasoning

### 6.3 Preprocessing (`TabularPreprocessor`)

Industrial-grade preprocessing pipeline using sklearn `ColumnTransformer`:

1. **Column type detection**: Numeric vs. categorical (auto-detect based on dtype + cardinality threshold of 10)
2. **High-missing column drop**: Columns with >60% missing values are dropped
3. **Numeric pipeline**: `SimpleImputer(median)` → `RobustScaler` → `VarianceThreshold(0.0)` — also adds missing indicator columns
4. **Categorical pipeline**: `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown='ignore')` — also adds missing indicator columns
5. **Target encoding**: `LabelEncoder` for string/categorical targets
6. **SMOTE auto-balancing**: If `auto_balance=True` in config AND classification task, applies SMOTE synthetic oversampling for skewed classes

### 6.4 Model Training

Training runs as a **Celery background task** (`train_model_task`):

#### Classification (10 algorithms)
- Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN, Decision Tree, AdaBoost, Extra Trees, XGBoost*, LightGBM*
- Scoring: 5-fold cross-validation on `accuracy`
- Test metrics: accuracy, f1_weighted, precision_weighted, recall_weighted, roc_auc (binary)
- Best model: highest mean CV accuracy

#### Regression (13 algorithms)
- Linear Regression, Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting, SVR, KNN, Decision Tree, AdaBoost, Extra Trees, XGBoost*, LightGBM*
- Scoring: 5-fold cross-validation on `r2`
- Test metrics: RMSE, MAE, R², MAPE
- Best model: highest mean CV R²

#### Clustering (K-Means + DBSCAN)
- K-Means: tries k=2 to k=10
- DBSCAN: eps=0.5, min_samples=5
- Scoring: silhouette_score, calinski_harabasz, davies_bouldin
- Best model: highest silhouette score

#### Time-Series (3 algorithms)
- ARIMA (statsmodels)
- Prophet (Facebook Prophet)
- LSTM (TensorFlow/Keras)

#### Image (Transfer Learning)
- MobileNet, EfficientNet (TensorFlow/Keras)
- Object Detection: detector module (future: YOLOv8)

*XGBoost/LightGBM: Auto-detect GPU (CUDA) and use `tree_method='gpu_hist'` / `device='gpu'` if available.

### 6.5 Evaluation & Visualization (`ModelVisualizer`)

After training, the system generates evaluation graphs:
- **Classification**: Confusion matrix, ROC curve, feature importance
- **Regression**: Residual plot, predicted vs. actual, feature importance
- Graphs are saved as PNG files in the model package under `graphs/`

### 6.6 Explainability (`SHAPExplainer`)

SHAP (SHapley Additive exPlanations) integration:
- **Explainer types**: Tree (RF, XGBoost, LightGBM), Linear (LogReg, Ridge), Kernel (SVM, KNN — slower), Deep (neural networks)
- **Dataset explain**: Mean absolute SHAP values → ranked feature importance with normalized scores
- **Single prediction explain**: Per-feature contribution (value, SHAP value, direction: positive/negative) sorted by absolute impact
- **Visualizations**: Waterfall plot (single prediction), summary bar plot, beeswarm plot — all rendered as base64 PNG strings
- **Text explanation**: Auto-generated human-readable text: "The prediction was primarily influenced by: 1. **age** (value: 45.00) increased the prediction by 0.1234"

### 6.7 Model Packaging (`ModelPackager`)

After training, the best model is packaged into a self-contained directory:

```
model_package/
├── model.pkl                  # Trained model (joblib)
├── preprocessor.pkl           # Fitted preprocessing pipeline (joblib)
├── feature_schema.json        # Column types, categories, min/max, defaults
├── ui_schema.json             # Auto-generated form schema for Streamlit prediction UI
├── metadata.json              # Name, experiment_id, problem_type, target_column, best_model, best_score, training_results, packaged_at
├── target_classes.json        # Class label mapping (classification only)
├── model_info.json            # Headline/summary of trained model
├── loader.py                  # Python script to load and use the model
├── streamlit_app.py           # Auto-generated Streamlit prediction app
└── graphs/                    # Evaluation charts
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── residual_plot.png
    └── feature_importance.png
```

The **`ui_schema.json`** enables automatic UI generation. Example:
```json
{
  "model_name": "customer_churn_predictor",
  "target_column": "churn",
  "fields": [
    { "name": "age", "type": "number", "input_type": "slider", "min": 18, "max": 80, "default": 35 },
    { "name": "contract_type", "type": "categorical", "input_type": "dropdown", "options": ["Month-to-month", "One year", "Two year"] }
  ]
}
```

The entire package is uploaded to MinIO under: `models/user_{id}/experiment_{id}/`

---

## 7. Frontend (React) — Detailed Pages

### Routing (App.jsx)

| Route | Component | Auth Required |
|-------|-----------|---------------|
| `/landing` | LandingPage | No (redirects to `/` if logged in) |
| `/login` | Login | No (redirects to `/` if logged in) |
| `/auth/callback` | AuthCallback | No (handles OAuth redirect) |
| `/` | Dashboard | Yes |
| `/datasets` | Datasets | Yes |
| `/datasets/:datasetId/view` | DatasetViewer | Yes |
| `/training` | Training | Yes |
| `/models` | Models | Yes |
| `/predictions/:modelId?` | Predictions | Yes |
| `/pricing` | PricingPage | Yes |
| `/settings` | SettingsPage | Yes |

### State Management

- **Zustand** (`authStore.js`): Auth state (user, token, isAuthenticated). Persisted to `localStorage` under key `inferx-auth`. Actions: login, register, verifyEmail, resendVerification, forgotPassword, resetPassword, handleOAuthCallback, logout, checkAuth.
- **Zustand** (`trainingStore.js`): Training state (active experiments, polling).
- **React Query** (`@tanstack/react-query`): Server state caching for datasets, models, etc. Cache cleared on logout.

### API Client (`api.js`)

Axios instance with:
- Base URL: `VITE_API_URL` or `/api`
- Timeout: 30s (some endpoints up to 120s)
- **Request interceptor**: Reads JWT from localStorage, sets `Authorization: Bearer <token>`
- **Response interceptor**: On 401, clears auth and redirects to `/login`
- Exports: `datasetsApi`, `trainingApi`, `modelsApi`, `predictionsApi`, `ordersApi`

---

## 8. MinIO Storage Layout

Three buckets, auto-created on startup:

| Bucket | Purpose | Path Pattern |
|--------|---------|-------------|
| `datasets` | Raw uploaded files | `user_{userId}/dataset_{datasetId}/{filename}` |
| `models` | Trained model packages | `user_{userId}/experiment_{experimentId}/{file}` |
| `artifacts` | Misc artifacts | Various |

### MinIO Service Capabilities

- Upload: file, bytes, stream, JSON
- Download: file, bytes, JSON
- Presigned URLs: GET (download) and PUT (upload), default 1-hour expiry
- Delete: single object or prefix-based bulk delete
- List objects with prefix filtering
- Helper methods: `upload_dataset()`, `upload_model_package()`, `download_model_package()`

---

## 9. Billing System

### 3-Tier Subscription Model

| Tier | Price | Rate Limit | Training | Algorithms | Upload | Special |
|------|-------|------------|----------|-----------|--------|---------|
| **Free** | ₹0 | 20 req/min | 1 per 24h | 4 basic | 50 MB | Kaggle limited to 5 results, no synthetic gen |
| **Pro** | ₹499/month | 200 req/min | Unlimited | 10+ (XGBoost, YOLOv8) | — | Ensemble super-model |
| **Advance** | ₹4,999/year | 200 req/min (priority) | Unlimited | All | 2 GB | GPU access, synthetic data gen, unlimited Kaggle |

### Payment Flow (Razorpay)

1. Frontend calls `POST /api/billing/create-order` with plan name
2. Backend creates Razorpay order, saves `Payment` record (status: 'created')
3. Frontend opens Razorpay checkout modal
4. On success, frontend sends signature to `POST /api/billing/verify-payment`
5. Backend verifies HMAC signature → upgrades `user.plan_type` and `plan_expires_at`
6. Razorpay webhook (`POST /api/billing/webhook`) acts as source of truth for edge cases

### Plan Enforcement

- **Rate limiting**: Dynamic — Flask-Limiter checks JWT on every request; free=20/min, pro/advance=200/min
- **Training limit**: Free users: backend queries experiments table for jobs in last 24h; blocks if ≥1
- **Algorithm restriction**: XGBoost/YOLOv8 require pro/advance plan
- **Daily downgrade cron**: Celery beat task at midnight checks `plan_expires_at < now()` and downgrades to free

### Invoice Generation

`fpdf2` generates PDF invoices with transaction ID, date, amount (INR), and plan details. Streamed as download.

---

## 10. Authentication System

### Email/Password Registration Flow

1. User submits email, username, password
2. Backend hashes password (Werkzeug), creates user with `email_verified=False`
3. Generates 6-digit OTP, hashes it (SHA-256), stores hash + expiry (10 min)
4. Sends OTP email via mailer microservice
5. User enters OTP → backend verifies hash → sets `email_verified=True` → returns JWT

### OTP Security

- OTP stored as hash (not plaintext)
- 10-minute expiry (`OTP_EXPIRY_MINUTES`)
- 60-second resend cooldown (`OTP_RESEND_COOLDOWN_SECONDS`)
- Separate flows for email verification and password reset

### OAuth (Google & GitHub)

- Uses Authlib for OAuth integration
- Callback creates/updates user with `oauth_provider` and `oauth_provider_id`
- Auto-verified (no OTP needed)
- `password_hash` is nullable for OAuth-only users
- Redirects to frontend `/auth/callback?token=...&user=...`

### JWT Configuration

- Access token: 24-hour expiry
- Refresh token: 30-day expiry
- Secret: `JWT_SECRET_KEY` from env

---

## 11. Notification System

### Email Templates (Mailer Microservice)

| Template | Trigger | Content |
|----------|---------|---------|
| `verification` | Registration | 6-digit OTP with expiry |
| `password_reset` | Forgot password | 6-digit OTP with expiry |
| `training_completed` | Model training finishes | Success notice with model name |
| `training_failed` | Model training fails | Error details |
| `rate_limit_warning` | User hits rate limit | Warning with upgrade suggestion |
| `billing_updated` | Plan upgrade/downgrade | New plan details |

### Notification Preferences

Each user has toggle preferences (stored in `notification_preferences` table). The backend checks preferences before dispatching emails.

---

## 12. Docker Compose Services (8 total)

| Service | Image | Port | Depends On |
|---------|-------|------|------------|
| `postgres` | postgres:15-alpine | 5432 | — |
| `redis` | redis:7-alpine | 6379 | — |
| `minio` | minio/minio:latest | 9000 (API), 9001 (Console) | — |
| `backend` | Custom (Dockerfile.backend) | 5000 | postgres, redis, minio, mailer |
| `celery` | Same as backend | — | backend, redis |
| `celery-beat` | Same as backend | — | redis, celery |
| `mailer` | Custom (Dockerfile.mailer) | 4000 | — |
| `frontend` | Custom (Dockerfile.frontend) | 3000 | backend |
| `streamlit` | Custom (Dockerfile.streamlit) | 8501 | backend |

**Volumes**: `postgres_data`, `minio_data` (persistent)
**GPU override**: `docker-compose.gpu.yml` adds NVIDIA runtime to backend + celery

---

## 13. Environment Variables (Complete)

| Variable | Description | Default |
|----------|-------------|---------|
| `SECRET_KEY` | Flask secret key | dev-secret-key |
| `JWT_SECRET_KEY` | JWT signing key | jwt-secret-key |
| `OTP_EXPIRY_MINUTES` | OTP validity | 10 |
| `OTP_RESEND_COOLDOWN_SECONDS` | Resend cooldown | 60 |
| `DATABASE_URL` | PostgreSQL connection | postgresql://postgres:postgres@localhost:5432/inferx_ml |
| `REDIS_URL` | Redis connection | redis://localhost:6379/0 |
| `MINIO_ENDPOINT` | MinIO server | localhost:9000 |
| `MINIO_ACCESS_KEY` | MinIO access | minioadmin |
| `MINIO_SECRET_KEY` | MinIO secret | minioadmin |
| `MINIO_SECURE` | Use HTTPS | false |
| `GROQ_API_KEY` | Groq cloud API key | (required for AI features) |
| `GEMINI_API_KEY` | Google Gemini API key | (optional, fallback) |
| `OLLAMA_URL` | Local Ollama endpoint | http://host.docker.internal:11434 |
| `OLLAMA_MODEL` | Ollama model | llama3.2 |
| `RAZORPAY_KEY_ID` | Razorpay key | (required for billing) |
| `RAZORPAY_KEY_SECRET` | Razorpay secret | (required for billing) |
| `RAZORPAY_WEBHOOK_SECRET` | Webhook signature secret | (required) |
| `MOCK_PAYMENTS` | Skip Razorpay for dev | false |
| `PRO_PLAN_AMOUNT_PAISE` | Pro plan price | 49900 (₹499) |
| `PRO_PLAN_DURATION_DAYS` | Pro plan duration | 30 |
| `ADVANCE_PLAN_AMOUNT_PAISE` | Advance plan price | 499900 (₹4999) |
| `ADVANCE_PLAN_DURATION_DAYS` | Advance plan duration | 365 |
| `SMTP_HOST` / `SMTP_PORT` / `SMTP_USER` / `SMTP_PASS` / `SMTP_FROM` | Email SMTP config | — |
| `MAILER_URL` | Mailer microservice URL | http://mailer:4000 |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | Google OAuth | (optional) |
| `GITHUB_CLIENT_ID` / `GITHUB_CLIENT_SECRET` | GitHub OAuth | (optional) |
| `FRONTEND_URL` | Frontend URL for redirects | http://localhost:5173 |
| `VITE_API_URL` | Frontend API base URL | http://localhost:5000/api |
| `KAGGLE_USERNAME` / `KAGGLE_KEY` | Kaggle API credentials | (optional) |

---

## 14. Key Architectural Patterns

### 14.1 App Factory Pattern
Flask uses `create_app()` factory in `app/__init__.py`. Extensions (db, jwt, limiter, cors, migrate) are initialized at module level but bound to the app in the factory.

### 14.2 Schema Auto-Sync
On startup, `_sync_billing_schema()`, `_sync_auth_schema()`, and `_sync_oauth_schema()` run `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` statements to backfill new columns without requiring formal migrations.

### 14.3 Dynamic Rate Limiting
`get_user_rate_limit()` is called on every request. It optionally verifies the JWT, looks up the user's plan, and returns "200 per minute" for paid users or "20 per minute" for free users.

### 14.4 Background Task Architecture
- Training and profiling run as Celery tasks
- Each task creates its own Flask app context (`create_app()`) since Celery workers are separate processes
- Progress is reported via `task.update_state(state='PROGRESS', meta={...})`
- Thinking logs are written to `experiment.thinking_logs` for real-time frontend polling

### 14.5 MinIO as Central Storage
All files (datasets, models, artifacts) are stored in MinIO, never on the local filesystem. This makes the system horizontally scalable.

### 14.6 Singleton MinIO Service
`get_minio_service()` returns a module-level singleton to avoid creating multiple MinIO clients.

### 14.7 AI Fallback Chain
The AI service tries Groq → Gemini → Ollama. If a provider fails with a rate limit, network error, or auth error, it falls through to the next. If all fail, it returns a deterministic fallback result.

### 14.8 Model-Agnostic Prediction
The prediction system loads `model.pkl` + `preprocessor.pkl` + `ui_schema.json` from MinIO, transforms input through the preprocessor, runs the model, and returns results — works for any sklearn-compatible model.

---

## 15. Demo Credentials

After running `python migrations.py seed`:
```
Email: demo@inferx.ml
Password: demo123
```

---

## 16. Running the Project

### Docker (Full Stack)
```bash
cp .env.example .env          # Configure environment variables
docker-compose up --build     # Starts all 8+ services
```

### Local Development
```bash
# 1. Start infrastructure
docker-compose up -d postgres redis minio

# 2. Backend
cd backend && python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python migrations.py init && python migrations.py seed
python run.py                 # → http://localhost:5000

# 3. Frontend
cd frontend && npm install && npm run dev  # → http://localhost:3000

# 4. Celery Worker
cd backend && celery -A app.celery_app worker --loglevel=info

# 5. Streamlit (optional)
cd streamlit_app && pip install -r requirements.txt && streamlit run app.py  # → http://localhost:8501
```

---

*This document was auto-generated from the InferX-ML codebase on 2026-09-13.*
