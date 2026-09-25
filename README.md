# InferX ML

## Overview

**InferX ML** is a no-code machine learning platform designed to simplify the end-to-end ML workflow for **structured, time-series, and image data**. The platform enables users to upload datasets, automatically train predictive models, and generate insights and predictions **without writing any machine learning code**.

InferX ML follows a **rule-based AutoML approach**, where the system automatically performs data profiling, preprocessing, model selection, training, evaluation, and explainability. Instead of deploying models as external services, predictions are executed internally and displayed directly through an interactive user interface, keeping the system **simple, efficient, and easy to use**.

The platform is built entirely on **open-source technologies**, including **Streamlit** for the no-code interface, **Pandas and scikit-learn** for data processing and model training, and **MinIO** for scalable object storage of datasets, models, and artifacts. InferX ML is optimized for **educational use, rapid prototyping, and small-to-medium scale analytics**, while being architecturally ready to scale to distributed systems in future iterations.

---

## Key Highlights

* 🚀 **No-code, user-friendly ML workflow**
* 📊 **Supports tabular (CSV/Excel), time-series, and image datasets**
* ⚙️ **AI-powered goal analysis** (Groq/Gemini integration)
* ⚖️ **Automated Data Auto-Balancing** (Detects skewed classes and applies SMOTE synthetic generation)
* ⚙️ **Automated data profiling, preprocessing, and model training**
* 📈 **Built-in model evaluation and explainability**
* 🧩 **Open-source, lightweight, and scalable-by-design**

---

# 🏗️ System Architecture (React + Flask + Streamlit + MinIO)

## High-Level Architecture

```
┌────────────────────────────────────────────┐
│              User Browser                  │
└───────────────┬───────────────┬────────────┘
                │               │
                ▼               ▼
┌────────────────────────┐  ┌────────────────────────┐
│     React Frontend     │  │    Streamlit Frontend  │
│   (Main Application)   │  │   (Prediction & ML UI) │
│                        │  │                        │
│ • Login / Dashboard    │  │ • Training progress    │
│ • Dataset management   │  │ • Predictions          │
│ • Job history          │  │ • Charts & insights    │
└───────────────┬────────┘  └───────────────┬────────┘
                │ REST API                  │ REST / Python
                ▼                           ▼
┌────────────────────────────────────────────┐
│              Flask Backend                 │
│          (Core Orchestrator API)           │
│                                            │
│ • Auth & user management                   │
│ • Dataset registration                     │
│ • AutoML workflow control                  │
│ • Prediction orchestration                 │
└───────────────┬───────────────┬────────────┘
                │               │
                ▼               ▼
┌────────────────────────┐  ┌────────────────────────┐
│      PostgreSQL        │  │        MinIO           │
│   (Metadata Store)     │  │  (Object Storage)      │
│                        │  │                        │
│ • Users                │  │ • Raw datasets         │
│ • Experiments          │  │ • Processed data       │
│ • Job status           │  │ • Trained models       │
│ • Configs              │  │ • Pipelines            │
└────────────────────────┘  │ • Metrics & artifacts  │
                             └────────────────────────┘
```

---

# 🔄 Flow Summary

```
React UI
   ↓
Flask API
   ↓
MinIO (store data)
   ↓
Pandas AutoML
   ↓
Model + Artifacts → MinIO
   ↓
Streamlit UI
   ↓
Predictions in Browser
```

---

# 🚀 Getting Started

## Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **Docker & Docker Compose** (for database and storage)
- **Git**

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/bharat3214/InferX-ML.git
cd InferX-ML
```

### 2. Set Up Environment Variables

```bash
cp .env.example .env
# Edit .env with your configuration if needed
```

---

## 🐳 Running with Docker (Recommended)

### Start Services

Choose the command based on your hardware:

**Option A: CPU Only (Standard)**
```bash
docker-compose up --build
```

**Option B: GPU Enabled (Requires NVIDIA GPU & Container Toolkit)**
```bash
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

#### Advanced: Manual Builds

You can also build individual images with custom arguments:

```bash
# Build backend for GPU manually
docker build -t inferx-backend-gpu \
  --build-arg BASE_IMAGE=nvidia/cuda:11.8.0-runtime-ubuntu22.04 \
  -f docker/Dockerfile.backend .
```

**Access Points:**
- 🌐 **Frontend (React)**: http://localhost:3000
- 🔌 **Backend API**: http://localhost:5000
- 📊 **Streamlit App**: http://localhost:8501
- 💾 **MinIO Console**: http://localhost:9001 (admin: minioadmin/minioadmin)

### Stop Services

```bash
docker-compose down
```

---

## 🛠️ Running Locally (Development)

### Step 1: Start Infrastructure Only

```bash
docker-compose up -d postgres redis minio
```

### Step 2: Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize database
python migrations.py init

# (Optional) Seed with demo data
python migrations.py seed

# Run backend server
python run.py
```

Backend runs at: http://localhost:5000

### Step 3: Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

Frontend runs at: http://localhost:3000

### Step 4: Streamlit App (Optional)

```bash
cd streamlit_app

# Install dependencies
pip install -r requirements.txt

# Run Streamlit
streamlit run app.py
```

Streamlit runs at: http://localhost:8501

### Step 5: Celery Worker (For Background Training)

```bash
cd backend

# In a new terminal
celery -A app.celery_app worker --loglevel=info
```

---

## 🧪 Running Tests

```bash
# Install test dependencies
pip install pytest

# Run all tests
pytest

# Run with coverage
pytest --cov=.
```

---

## 📁 Project Structure

```
InferX-ML/
├── backend/                 # Flask API Server
│   ├── app/
│   │   ├── models/         # Database models
│   │   ├── routes/         # API endpoints
│   │   ├── services/       # Business logic
│   │   └── tasks/          # Celery background tasks
│   ├── migrations.py       # Database management
│   └── run.py              # Entry point
│
├── frontend/               # React Dashboard
│   ├── src/
│   │   ├── pages/          # Page components
│   │   ├── components/     # Reusable UI components
│   │   ├── services/       # API client
│   │   └── store/          # State management
│   └── package.json
│
├── ml_engine/              # AutoML Core
│   ├── automl/
│   │   ├── tabular/        # Classification, Regression, Clustering
│   │   ├── timeseries/     # ARIMA, Prophet, LSTM
│   │   └── vision/         # Image Classification
│   ├── preprocessing/      # Data preprocessors
│   ├── explainability/     # SHAP explanations
│   └── packaging/          # Model packaging
│
├── streamlit_app/          # Prediction UI
├── docker/                 # Dockerfiles
├── tests/                  # Unit & Integration tests
└── docker-compose.yml      # Full stack orchestration
```

---

## 🔑 Demo Credentials

After running `python migrations.py seed`:

```
Email: demo@inferx.ml
Password: demo123
```

---

## 🛡️ Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FLASK_ENV` | Environment mode | development |
| `DATABASE_URL` | PostgreSQL connection | postgresql://postgres:postgres@localhost:5432/inferx_ml |
| `REDIS_URL` | Redis connection | redis://localhost:6379/0 |
| `MINIO_ENDPOINT` | MinIO server | localhost:9000 |
| `MINIO_ACCESS_KEY` | MinIO access key | minioadmin |
| `MINIO_SECRET_KEY` | MinIO secret key | minioadmin |
| `JWT_SECRET_KEY` | JWT signing key | your-jwt-secret-key |
| `GROQ_API_KEY` | Groq API key for AI analysis | *(required for AI features)* |
| `GEMINI_API_KEY` | Google Gemini API key (alternative) | *(optional)* |

---

## 📝 API Documentation

### Authentication
- `POST /api/auth/register` - Create new account
- `POST /api/auth/login` - Login & get token
- `GET /api/auth/me` - Get current user

### Datasets
- `GET /api/datasets` - List all datasets
- `POST /api/datasets/upload` - Upload dataset
- `GET /api/datasets/:id` - Get dataset details

### Training
- `POST /api/training/start` - Start training job
- `GET /api/training/:id/status` - Get job status

### Models
- `GET /api/models` - List trained models
- `GET /api/models/:id/schema` - Get prediction form schema

### Predictions
- `POST /api/predict/:modelId` - Make prediction
- `POST /api/predict/:modelId/explain` - Get explanation

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙋 Support

For issues or questions, please open an issue on GitHub.
