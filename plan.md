# 📋 InferX-ML: Complete Project Plan

> **No-Code AI/ML Platform** - Build predictive, forecasting, and computer vision models without writing code.

---

## 🎯 Project Overview

InferX-ML is a **no-code machine learning platform** that enables users to:
- Upload datasets (CSV, Excel, Images)
- Define goals in natural language
- Automatically train and evaluate ML models
- Get explainable predictions with auto-generated UI

The platform uses a **rule-based AutoML approach** where users don't manually select algorithms - the system decides the best model automatically.

---

## 🛠️ Technical Stack

### Frontend
| Technology | Purpose |
|------------|---------|
| **React.js** | Main application UI (dashboard, dataset management, job history) |
| **Streamlit** | ML-specific UI (training progress, predictions, charts) |
| **Tailwind CSS** | Styling framework |

### Backend
| Technology | Purpose |
|------------|---------|
| **Python 3.10+** | Core language |
| **Flask** | REST API & orchestration |
| **Celery** | Background task processing (model training) |
| **Redis** | Task queue & caching |

### Database & Storage
| Technology | Purpose |
|------------|---------|
| **PostgreSQL** | Metadata store (users, experiments, job status) |
| **MinIO** | Object storage (datasets, models, artifacts) |

### ML Libraries
| Library | Purpose |
|---------|---------|
| **Pandas** | Data manipulation |
| **NumPy** | Numerical operations |
| **scikit-learn** | ML algorithms (RF, XGBoost, Logistic Regression, K-Means, DBSCAN) |
| **Prophet** | Time-series forecasting |
| **statsmodels** | ARIMA models |
| **TensorFlow/Keras** | Deep learning (LSTM, MobileNet, EfficientNet) |
| **SHAP** | Model explainability |
| **Google Gemini API** | AI-powered goal analysis & target column detection |

### DevOps
| Technology | Purpose |
|------------|---------|
| **Docker** | Containerization |
| **Docker Compose** | Multi-service orchestration |
| **GitHub Actions** | CI/CD |

---

## 📊 Supported Data Types & Capabilities

### 1️⃣ Tabular Data (CSV/Excel)
| Task | Algorithms | Example Use Cases |
|------|------------|-------------------|
| **Classification** | Logistic Regression, Random Forest, XGBoost | Customer churn, spam detection, loan approval |
| **Regression** | Linear Regression, Random Forest, XGBoost | Price prediction, sales forecasting |
| **Clustering** | K-Means, DBSCAN | Customer segmentation, anomaly grouping |

**Supported Formats:** `.csv`, `.xls`, `.xlsx`

---

### 2️⃣ Time-Series Data
| Task | Algorithms | Example Use Cases |
|------|------------|-------------------|
| **Forecasting** | ARIMA, Prophet, LSTM | Sales prediction, demand forecasting |
| **Anomaly Detection** | Statistical methods | Fraud detection, sensor monitoring |
| **Trend Analysis** | Moving averages, decomposition | Business intelligence |

**Auto-Detection:** Platform automatically identifies timestamp columns by:
- Column names (`date`, `timestamp`, `time`, `datetime`)
- Value parsing (ISO format detection)

---

### 3️⃣ Image Data
| Task | Models | Example Use Cases |
|------|--------|-------------------|
| **Image Classification** | MobileNet, EfficientNet | Defect detection, product categorization |
| **Object Detection** | *(Future: YOLOv8)* | Inventory counting |

**Supported Formats:** `.jpg`, `.png`, ZIP folders organized by class

---

## 🔄 AutoML Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           USER JOURNEY                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. UPLOAD         2. DESCRIBE GOAL     3. AI ANALYSIS      4. TRAIN        │
│  ─────────         ────────────────     ─────────────       ───────         │
│  CSV/Excel/        Natural language     Gemini AI suggests  Automatic       │
│  Images            prompt               target column &     model training  │
│                                         problem type                        │
│                                                                              │
│  5. PREDICT        6. DOWNLOAD          7. DEPLOY           8. MONITOR       │
│  ─────────         ────────────────     ────────────────    ─────────        │
│  Auto-generated    Download model       Deploy model for    Track model      │
│  prediction UI     package (.zip)       production use      performance      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        INTERNAL AUTOML PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐   ┌──────────────┐   ┌───────────────┐   ┌────────────────┐   │
│  │  Gemini  │ → │   Problem    │ → │    Model      │ → │    Feature     │   │
│  │ Analyzer │   │   Detector   │   │   Selector    │   │   Engineer     │   │
│  └──────────┘   └──────────────┘   └───────────────┘   └────────────────┘   │
│       │                                                         │            │
│       │         ┌──────────────┐   ┌───────────────┐           │            │
│       │         │  Best Model  │ ← │   Evaluator   │ ← ────────┘            │
│       │         └──────────────┘   └───────────────┘                        │
│       │                │                                                     │
│       ▼                ▼                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    MODEL PACKAGING + ORDER GENERATION                │    │
│  │  ┌────────────────┬────────────────┬────────────────┬─────────────┐ │    │
│  │  │ Trained Model  │ Preprocessing  │ Streamlit UI   │  Gemini     │ │    │
│  │  │ (.pkl / .h5)   │   Pipeline     │  (Predictions) │  Reports    │ │    │
│  │  └────────────────┴────────────────┴────────────────┴─────────────┘ │    │
│  │  + Metadata (metrics, version, timestamp)                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Model Package Structure

When a model is trained, the platform creates a **self-contained package**:

```
📦 model_package/
├── model.pkl                    # Trained model (sklearn)
├── model.h5                     # OR deep learning model (Keras)
├── preprocessor.pkl             # Fitted preprocessing pipeline
├── feature_schema.json          # Column types, categories, ranges
├── ui_schema.json               # Auto-generated form schema
├── metadata.json                # Metrics, version, timestamp
├── requirements.txt             # Python dependencies
└── 📊 graphs/                   # Evaluation visualizations
    ├── confusion_matrix.png     # Classification error matrix
    ├── roc_curve.png            # Model AUC performance
    ├── residual_plot.png        # Regression error spread
    └── feature_importance.png   # SHAP/Tree importance charts
```

### JSON-based UI Schema Example

```json
{
  "model_name": "customer_churn_predictor",
  "version": "1.0.0",
  "target_column": "churn",
  "fields": [
    {
      "name": "age",
      "type": "number",
      "input_type": "slider",
      "min": 18,
      "max": 80,
      "default": 35
    },
    {
      "name": "monthly_charges",
      "type": "number",
      "input_type": "number",
      "min": 0,
      "max": 500
    },
    {
      "name": "contract_type",
      "type": "categorical",
      "input_type": "dropdown",
      "options": ["Month-to-month", "One year", "Two year"]
    },
    {
      "name": "internet_service",
      "type": "categorical",
      "input_type": "radio",
      "options": ["DSL", "Fiber optic", "No"]
    }
  ]
}
```

This schema enables **automatic UI generation** for predictions without any frontend coding.

---

## 🗂️ Project Directory Structure

```
InferX-ML/
│
├── 📁 frontend/                      # React Frontend
│   ├── public/
│   ├── src/
│   │   ├── components/               # Reusable UI components
│   │   ├── pages/                    # Page components
│   │   │   ├── Dashboard.jsx
│   │   │   ├── Datasets.jsx
│   │   │   ├── Training.jsx
│   │   │   └── Predictions.jsx
│   │   ├── services/                 # API service layer
│   │   ├── hooks/                    # Custom React hooks
│   │   └── utils/                    # Utility functions
│   ├── package.json
│   └── tailwind.config.js
│
├── 📁 backend/                       # Flask Backend
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py                 # Configuration
│   │   ├── models/                   # SQLAlchemy models
│   │   │   ├── user.py
│   │   │   ├── dataset.py
│   │   │   └── experiment.py
│   │   ├── routes/                   # API routes
│   │   │   ├── auth.py
│   │   │   ├── datasets.py
│   │   │   ├── training.py           # Includes /analyze-prompt endpoint
│   │   │   └── predictions.py
│   │   ├── services/                 # Business logic
│   │   │   ├── data_profiler.py
│   │   │   ├── problem_detector.py
│   │   │   ├── gemini_service.py     # AI-powered goal analysis (Groq/Gemini)
│   │   │   ├── minio_service.py
│   │   │   └── explainer.py
│   │   └── utils/                    # Utilities
│   ├── requirements.txt
│   └── run.py
│
├── 📁 ml_engine/                     # Core ML Engine
│   ├── automl/
│   │   ├── tabular/
│   │   │   ├── classifier.py
│   │   │   ├── regressor.py
│   │   │   └── clusterer.py
│   │   ├── timeseries/
│   │   │   ├── arima.py
│   │   │   ├── prophet.py
│   │   │   └── lstm.py
│   │   └── vision/
│   │       ├── classifier.py
│   │       └── models.py
│   ├── preprocessing/
│   │   ├── tabular_preprocessor.py
│   │   ├── timeseries_preprocessor.py
│   │   └── image_preprocessor.py
│   ├── explainability/
│   │   ├── shap_explainer.py
│   │   └── feature_importance.py
│   └── packaging/
│       ├── model_packager.py
│       └── ui_schema_generator.py
│
├── 📁 streamlit_app/                 # Streamlit UI
│   ├── app.py
│   ├── pages/
│   │   ├── training_status.py
│   │   ├── predictions.py
│   │   └── explainability.py
│   └── components/
│       └── dynamic_form.py           # JSON schema → Form
│
├── 📁 docker/                        # Docker configs
│   ├── Dockerfile.frontend
│   ├── Dockerfile.backend
│   ├── Dockerfile.streamlit
│   └── docker-compose.yml
│
├── 📁 tests/                         # Test suite
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── .env.example
├── .gitignore
├── README.md
└── plan.md
```

---

## 🚀 Development Phases

### Phase 1: Foundation (Week 1-2)
- [ ] Project setup & Docker configuration
- [ ] PostgreSQL & MinIO setup
- [ ] Flask backend skeleton with auth
- [ ] React frontend skeleton with routing
- [ ] Basic file upload to MinIO

### Phase 2: Data Pipeline (Week 3-4)
- [ ] Data Profiler (auto-detect data types, missing values, distributions)
- [ ] Problem Detector (classify task type: classification/regression/clustering/timeseries)
- [ ] Feature Engineer (auto preprocessing, encoding, scaling)
- [ ] Dataset management UI
- [ ] **Auto-Balancing (SMOTE)**: Detect extreme imbalance and prompt users to generate synthetic minority records.

### Phase 3: Tabular AutoML (Week 5-6)
- [ ] Model Selector (rule-based algorithm selection)
- [ ] Trainer (parallel training with Celery)
- [ ] Evaluator (metrics, cross-validation)
- [ ] Visualizer (generate & save confusion matrix, ROC curve, residual plots)
- [x] **Model Leaderboard**: Real-time evaluation of 8-10 models during training.
- [x] **Ensemble Super-Model**: Pro feature to combine Top 3 models into a Voting Classifier/Regressor.
- [ ] Best model selection logic

### Phase 4: Time-Series & Image (Week 7-8)
- [ ] Time-series pipeline (ARIMA, Prophet, LSTM)
- [ ] Image classification pipeline (MobileNet, EfficientNet)
- [ ] Training progress UI

### Phase 5: Explainability & Packaging (Week 9-10)
- [ ] SHAP integration
- [ ] Feature importance visualization
- [ ] Model packaging (pkl, h5, preprocessor, schemas)
- [ ] JSON-based UI schema generator

### Phase 6: Prediction & Deploy (Week 11-12)
- [ ] Dynamic prediction form (from UI schema)
- [ ] Batch prediction (file upload)
- [ ] Single prediction (form input)
- [ ] Model download/export
- [ ] Streamlit prediction UI

### Phase 7: Polish & Testing (Week 13-14)
- [ ] End-to-end testing
- [ ] Performance optimization
- [ ] Documentation
- [ ] Demo videos

### Phase 8: AI App Builder (Week 15-16)
- [ ] Integrate **anomalyco/opencode** agent
- [ ] Create Sandboxed environment (Docker) for agent execution
- [ ] Implement "Text-to-App" UI (users describe app needs)
- [ ] Backend service to route prompts to opencode agent
- [ ] Deployment pipeline for generated frontend+backend apps
- [ ] Zip download or live preview of generated apps

---

## 📐 API Endpoints

### Authentication
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/register` | User registration |
| POST | `/api/auth/login` | User login |
| POST | `/api/auth/logout` | User logout |

### Datasets
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/datasets` | List all datasets |
| POST | `/api/datasets/upload` | Upload new dataset |
| GET | `/api/datasets/{id}` | Get dataset details |
| GET | `/api/datasets/{id}/profile` | Get data profile (stats, types) |
| DELETE | `/api/datasets/{id}` | Delete dataset |

### Training
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/training/start` | Start training job |
| POST | `/api/training/analyze-prompt` | **NEW:** AI-powered target column suggestion |
| GET | `/api/training/{job_id}/status` | Get training status |
| GET | `/api/training/{job_id}/logs` | Get training logs |
| POST | `/api/training/{job_id}/cancel` | Cancel training |

### Models
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/models` | List all trained models |
| GET | `/api/models/{id}` | Get model details |
| GET | `/api/models/{id}/download` | Download model package |
| GET | `/api/models/{id}/schema` | Get UI schema |
| DELETE | `/api/models/{id}` | Delete model |

### Predictions
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/predict/{model_id}` | Single prediction |
| POST | `/api/predict/{model_id}/batch` | Batch prediction (file) |
| GET | `/api/predict/{model_id}/explain` | Get prediction explanation |

### AI App Builder
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/builder/generate` | Generate full app from prompt |
| GET | `/api/builder/{project_id}/status` | Check generation progress |
| GET | `/api/builder/{project_id}/download` | Download generated code |
| POST | `/api/builder/{project_id}/deploy` | Deploy generated app |


---

## 🧪 Testing Strategy

### Unit Tests
- Individual pipeline components (profiler, preprocessor, trainer)
- Utility functions
- API route handlers

### Integration Tests
- Full pipeline tests (upload → train → predict)
- Database operations
- MinIO storage operations

### End-to-End Tests
- Complete user flows with browser automation
- Different data types and scenarios

---

## 📊 Success Metrics

| Metric | Target |
|--------|--------|
| Model training success rate | > 95% |
| Average training time (tabular, <10k rows) | < 2 minutes |
| UI response time | < 200ms |
| Prediction latency | < 500ms |

---

## 🔐 Security Considerations

- [ ] JWT-based authentication
- [ ] Rate limiting on API endpoints
- [ ] Input validation & sanitization
- [ ] Secure file upload handling
- [ ] Environment variable management
- [ ] CORS configuration

---

## 📚 References

- [scikit-learn Documentation](https://scikit-learn.org/)
- [Prophet Documentation](https://facebook.github.io/prophet/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [MinIO Documentation](https://min.io/docs/)
- [Flask Documentation](https://flask.palletsprojects.com/)

---

## 📝 Notes

- **Rule-based AutoML**: Users do NOT select models manually. The system decides based on:
  - Data type (tabular/timeseries/image)
  - Problem type (classification/regression/clustering)
  - Dataset size
  - Feature characteristics

- **🆕 AI-Powered Goal Analysis**: Users can describe their goal in natural language (e.g., "I want to predict inventory reorder quantity"). Gemini AI analyzes the dataset and suggests:
  - Target column
  - Problem type (classification/regression/timeseries)
  - Preprocessing recommendations
  - Environmental factors to consider

- **🆕 Order Management**: After predictions, the system generates AI-powered order recommendations:
  - Automatic order quantity calculation
  - Risk assessment
  - Human approval workflow (pending → approved → fulfilled)
  - Future: Supplier API integration for automated ordering

* ⚖️ **Automated Data Auto-Balancing** (Detects skewed classes and applies SMOTE synthetic generation)
* 🏆 **Dynamic Model Leaderboard** (Watch models compete in real-time while training)
* 🔮 **One-Click Ensemble** (Combine Top 3 performers into a robust Super-Model)
* ⚙️ **Automated data profiling, preprocessing, and model training**
* 📈 **Built-in model evaluation and explainability**: Marked as **VERY IMPORTANT** - every prediction should have clear explanations.

---

*Last Updated: January 10, 2026*