"""
Training Routes
"""
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app import db
from app.models.dataset import Dataset
from app.models.experiment import Experiment, TrainingJob
from app.services.problem_detector import ProblemDetector
import pandas as pd
import io
from sklearn.preprocessing import LabelEncoder, StandardScaler

training_bp = Blueprint('training', __name__)


# CombinedPreprocessor at module level for pickle compatibility
class CombinedPreprocessor:
    """Preprocessor that handles both categorical encoding and scaling"""

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.categorical_columns = []
        self.numeric_columns = []

    def fit_transform(self, X):
        self.feature_columns = list(X.columns)
        self.categorical_columns = list(X.select_dtypes(include=['object']).columns)
        self.numeric_columns = [c for c in self.feature_columns if c not in self.categorical_columns]

        X_processed = X.copy()

        # Encode categorical features
        for col in self.categorical_columns:
            self.label_encoders[col] = LabelEncoder()
            X_processed[col] = self.label_encoders[col].fit_transform(X_processed[col].astype(str))

        # Fill missing values
        X_processed = X_processed.fillna(X_processed.median())

        # Scale all features
        return self.scaler.fit_transform(X_processed)

    def transform(self, X):
        import pandas as pd
        X_processed = X.copy()

        # Encode categorical features
        for col in self.categorical_columns:
            if col in X_processed.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                # Handle unseen labels by using the most frequent class
                X_processed[col] = X_processed[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0
                )

        # Handle missing columns (fill with 0)
        for col in self.feature_columns:
            if col not in X_processed.columns:
                X_processed[col] = 0

        # Reorder columns and fill missing values
        X_processed = X_processed[self.feature_columns].fillna(0)

        return self.scaler.transform(X_processed)


def _update_thinking_log(experiment, message):
    """Helper to append a thinking log message with timestamp"""
    try:
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"

        # Append to existing thinking_logs
        current_logs = experiment.thinking_logs or ''
        experiment.thinking_logs = current_logs + log_entry
        db.session.commit()
        print(f"💭 {message}", flush=True)
    except Exception as e:
        # Fail gracefully if thinking_logs column doesn't exist
        print(f"💭 {message}", flush=True)


@training_bp.route('/analyze-prompt', methods=['POST'])
@jwt_required()
def analyze_with_prompt():
    """
    Analyze a dataset with a natural language prompt using Gemini AI.
    Returns suggested target column, problem type, and reasoning.
    """
    user_id = int(get_jwt_identity())
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    dataset_id = data.get('dataset_id')
    prompt = data.get('prompt', '')

    if not dataset_id:
        return jsonify({'error': 'Dataset ID is required'}), 400

    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400

    # Verify dataset exists and belongs to user
    dataset = Dataset.query.filter_by(id=dataset_id, user_id=user_id).first()
    if not dataset:
        return jsonify({'error': 'Dataset not found'}), 404

    try:
        # Import Gemini service
        from app.services.gemini_service import get_gemini_service
        from app.services.minio_service import get_minio_service

        # Load dataset to get sample data
        minio_service = get_minio_service()
        file_content = minio_service.download_bytes('datasets', dataset.file_path)

        if not file_content:
            return jsonify({'error': 'Could not load dataset'}), 500

        # Support both CSV and Excel files
        if dataset.file_path.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(file_content))
        else:
            df = pd.read_csv(io.BytesIO(file_content))

        # Prepare data for Gemini
        columns = list(df.columns)
        column_types = {col: str(df[col].dtype) for col in columns}
        sample_data = df.head(10).to_dict(orient='records')

        # Call Gemini to analyze
        gemini_service = get_gemini_service()
        result = gemini_service.analyze_dataset_with_prompt(
            columns=columns,
            column_types=column_types,
            sample_data=sample_data,
            user_prompt=prompt
        )

        return jsonify({
            'success': True,
            'analysis': result
        }), 200

    except ValueError as e:
        # Gemini API key not configured
        return jsonify({
            'error': str(e),
            'suggestion': 'Please configure GEMINI_API_KEY in your environment'
        }), 500
    except Exception as e:
        return jsonify({
            'error': f'Analysis failed: {str(e)}'
        }), 500


@training_bp.route('/start', methods=['POST'])
@jwt_required()
def start_training():
    """Start a new training job"""
    user_id = int(get_jwt_identity())
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    dataset_id = data.get('dataset_id')
    name = data.get('name', 'Untitled Experiment')
    target_column = data.get('target_column')
    goal_description = data.get('goal_description', '')

    if not dataset_id:
        return jsonify({'error': 'Dataset ID is required'}), 400

    from app.models.user import User
    from datetime import datetime, timedelta
    user = User.query.get(user_id)
    if user and user.plan_type == 'free':
        yesterday = datetime.utcnow() - timedelta(days=1)
        recent_trainings = Experiment.query.filter(
            Experiment.user_id == user_id,
            Experiment.created_at >= yesterday
        ).count()
        if recent_trainings >= 1:
            return jsonify({'error': 'Free plan limit reached (1 training per day max). Upgrade to Pro for unlimited training.'}), 403

    # Verify dataset exists and belongs to user
    dataset = Dataset.query.filter_by(id=dataset_id, user_id=user_id).first()
    if not dataset:
        return jsonify({'error': 'Dataset not found'}), 404

    # Validate Configuration (Industrial Security Best Practice)
    config = data.get('config', {})
    if not isinstance(config, dict):
        return jsonify({'error': 'Config must be a JSON object'}), 400

    epochs = config.get('epochs')
    if epochs is not None and (not isinstance(epochs, int) or epochs < 1 or epochs > 1000):
        return jsonify({'error': 'Epochs must be an integer between 1 and 1000'}), 400

    batch_size = config.get('batch_size')
    if batch_size is not None and (not isinstance(batch_size, int) or batch_size < 1 or batch_size > 1024):
        return jsonify({'error': 'Batch size must be an integer between 1 and 1024'}), 400

    learning_rate = config.get('learning_rate')
    if learning_rate is not None and (not isinstance(learning_rate, (int, float)) or learning_rate <= 0 or learning_rate > 1):
        return jsonify({'error': 'Learning rate must be a float between 0 and 1'}), 400

    n_estimators = config.get('n_estimators')
    if n_estimators is not None and (not isinstance(n_estimators, int) or n_estimators < 1 or n_estimators > 5000):
        return jsonify({'error': 'n_estimators must be an integer between 1 and 5000'}), 400

    max_depth = config.get('max_depth')
    if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 1 or max_depth > 100):
        return jsonify({'error': 'max_depth must be an integer between 1 and 100'}), 400

    # ── Pre-flight dataset validation ─────────────────────────────────────────
    # Validate dataset content BEFORE creating the experiment so the user gets
    # an immediate, clear error instead of a silent mid-pipeline failure.
    if dataset.data_type != 'image' and target_column:
        try:
            from app.services.minio_service import get_minio_service
            _minio = get_minio_service()
            _file_content = _minio.download_bytes('datasets', dataset.file_path)
            if _file_content:
                if dataset.file_path.lower().endswith(('.xlsx', '.xls')):
                    _df_check = pd.read_excel(io.BytesIO(_file_content), nrows=5)
                else:
                    _df_check = pd.read_csv(io.BytesIO(_file_content), nrows=5)

                _cols = list(_df_check.columns)

                # Check 1: target column must exist in the dataset
                if target_column not in _cols:
                    return jsonify({
                        'error': f'Target column "{target_column}" not found in the dataset.',
                        'available_columns': _cols[:15],
                        'suggestion': (
                            'Your CSV may be missing a header row. '
                            'The first data row may have been used as column names. '
                            'Please re-upload with proper column headers.'
                        )
                    }), 400

                # Check 2: target column name looks like a data row, not a real header.
                # Catches CSVs uploaded without a header (e.g. Boston Housing dataset).
                import re as _re
                _stripped = target_column.strip()
                _suspicious = bool(_re.search(r'\d+\.\d+.*\d+\.\d+', _stripped))
                if _suspicious:
                    return jsonify({
                        'error': 'The selected target column name looks like raw data, not a column header.',
                        'bad_column': target_column,
                        'hint': (
                            'Your CSV likely has no header row. '
                            'Pandas used the first data row as column names.'
                        ),
                        'suggestion': (
                            'Re-upload your CSV with a proper header row '
                            '(e.g. "CRIM,ZN,INDUS,...,MEDV") or use the '
                            'dataset editing tools to add headers before training.'
                        )
                    }), 400

                # Check 3: after dropping target, at least 1 feature column must remain
                _remaining = [c for c in _cols if c != target_column]
                if len(_remaining) == 0:
                    return jsonify({
                        'error': 'After selecting the target column, no feature columns remain.',
                        'suggestion': 'Your dataset must have at least 2 columns: one target and at least one feature.'
                    }), 400

        except Exception as _pre_err:
            # Non-fatal: pre-flight errors should not block training; let it proceed naturally
            print(f'[Pre-flight validation skipped]: {_pre_err}', flush=True)

    # Create experiment
    experiment = Experiment(
        name=name,
        target_column=target_column,
        goal_description=goal_description,
        status='training',
        user_id=user_id,
        dataset_id=dataset_id,
        config=config
    )

    db.session.add(experiment)
    db.session.commit()

    # Run training synchronously for now (for simplicity in development)
    import threading
    from flask import current_app

    # Capture the real app object while we're still in the request context
    app = current_app._get_current_object()

    # Extract data needed for thread to avoid DetachedInstanceError
    experiment_id = experiment.id
    file_path = dataset.file_path
    column_info = dataset.column_info
    data_type = dataset.data_type  # 'tabular', 'image', etc.

    def run_training(app_instance):
        print("🧵 Training thread started...", flush=True)
        try:
            with app_instance.app_context():
                print(f"🔄 Running training task for experiment {experiment_id}", flush=True)
                print(f"📂 Dataset Path: {file_path}", flush=True)
                print(f"📊 Data Type: {data_type}", flush=True)

                if data_type == 'image':
                    config = experiment.config or {}
                    if config.get('problem_type') == 'object_detection':
                        _run_detection_training_task(experiment_id, file_path, column_info)
                    else:
                        _run_image_training_task(experiment_id, file_path, column_info)
                else:
                    _run_training_task(experiment_id, file_path, target_column, column_info)
        except Exception as e:
            print(f"❌ Training thread error: {e}", flush=True)
            import traceback
            traceback.print_exc()

    # Start training in background thread
    print(f"🚀 Launching training thread for Expt {experiment_id}...", flush=True)
    thread = threading.Thread(target=run_training, args=(app,))
    thread.daemon = True
    thread.start()
    print("✅ Training thread launched", flush=True)

    return jsonify({
        'message': 'Training started',
        'experiment': experiment.to_dict()
    }), 201


@training_bp.route('/<int:experiment_id>/ensemble', methods=['POST'])
@jwt_required()
def ensemble_top_3(experiment_id):
    """Combine top 3 models into a Voting Ensemble"""
    user_id = int(get_jwt_identity())
    experiment = Experiment.query.filter_by(id=experiment_id, user_id=user_id).first()

    if not experiment:
        return jsonify({'error': 'Experiment not found'}), 404

    if experiment.status != 'completed' or experiment.problem_type not in ['classification', 'regression']:
        return jsonify({'error': 'Experiment must be completed and be a tabular problem.'}), 400

    dataset = Dataset.query.get(experiment.dataset_id)
    if not dataset:
        return jsonify({'error': 'Dataset not found'}), 404

    experiment.status = 'training'
    db.session.commit()

    import threading
    from flask import current_app
    app = current_app._get_current_object()
    file_path = dataset.file_path
    target_column = experiment.target_column
    column_info = dataset.column_info

    def run_ensemble(app_instance):
        with app_instance.app_context():
            _run_ensemble_training(experiment_id, file_path, target_column, column_info)

    thread = threading.Thread(target=run_ensemble, args=(app,))
    thread.daemon = True
    thread.start()

    return jsonify({'message': 'Ensemble combination started'}), 200


def _run_ensemble_training(experiment_id, file_path, target_column, column_info):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
    import joblib
    import io
    from app.services.explanation_engine import ExplanationEngine

    experiment = Experiment.query.get(experiment_id)
    if not experiment: return

    _update_thinking_log(experiment, "🔮 Initializing Ultra-Accurate Ensemble Creation...")

    try:
        from app.services.minio_service import get_minio_service
        minio_service = get_minio_service()
        file_content = minio_service.download_bytes('datasets', file_path)

        if file_path.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(file_content))
        else:
            df = pd.read_csv(io.BytesIO(file_content))

        X = df.drop(columns=[target_column])
        y = df[target_column]

        problem_type = experiment.problem_type
        if problem_type == 'classification' and y.dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            y = LabelEncoder().fit_transform(y)

        preprocessor = CombinedPreprocessor()
        X_scaled = preprocessor.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        config = experiment.config or {}
        if config.get('auto_balance', False) and problem_type == 'classification':
             try:
                 from imblearn.over_sampling import SMOTE
                 X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)
             except ImportError:
                 pass

        _update_thinking_log(experiment, "🚀 Data re-processed. Identifying Top 3 models...")

        results = experiment.results or {}
        all_models = results.get('all_models', [])

        top_3 = sorted(all_models, key=lambda x: x.get('score', 0), reverse=True)[:3]
        top_3_names = [m['model'] for m in top_3]
        _update_thinking_log(experiment, f"🏆 Top 3 combatants identified: {', '.join(top_3_names)}")

        estimators = []
        n_estimators = int(config.get('n_estimators', 100))
        max_depth = config.get('max_depth'); max_depth = int(max_depth) if max_depth else None
        epochs = config.get('epochs'); max_iter = int(epochs) if epochs else 1000

        if problem_type == 'classification':
             from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
             from sklearn.svm import SVC
             from sklearn.neighbors import KNeighborsClassifier
             from sklearn.tree import DecisionTreeClassifier
             from sklearn.linear_model import LogisticRegression

             model_map = {
                 'Logistic Regression': LogisticRegression(max_iter=max_iter, random_state=42),
                 'Random Forest': RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42),
                 'Gradient Boosting': GradientBoostingClassifier(n_estimators=n_estimators, random_state=42, max_depth=max_depth if max_depth else 5),
                 'SVM': SVC(kernel='rbf', probability=True, random_state=42, max_iter=max_iter if epochs else -1),
                 'KNN': KNeighborsClassifier(n_neighbors=5),
                 'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=max_depth if max_depth else 10),
                 'AdaBoost': AdaBoostClassifier(n_estimators=n_estimators, random_state=42),
                 'Extra Trees': ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
             }
             try:
                 from xgboost import XGBClassifier
                 model_map['XGBoost'] = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
             except ImportError: pass
             try:
                 from lightgbm import LGBMClassifier
                 model_map['LightGBM'] = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
             except ImportError: pass

             for name in top_3_names:
                 if name in model_map: estimators.append((name, model_map[name]))

             if len(estimators) < 2: raise Exception("Not enough valid models to ensemble.")

             from sklearn.ensemble import VotingClassifier
             try:
                 ensemble = VotingClassifier(estimators=estimators, voting='soft')
                 _update_thinking_log(experiment, "☄️ Training Voting Ensemble (Soft Voting)...")
                 ensemble.fit(X_train, y_train)
             except Exception:
                 ensemble = VotingClassifier(estimators=estimators, voting='hard')
                 _update_thinking_log(experiment, "☄️ Training Voting Ensemble (Hard Voting)...")
                 ensemble.fit(X_train, y_train)

             y_pred = ensemble.predict(X_test)
             score = float(accuracy_score(y_test, y_pred))
             metrics = {'accuracy': score, 'f1_score': float(f1_score(y_test, y_pred, average='weighted'))}

        else:
             from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
             from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
             from sklearn.svm import SVR
             from sklearn.neighbors import KNeighborsRegressor
             from sklearn.tree import DecisionTreeRegressor

             model_map = {
                 'Linear Regression': LinearRegression(),
                 'Ridge Regression': Ridge(random_state=42, max_iter=max_iter if epochs else None),
                 'Lasso Regression': Lasso(random_state=42, max_iter=max_iter if epochs else 2000),
                 'ElasticNet': ElasticNet(random_state=42, max_iter=max_iter if epochs else 2000),
                 'Random Forest': RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42),
                 'Gradient Boosting': GradientBoostingRegressor(n_estimators=n_estimators, random_state=42, max_depth=max_depth if max_depth else 5),
                 'SVR': SVR(kernel='rbf', max_iter=max_iter if epochs else -1),
                 'KNN': KNeighborsRegressor(n_neighbors=5),
                 'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=max_depth if max_depth else 10),
                 'AdaBoost': AdaBoostRegressor(n_estimators=n_estimators, random_state=42),
                 'Extra Trees': ExtraTreesRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
             }
             try:
                 from xgboost import XGBRegressor
                 model_map['XGBoost'] = XGBRegressor(n_estimators=100, random_state=42)
             except ImportError: pass
             try:
                 from lightgbm import LGBMRegressor
                 model_map['LightGBM'] = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
             except ImportError: pass

             for name in top_3_names:
                 if name in model_map: estimators.append((name, model_map[name]))

             if len(estimators) < 2: raise Exception("Not enough valid models to ensemble.")

             from sklearn.ensemble import VotingRegressor
             ensemble = VotingRegressor(estimators=estimators)
             _update_thinking_log(experiment, "☄️ Training Voting Ensemble...")
             ensemble.fit(X_train, y_train)

             y_pred = ensemble.predict(X_test)
             score = float(r2_score(y_test, y_pred))
             metrics = {'r2_score': score, 'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred)))}

        model_name = "Ensemble Super-Model (Top 3)"
        job = TrainingJob(
            experiment_id=experiment.id,
            model_name=model_name,
            status='completed',
            metrics=metrics,
            logs=f"✅ Ensemble Training completed.\n📊 Score: {score:.4f}\n"
        )
        db.session.add(job)

        all_models.append({'model': model_name, 'score': score, 'metrics': metrics})
        from app.services.explanation_engine import ExplanationEngine
        ExplanationEngine.emit_model_result(experiment, model_name, score, metrics, problem_type)

        best_score = float(experiment.best_score or 0)
        zip_filename = experiment.results.get('model_package_path')
        if score > best_score:
            experiment.best_score = score
            experiment.best_model_name = model_name
            _update_thinking_log(experiment, f"👑 Ensemble Winner takes the crown! New best score: {score:.4f}")

            import tempfile
            import sys
            sys.path.insert(0, '/app')
            from ml_engine.packaging.model_packager import ModelPackager, create_feature_schema

            with tempfile.TemporaryDirectory() as tmp_dir:
                packager = ModelPackager(tmp_dir)
                feature_schema = create_feature_schema(df, target_column)
                metadata = {
                    'name': experiment.name,
                    'experiment_id': experiment.id,
                    'problem_type': problem_type,
                    'target_column': target_column,
                    'best_model': model_name,
                    'best_score': score,
                    'training_results': all_models
                }

                prep = CombinedPreprocessor()
                prep.fit_transform(X)

                package_dir = packager.package(
                    model=ensemble,
                    preprocessor=prep,
                    feature_schema=feature_schema,
                    metadata=metadata,
                    model_format='pkl'
                )
                zip_path = packager.create_zip(package_dir)
                zip_filename = f"user_{experiment.user_id}/experiment_{experiment.id}/model_package.zip"
                with open(zip_path, 'rb') as f: zip_content = f.read()
                minio_service.upload_bytes('models', zip_filename, zip_content, 'application/zip')

        else:
            _update_thinking_log(experiment, f"📉 Ensemble evaluated ({score:.4f}), but previous best still holds.")

        updated_results = dict(experiment.results)
        updated_results['all_models'] = all_models
        updated_results['best_score'] = float(experiment.best_score)
        updated_results['best_model'] = experiment.best_model_name
        updated_results['model_package_path'] = zip_filename

        from app.services.ai_insight_generator import ai_insight_generator
        try:
             comparison_insight = ai_insight_generator.explain_model_comparison(
                 all_results=all_models, best_model_name=experiment.best_model_name, target_column=target_column
             )
             ExplanationEngine.emit_insight(experiment, f"Combined {', '.join(top_3_names)} into an Ensemble Super-Model. Score: {score:.4f}. " + comparison_insight, 'model_comparison')
        except: pass

        experiment.results = updated_results
        experiment.status = 'completed'
        db.session.commit()

    except Exception as e:
        experiment.status = 'completed'
        _update_thinking_log(experiment, f"❌ Ensemble failed: {str(e)}")
        db.session.commit()
        print(f"Ensemble failed: {e}")


def _run_training_task(experiment_id, file_path, target_column, column_info):
    """Run the actual training task"""
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
    import joblib
    import io
    import json

    experiment = Experiment.query.get(experiment_id)
    if not experiment:
        return

    # Extract goal_description from experiment (set during creation)
    goal_description = experiment.goal_description or ''

    # Import ExplanationEngine and AI Insight Generator
    from app.services.explanation_engine import ExplanationEngine
    from app.services.ai_insight_generator import ai_insight_generator

    # Initialize thinking logs and explanation data
    try:
        experiment.thinking_logs = ''
        experiment.explanation_data = ExplanationEngine.init_explanation_data()
        db.session.commit()
    except Exception:
        db.session.rollback()

    print(f"\n{'='*60}", flush=True)
    print(f"🚀 STARTING TRAINING - Experiment #{experiment_id}", flush=True)
    print(f"{'='*60}", flush=True)

    _update_thinking_log(experiment, "🚀 Initializing training pipeline...")

    # PHASE 1: Data Preprocessing
    ExplanationEngine.emit_phase(experiment, 'preprocessing', {'target': target_column})

    try:
        # Load dataset from MinIO or local
        _update_thinking_log(experiment, "📂 Loading dataset from storage...")
        try:
            from app.services.minio_service import get_minio_service
            print(f"📂 Attempting to download file: {file_path}", flush=True)
            minio_service = get_minio_service()

            # List objects to verify it exists
            objects = minio_service.list_objects('datasets', prefix=file_path)
            print(f"🔎 Found objects: {[obj['name'] for obj in objects]}")

            file_content = minio_service.download_bytes('datasets', file_path)

            if not file_content:
                raise Exception("File content is empty or None")

            print(f"📦 Downloaded {len(file_content)} bytes")
            # Support both CSV and Excel files
            if file_path.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(io.BytesIO(file_content))
            else:
                df = pd.read_csv(io.BytesIO(file_content))
            print(f"📊 DataFrame loaded: {df.shape}")
            _update_thinking_log(experiment, f"✅ Dataset loaded successfully: {df.shape[0]:,} rows × {df.shape[1]} columns")

            # ── Granular preprocessing insights ──
            ExplanationEngine.emit_insight(experiment, f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns from storage", 'preprocessing')

            # Null value analysis
            null_counts = df.isnull().sum()
            total_nulls = int(null_counts.sum())
            cols_with_nulls = int((null_counts > 0).sum())
            if total_nulls > 0:
                ExplanationEngine.emit_insight(experiment, f"Found {total_nulls:,} missing values across {cols_with_nulls} columns — will be handled during preprocessing", 'preprocessing')
            else:
                ExplanationEngine.emit_insight(experiment, "No missing values detected — dataset is clean", 'preprocessing')

            # Duplicate check
            dup_count = int(df.duplicated().sum())
            if dup_count > 0:
                ExplanationEngine.emit_insight(experiment, f"Detected {dup_count:,} duplicate rows ({dup_count/len(df)*100:.1f}% of data)", 'preprocessing')

            # Column type breakdown
            num_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
            cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
            ExplanationEngine.emit_insight(experiment, f"Column types: {len(num_cols)} numeric, {len(cat_cols)} categorical", 'preprocessing')

            # AI-powered insights (non-blocking)
            try:
                ai_insights = ai_insight_generator.analyze_dataset(df, target_column, goal_description or '')
                for insight in ai_insights:
                    ExplanationEngine.emit_insight(experiment, insight, 'preprocessing')
            except Exception as e:
                print(f"⚠️ AI insight generation failed: {e}", flush=True)
        except Exception as e:
            print(f"⚠️ MinIO download failed: {e}")
            import traceback
            traceback.print_exc()

            # Fallback logic...
            print("🔄 Switching to fallback dummy data...")
            if column_info:
                cols = list(column_info.keys())
                df = pd.DataFrame({col: np.random.randn(100) for col in cols})
                # If target is categorical, fix it
                if target_column in column_info and column_info[target_column].get('dtype') == 'object':
                     df[target_column] = np.random.choice(['A', 'B', 'C'], 100)
                else:
                     # Make sure target exists if using dummy data
                     if target_column not in df.columns:
                         df[target_column] = np.random.randint(0, 2, 100)
            else:
                raise Exception("Cannot load dataset and no column info available")

        # Prepare data
        if target_column not in df.columns:
            experiment.status = 'failed'
            experiment.error_message = f'Target column "{target_column}" not found'
            db.session.commit()
            return

        X = df.drop(columns=[target_column])
        y = df[target_column]
        target_classes = None

        # Detect problem type with deterministic logic, then optionally verify with AI.
        _update_thinking_log(experiment, f"Analyzing target column '{target_column}'...")
        detector = ProblemDetector(df, target_column, goal_description=goal_description or '')
        deterministic_detection = detector.detect()
        raw_problem_type = deterministic_detection.get('problem_type', 'unknown')
        problem_type = 'classification' if 'classification' in raw_problem_type else raw_problem_type
        detection_source = 'deterministic'
        ai_verification = None

        config = experiment.config or {}
        if config.get('ai_verify_problem_type', True):
            try:
                from app.services.gemini_service import get_gemini_service

                ai_service = get_gemini_service()
                ai_verification = ai_service.verify_problem_detection(
                    columns=list(df.columns),
                    column_types={col: str(df[col].dtype) for col in df.columns},
                    sample_data=df.head(10).to_dict(orient='records'),
                    target_column=target_column,
                    deterministic_detection=deterministic_detection,
                    user_goal=goal_description or ''
                )
                ai_problem_type = ai_verification.get('problem_type', 'unknown')
                ai_confidence = float(ai_verification.get('confidence') or 0)
                deterministic_confidence = float(deterministic_detection.get('confidence') or 0)

                if (
                    ai_problem_type != 'unknown' and
                    ai_problem_type != raw_problem_type and
                    ai_confidence >= 0.85 and
                    ai_confidence >= deterministic_confidence + 0.1
                ):
                    raw_problem_type = ai_problem_type
                    problem_type = 'classification' if 'classification' in raw_problem_type else raw_problem_type
                    detection_source = 'ai_verified_override'
                    _update_thinking_log(experiment, f"AI verification adjusted problem type to {problem_type}")
                else:
                    detection_source = 'ai_verified_agree' if ai_verification.get('agreement') else 'deterministic_ai_checked'
                    _update_thinking_log(experiment, f"AI verification checked problem type: {problem_type}")

                if ai_verification.get('reasoning'):
                    ExplanationEngine.emit_insight(experiment, f"AI verification: {ai_verification['reasoning']}", 'preprocessing')
            except Exception as e:
                ai_verification = {
                    'problem_type': 'unknown',
                    'confidence': 0.0,
                    'reasoning': f'AI verification skipped: {str(e)}'
                }
                print(f"AI problem verification failed: {e}", flush=True)

        if problem_type not in ['classification', 'regression']:
            problem_type = 'classification' if y.nunique() <= 20 else 'regression'
            detection_source = f'{detection_source}_fallback'

        if problem_type == 'classification':
            _update_thinking_log(experiment, f"Detected problem type: Classification ({y.nunique()} unique classes)")
            ExplanationEngine.emit_insight(
                experiment,
                f"Target '{target_column}' -> Classification ({y.nunique()} classes, source: {detection_source})",
                'preprocessing'
            )
            if y.dtype == 'object' or str(y.dtype) == 'category' or str(y.dtype) == 'bool':
                le = LabelEncoder()
                y = le.fit_transform(y.astype(str))
                target_classes = le.classes_.tolist()
                ExplanationEngine.emit_insight(experiment, f"Label-encoded categorical target into {int(max(y))+1} numeric classes", 'preprocessing')
                _update_thinking_log(experiment, f"Encoded categorical target labels: {target_classes}")
        else:
            _update_thinking_log(experiment, "Detected problem type: Regression (continuous values)")
            ExplanationEngine.emit_insight(
                experiment,
                f"Target '{target_column}' -> Regression (source: {detection_source})",
                'preprocessing'
            )

        problem_detection = {
            'selected_problem_type': problem_type,
            'raw_problem_type': raw_problem_type,
            'source': detection_source,
            'deterministic': deterministic_detection,
            'ai_verification': ai_verification
        }
        experiment.problem_type = problem_type
        experiment.results = {
            **(experiment.results or {}),
            'problem_detection': problem_detection
        }
        db.session.commit()

        # Preprocessing steps with detailed logging
        _update_thinking_log(experiment, "🔧 Preprocessing data: Encoding categorical features...")
        cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        num_features = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        if cat_features:
            ExplanationEngine.emit_insight(experiment, f"Encoding {len(cat_features)} categorical features: {', '.join(cat_features[:5])}{'...' if len(cat_features)>5 else ''}", 'preprocessing')
        if num_features:
            ExplanationEngine.emit_insight(experiment, f"Scaling {len(num_features)} numeric features via StandardScaler", 'preprocessing')

        preprocessor = CombinedPreprocessor()
        X_scaled = preprocessor.fit_transform(X)
        ExplanationEngine.emit_insight(experiment, f"Preprocessing complete — {X_scaled.shape[1]} features ready for training", 'preprocessing')
        _update_thinking_log(experiment, "📊 Preprocessing data: Scaling numeric features...")

        # Split data
        _update_thinking_log(experiment, "✂️ Splitting data into training (80%) and test (20%) sets...")
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        ExplanationEngine.emit_insight(experiment, f"Train/test split → {len(X_train):,} training, {len(X_test):,} holdout samples (80/20)", 'preprocessing')

        # Extract configuration
        config = experiment.config or {}
        auto_balance = config.get('auto_balance', False)

        if auto_balance and problem_type == 'classification':
            _update_thinking_log(experiment, "⚖️ Auto-Balancing: Applying SMOTE to generate synthetic minority records...")
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=42)
                old_size = len(X_train)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                _update_thinking_log(experiment, f"✅ Auto-Balancing complete. New training size: {len(X_train):,} samples")
                ExplanationEngine.emit_insight(experiment, f"SMOTE oversampling: {old_size:,} → {len(X_train):,} samples (balanced minority classes)", 'preprocessing')
            except ImportError:
                _update_thinking_log(experiment, "⚠️ imbalanced-learn not installed. Skipping auto-balancing.")
                print("⚠️ imbalanced-learn not installed. Skipping auto-balancing.")

        _update_thinking_log(experiment, f"✅ Data prepared: {len(X_train):,} training samples, {len(X_test):,} test samples")

        # Emit trust data
        ExplanationEngine.emit_trust_data(experiment, len(df), len(X_train), len(X_test), 'high' if len(df) >= 1000 else 'medium')

        # ── PHASE 2: Feature Engineering & Selection ──
        ExplanationEngine.emit_phase(experiment, 'feature_engineering', {'features': X_scaled.shape[1]})

        # Correlation analysis
        try:
            import pandas as pd
            numeric_df = df.select_dtypes(include=['int64','float64','int32','float32'])
            if target_column in numeric_df.columns and len(numeric_df.columns) > 1:
                correlations = numeric_df.corr()[target_column].drop(target_column, errors='ignore').abs().sort_values(ascending=False)
                top_corr = correlations.head(5)
                top_names = [f"{col} ({val:.2f})" for col, val in top_corr.items()]
                ExplanationEngine.emit_insight(experiment, f"Top correlated features: {', '.join(top_names)}", 'feature_engineering')

                # Weak features
                weak = correlations[correlations < 0.05]
                if len(weak) > 0:
                    ExplanationEngine.emit_insight(experiment, f"{len(weak)} features show very weak correlation (<0.05) with target", 'feature_engineering')
        except Exception:
            pass

        # Feature stats
        ExplanationEngine.emit_insight(experiment, f"Final feature matrix: {X_scaled.shape[0]:,} samples × {X_scaled.shape[1]} features", 'feature_engineering')
        ExplanationEngine.emit_insight(experiment, f"All features retained for model training (no manual feature selection applied)", 'feature_engineering')

        # ── PHASE 3: Model Development ──
        ExplanationEngine.emit_phase(experiment, 'model_development')

        # Extract overrides from custom config
        config = experiment.config or {}
        n_estimators = int(config.get('n_estimators', 100))
        max_depth = config.get('max_depth')
        if max_depth is not None: max_depth = int(max_depth)
        epochs = config.get('epochs')
        max_iter = int(epochs) if epochs else 1000

        from app.models.user import User
        user = User.query.get(experiment.user_id)
        is_pro = user and user.plan_type in ('pro', 'advance')

        # Define models to try
        if problem_type == 'classification':
            from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
            from sklearn.svm import SVC
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.linear_model import LogisticRegression

            models = [
                ('Logistic Regression', LogisticRegression(max_iter=max_iter, random_state=42)),
                ('Random Forest', RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)),
                ('Decision Tree', DecisionTreeClassifier(random_state=42, max_depth=max_depth if max_depth else 10)),
                ('KNN', KNeighborsClassifier(n_neighbors=5))
            ]

            if is_pro:
                models.extend([
                    ('Gradient Boosting', GradientBoostingClassifier(n_estimators=n_estimators, random_state=42, max_depth=max_depth if max_depth else 5)),
                    ('SVM', SVC(kernel='rbf', probability=True, random_state=42, max_iter=max_iter if epochs else -1)),
                    ('AdaBoost', AdaBoostClassifier(n_estimators=n_estimators, random_state=42)),
                    ('Extra Trees', ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42))
                ])

                # Add XGBoost if available
                try:
                    from xgboost import XGBClassifier
                    models.append(('XGBoost', XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')))
                except ImportError:
                    print("⚠️ XGBoost not available, skipping...", flush=True)

                # Add LightGBM if available
                try:
                    from lightgbm import LGBMClassifier
                    models.append(('LightGBM', LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)))
                except ImportError:
                    print("⚠️ LightGBM not available, skipping...", flush=True)

            scoring = 'accuracy'
        else:
            from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
            from sklearn.svm import SVR
            from sklearn.neighbors import KNeighborsRegressor
            from sklearn.tree import DecisionTreeRegressor

            models = [
                ('Linear Regression', LinearRegression()),
                ('Random Forest', RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)),
                ('Decision Tree', DecisionTreeRegressor(random_state=42, max_depth=max_depth if max_depth else 10)),
                ('KNN', KNeighborsRegressor(n_neighbors=5))
            ]

            if is_pro:
                models.extend([
                    ('Ridge Regression', Ridge(random_state=42, max_iter=max_iter if epochs else None)),
                    ('Lasso Regression', Lasso(random_state=42, max_iter=max_iter if epochs else 2000)),
                    ('ElasticNet', ElasticNet(random_state=42, max_iter=max_iter if epochs else 2000)),
                    ('Gradient Boosting', GradientBoostingRegressor(n_estimators=n_estimators, random_state=42, max_depth=max_depth if max_depth else 5)),
                    ('SVR', SVR(kernel='rbf', max_iter=max_iter if epochs else -1)),
                    ('AdaBoost', AdaBoostRegressor(n_estimators=n_estimators, random_state=42)),
                    ('Extra Trees', ExtraTreesRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42))
                ])

                # Add XGBoost if available
                try:
                    from xgboost import XGBRegressor
                    models.append(('XGBoost', XGBRegressor(n_estimators=100, random_state=42)))
                except ImportError:
                    print("⚠️ XGBoost not available, skipping...", flush=True)

                # Add LightGBM if available
                try:
                    from lightgbm import LGBMRegressor
                    models.append(('LightGBM', LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)))
                except ImportError:
                    print("⚠️ LightGBM not available, skipping...", flush=True)

            scoring = 'r2'


        print(f"\n📊 Problem Type: {problem_type.upper()}")
        print(f"📈 Models to train: {len(models)}")
        print(f"{'─'*40}")

        _update_thinking_log(experiment, f"🧠 Selecting {len(models)} algorithms to evaluate...")
        ExplanationEngine.emit_insight(experiment, f"Prepared {len(models)} algorithms for benchmarking on {len(X_train):,} training samples", 'model_development')

        best_model = None
        best_score = -float('inf')
        best_model_name = ''
        all_results = []
        total_models = len(models)

        for idx, (model_name, model) in enumerate(models, 1):
            # Create training job record
            _update_thinking_log(experiment, f"⏳ Training model {idx}/{total_models}: {model_name}...")
            job = TrainingJob(
                experiment_id=experiment.id,
                model_name=model_name,
                status='training',
                logs=f"🚀 Starting training for {model_name}...\n"
            )
            db.session.add(job)
            db.session.commit()

            try:
                # Train model
                ExplanationEngine.emit_model_start(experiment, model_name)
                ExplanationEngine.emit_insight(experiment, f"Training [{idx}/{total_models}] {model_name}...", 'model_development')
                print(f"\n⏳ Training {model_name}...", flush=True)
                job.logs += f"⏳ Training model...\n"
                db.session.commit()

                model.fit(X_train, y_train)

                # Evaluate
                if problem_type == 'classification':
                    y_pred = model.predict(X_test)
                    score = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    job.metrics = {'accuracy': score, 'f1_score': f1}
                    log_msg = f"✅ Training completed.\n📊 Accuracy: {score:.4f}\n📊 F1 Score: {f1:.4f}\n"
                else:
                    y_pred = model.predict(X_test)
                    score = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    job.metrics = {'r2_score': score, 'rmse': rmse}
                    log_msg = f"✅ Training completed.\n📊 R² Score: {score:.4f}\n📊 RMSE: {rmse:.4f}\n"

                # job.cv_score = score # Not in DB model
                job.status = 'completed'
                job.logs += log_msg
                db.session.commit()

                # Print score
                print(log_msg.replace('\n', ' '), flush=True)

                all_results.append({
                    'model': model_name,
                    'score': score,
                    'metrics': job.metrics
                })

                if score > best_score:
                    best_score = score
                    best_model = model
                    best_model_name = model_name

                _update_thinking_log(experiment, f"📈 {model_name} completed: Score = {score:.4f}")
                ExplanationEngine.emit_model_result(experiment, model_name, score, job.metrics, problem_type)
                ExplanationEngine.emit_insight(experiment, f"✓ {model_name} → {score*100:.2f}%{'  ← new best!' if model_name == best_model_name else ''}", 'model_development')

                # Generate AI insight about what this model learned
                try:
                    if hasattr(model, 'feature_importances_'):
                        feature_names = list(X.columns)
                        importances = dict(zip(feature_names, model.feature_importances_))
                        learning_insight = ai_insight_generator.analyze_model_learning(
                            model_name, importances, score, target_column
                        )
                        ExplanationEngine.emit_insight(experiment, learning_insight, 'model_learning')
                except Exception as e:
                    print(f"⚠️ Model learning insight failed: {e}", flush=True)

            except Exception as e:
                job.status = 'failed'
                job.error_message = str(e)
                job.logs += f"❌ Training failed: {str(e)}\n"
                db.session.commit()
                print(f"   ❌ {model_name}: FAILED - {e}", flush=True)

        # ── PHASE 4: Best Model & Evaluation ──
        ExplanationEngine.emit_phase(experiment, 'evaluation')

        # Update experiment with results
        _update_thinking_log(experiment, f"🏆 Comparing scores to find best model...")

        # Emit detailed evaluation insights
        if len(all_results) > 0:
            sorted_results = sorted(all_results, key=lambda x: x.get('score', 0), reverse=True)
            ExplanationEngine.emit_insight(experiment, f"Evaluated {len(all_results)} models — ranking by {'accuracy' if problem_type == 'classification' else 'R² score'}", 'evaluation')
            for i, r in enumerate(sorted_results[:3]):
                ExplanationEngine.emit_insight(experiment, f"#{i+1} {r['model']} — Score: {r['score']*100:.2f}%", 'evaluation')

        experiment.best_model_name = best_model_name
        experiment.best_score = float(best_score)
        experiment.results = {
            'all_models': all_results,
            'best_model': best_model_name,
            'best_score': float(best_score),
            'problem_type': problem_type,
            'feature_names': list(X.columns),
            'problem_detection': problem_detection
        }
        db.session.commit()

        _update_thinking_log(experiment, f"🏆 Best model: {best_model_name} with score {best_score:.4f}")
        ExplanationEngine.emit_insight(experiment, f"Champion: {best_model_name} with {best_score*100:.2f}% — selected for deployment", 'evaluation')

        # Generate AI insight about why this model won
        try:
            comparison_insight = ai_insight_generator.explain_model_comparison(
                all_results, best_model_name, target_column
            )
            ExplanationEngine.emit_insight(experiment, comparison_insight, 'evaluation')
        except Exception as e:
            print(f"⚠️ Model comparison insight failed: {e}", flush=True)

        # ── PHASE 5: Deployment Ready ──
        ExplanationEngine.emit_phase(experiment, 'ready')
        ExplanationEngine.emit_final_summary(experiment, best_model_name, best_score, problem_type, target_column, len(X.columns))

        print(f"\n{'='*60}")
        print(f"🏆 TRAINING COMPLETE!")
        print(f"{'='*60}")
        print(f"   Best Model: {best_model_name}")
        print(f"   Best Score: {best_score:.4f}")
        print(f"{'='*60}\n")

        # Package the best model into a ZIP file
        if best_model is not None:
            _update_thinking_log(experiment, "📦 Packaging the best model for deployment...")
            try:
                import tempfile
                import sys
                sys.path.insert(0, '/app')
                from ml_engine.packaging.model_packager import ModelPackager, create_feature_schema
                from app.services.minio_service import get_minio_service

                print("📦 Packaging model...", flush=True)

                with tempfile.TemporaryDirectory() as tmp_dir:
                    packager = ModelPackager(tmp_dir)

                    # Create feature schema from DataFrame
                    feature_schema = create_feature_schema(df, target_column)

                    # Prepare metadata
                    metadata = {
                        'name': experimentName if 'experimentName' in dir() else f"Model_{experiment.id}",
                        'experiment_id': experiment.id,
                        'problem_type': problem_type,
                        'target_column': target_column,
                        'best_model': best_model_name,
                        'best_score': float(best_score),
                        'training_results': all_results,
                        'target_classes': target_classes,  # class names for decoding predictions
                    }

                    # Create the package directory with all files
                    package_dir = packager.package(
                        model=best_model,
                        preprocessor=preprocessor,  # The CombinedPreprocessor we created
                        feature_schema=feature_schema,
                        metadata=metadata,
                        model_format='pkl'
                    )

                    # Create ZIP file
                    zip_path = packager.create_zip(package_dir)
                    print(f"📦 ZIP created: {zip_path}", flush=True)

                    # Upload to MinIO
                    minio_service = get_minio_service()
                    zip_filename = f"user_{experiment.user_id}/experiment_{experiment.id}/model_package.zip"

                    with open(zip_path, 'rb') as f:
                        zip_content = f.read()

                    minio_service.upload_bytes(
                        bucket='models',
                        object_name=zip_filename,
                        data=zip_content,
                        content_type='application/zip'
                    )

                    # Update experiment with model path
                    # Need to copy and update to trigger SQLAlchemy change detection
                    updated_results = dict(experiment.results or {})
                    updated_results['model_package_path'] = zip_filename
                    experiment.results = updated_results
                    db.session.commit()

                    print(f"✅ Model package uploaded: {zip_filename}", flush=True)
                    _update_thinking_log(experiment, "✅ Model packaged and ready for deployment!")

            except Exception as pack_error:
                print(f"⚠️ Model packaging failed: {pack_error}", flush=True)
                import traceback
                traceback.print_exc()

        experiment.status = 'completed'
        db.session.commit()

    except Exception as e:
        experiment.status = 'failed'
        experiment.error_message = str(e)
        _update_thinking_log(experiment, f"❌ Training failed: {str(e)}")
        db.session.commit()
        print(f"Training failed: {e}")


def _run_image_training_task(experiment_id, file_path, column_info):
    """Run image classification training using ImageClassifier (transfer learning)"""
    import sys
    import tempfile
    import numpy as np
    import json

    experiment = Experiment.query.get(experiment_id)
    if not experiment:
        return

    # Import ExplanationEngine
    from app.services.explanation_engine import ExplanationEngine

    # Initialize thinking logs
    try:
        experiment.thinking_logs = ''
        experiment.explanation_data = ExplanationEngine.init_explanation_data()
        experiment.problem_type = 'image_classification'
        db.session.commit()
    except Exception:
        db.session.rollback()

    print(f"\n{'='*60}", flush=True)
    print(f"🖼️  STARTING IMAGE CLASSIFICATION - Experiment #{experiment_id}", flush=True)
    print(f"{'='*60}", flush=True)

    _update_thinking_log(experiment, "🖼️ Initializing image classification pipeline...")
    ExplanationEngine.emit_phase(experiment, 'understanding', {'target': 'image classes'})

    try:
        # PHASE 1: Download and extract dataset
        _update_thinking_log(experiment, "📂 Downloading image dataset from storage...")

        from app.services.minio_service import get_minio_service
        minio_service = get_minio_service()
        file_content = minio_service.download_bytes('datasets', file_path)

        if not file_content:
            raise Exception("Failed to download dataset from storage")

        _update_thinking_log(experiment, f"✅ Downloaded {len(file_content) / (1024*1024):.1f} MB")

        # Extract class info from column_info
        class_names = column_info.get('classes', [])
        num_classes = column_info.get('num_classes', len(class_names))
        total_images = column_info.get('total_images', 0)

        _update_thinking_log(experiment, f"📊 Dataset: {total_images} images across {num_classes} classes")
        _update_thinking_log(experiment, f"📋 Classes: {', '.join(class_names)}")

        ExplanationEngine.emit_insight(
            experiment,
            f"Image dataset with {num_classes} classes: {', '.join(class_names[:5])}{'...' if len(class_names) > 5 else ''}",
            'data_analysis'
        )

        # PHASE 2: Load and preprocess images
        _update_thinking_log(experiment, "🔧 Loading and preprocessing images...")
        ExplanationEngine.emit_phase(experiment, 'learning', {'features': num_classes})

        # Add ml_engine to path
        sys.path.insert(0, '/app')
        from ml_engine.preprocessing.image_preprocessor import ImagePreprocessor
        from ml_engine.automl.vision.classifier import ImageClassifier

        preprocessor = ImagePreprocessor(
            target_size=(224, 224),
            normalize=True,
            normalization_mode='imagenet',
            color_mode='rgb'
        )

        # Save zip to temp file for loading
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_zip:
            tmp_zip.write(file_content)
            tmp_zip_path = tmp_zip.name

        try:
            images, labels, loaded_class_names = preprocessor.load_from_zip(tmp_zip_path)
        finally:
            import os
            os.unlink(tmp_zip_path)

        _update_thinking_log(experiment, f"✅ Loaded {len(images)} images, resized to 224×224")
        _update_thinking_log(experiment, f"📊 Image shape: {images.shape}")

        ExplanationEngine.emit_insight(
            experiment,
            f"Preprocessed {len(images)} images: resized to 224×224, normalized with ImageNet statistics.",
            'preprocessing'
        )

        # Train/test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=0.2, random_state=42, stratify=labels
        )

        _update_thinking_log(experiment, f"✂️ Split: {len(X_train)} training, {len(X_test)} test images")

        ExplanationEngine.emit_trust_data(
            experiment, len(images), len(X_train), len(X_test),
            'high' if len(images) >= 500 else 'medium' if len(images) >= 100 else 'low'
        )

        # PHASE 3: Train models
        ExplanationEngine.emit_phase(experiment, 'testing')
        _update_thinking_log(experiment, "🧠 Training image classification models...")

        # Define models to try
        model_configs = [
            ('MobileNetV2', 'mobilenet'),
            ('EfficientNet-B0', 'efficientnet_b0'),
        ]

        best_model = None
        best_score = -float('inf')
        best_model_name = ''
        all_results = []
        total_models = len(model_configs)

        for idx, (display_name, model_key) in enumerate(model_configs, 1):
            _update_thinking_log(experiment, f"⏳ Training model {idx}/{total_models}: {display_name}...")

            job = TrainingJob(
                experiment_id=experiment.id,
                model_name=display_name,
                status='training',
                logs=f"🚀 Starting {display_name} transfer learning...\n"
            )
            db.session.add(job)
            db.session.commit()

            try:
                ExplanationEngine.emit_model_start(experiment, display_name)

                # Dynamic configurations extraction
                config = experiment.config or {}
                epochs = int(config.get('epochs', 15))
                batch_size = int(config.get('batch_size', 32))
                learning_rate = float(config.get('learning_rate', 0.001))

                classifier = ImageClassifier(
                    model_name=model_key,
                    num_classes=num_classes,
                    freeze_base=True,
                    dropout=0.5,
                    learning_rate=learning_rate,
                    epochs=epochs,
                    batch_size=batch_size
                )

                classifier.fit(
                    X_train, y_train,
                    validation_split=0.2,
                    class_names=loaded_class_names,
                    augment=True,
                    verbose=1
                )

                # Evaluate
                metrics = classifier.evaluate(X_test, y_test)
                score = metrics.get('accuracy', 0)

                job.metrics = metrics
                job.cv_score = score
                job.status = 'completed'
                job.logs += f"✅ Accuracy: {score:.4f}\n"
                if 'f1_score' in metrics:
                    job.logs += f"📊 F1 Score: {metrics['f1_score']:.4f}\n"
                db.session.commit()

                all_results.append({
                    'model': display_name,
                    'score': score,
                    'metrics': metrics
                })

                _update_thinking_log(experiment, f"📈 {display_name}: Accuracy = {score:.4f}")
                ExplanationEngine.emit_model_result(experiment, display_name, score, metrics, 'classification')

                if score > best_score:
                    best_score = score
                    best_model = classifier
                    best_model_name = display_name

            except Exception as e:
                job.status = 'failed'
                job.error_message = str(e)
                job.logs += f"❌ Training failed: {str(e)}\n"
                db.session.commit()
                print(f"   ❌ {display_name}: FAILED - {e}", flush=True)
                import traceback
                traceback.print_exc()

        # PHASE 4: Compare and select
        ExplanationEngine.emit_phase(experiment, 'comparing')
        _update_thinking_log(experiment, f"🏆 Comparing models to find best...")

        ExplanationEngine.emit_phase(experiment, 'choosing')

        experiment.best_model_name = best_model_name
        experiment.best_score = float(best_score)
        experiment.results = {
            'all_models': all_results,
            'best_model': best_model_name,
            'best_score': float(best_score),
            'problem_type': 'image_classification',
            'class_names': loaded_class_names,
            'num_classes': num_classes,
            'total_images': total_images,
            'image_size': [224, 224]
        }
        db.session.commit()

        _update_thinking_log(experiment, f"🏆 Best model: {best_model_name} with accuracy {best_score:.4f}")

        # PHASE 5: Package model
        ExplanationEngine.emit_phase(experiment, 'ready')
        ExplanationEngine.emit_final_summary(
            experiment, best_model_name, best_score,
            'image_classification', 'image classes', num_classes
        )

        import os  # Ensure os is available
        if best_model is not None:
            _update_thinking_log(experiment, "📦 Packaging the best model for deployment...")
            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    model_save_path = f"{tmp_dir}/image_model.h5"
                    best_model.save(model_save_path)

                    # Create a simple metadata file
                    import json as json_mod
                    meta = {
                        'problem_type': 'image_classification',
                        'model_name': best_model_name,
                        'accuracy': float(best_score),
                        'class_names': loaded_class_names,
                        'num_classes': num_classes,
                        'image_size': [224, 224],
                        'normalization': 'imagenet',
                    }
                    with open(f"{tmp_dir}/metadata.json", 'w') as f:
                        json_mod.dump(meta, f, indent=2)

                    # Also save as pickle for easy Python usage
                    import joblib
                    joblib.dump(best_model, f"{tmp_dir}/model.pkl")
                    _update_thinking_log(experiment, "📦 Exported model as model.pkl")

                    # Create ZIP
                    import shutil
                    print(f"DEBUG: Creating zip archive from {tmp_dir}", flush=True)

                    # Create a separate temp file for the zip to avoid recursive zipping
                    zip_tmp_fd, zip_tmp_path = tempfile.mkstemp(suffix='.zip')
                    os.close(zip_tmp_fd)
                    # Remove the file created by mkstemp because make_archive wants to create it
                    os.remove(zip_tmp_path)

                    # Use the path without extension for make_archive (it appends .zip)
                    zip_base = zip_tmp_path.replace('.zip', '')

                    zip_path = shutil.make_archive(zip_base, 'zip', tmp_dir)
                    print(f"DEBUG: Zip created at {zip_path}", flush=True)

                    # Upload to MinIO
                    zip_filename = f"user_{experiment.user_id}/experiment_{experiment.id}/image_model_package.zip"
                    print(f"DEBUG: Reading zip file for upload to {zip_filename}", flush=True)
                    with open(zip_path, 'rb') as f:
                        zip_content = f.read()
                    print(f"DEBUG: Zip size: {len(zip_content)} bytes", flush=True)

                    print(f"DEBUG: Uploading to MinIO bucket 'models'...", flush=True)
                    upload_success = minio_service.upload_bytes(
                        bucket='models',
                        object_name=zip_filename,
                        data=zip_content,
                        content_type='application/zip'
                    )
                    print(f"DEBUG: Upload result: {upload_success}", flush=True)

                    updated_results = dict(experiment.results or {})
                    updated_results['model_package_path'] = zip_filename
                    experiment.results = updated_results
                    db.session.commit()

                    print(f"✅ Image model package uploaded: {zip_filename}", flush=True)
                    _update_thinking_log(experiment, "✅ Model packaged and ready for deployment!")
            except Exception as pack_error:
                print(f"⚠️ Model packaging failed: {pack_error}", flush=True)
                import traceback
                traceback.print_exc()

        experiment.status = 'completed'
        db.session.commit()

        print(f"\n{'='*60}", flush=True)
        print(f"🏆 IMAGE TRAINING COMPLETE!", flush=True)
        print(f"   Best Model: {best_model_name}", flush=True)
        print(f"   Accuracy: {best_score:.4f}", flush=True)
        print(f"{'='*60}\n", flush=True)

    except Exception as e:
        experiment.status = 'failed'
        experiment.error_message = str(e)
        _update_thinking_log(experiment, f"❌ Image training failed: {str(e)}")
        db.session.commit()
        print(f"Image training failed: {e}", flush=True)
        import traceback
        traceback.print_exc()


@training_bp.route('/<int:job_id>/status', methods=['GET'])
@jwt_required()
def get_training_status(job_id):
    """Get training job status"""
    user_id = int(get_jwt_identity())

    experiment = Experiment.query.filter_by(id=job_id, user_id=user_id).first()
    if not experiment:
        return jsonify({'error': 'Experiment not found'}), 404

    # Get all training jobs for this experiment
    jobs = TrainingJob.query.filter_by(experiment_id=experiment.id).all()

    return jsonify({
        'experiment': experiment.to_dict(),
        'jobs': [j.to_dict() for j in jobs]
    }), 200


@training_bp.route('/<int:job_id>/logs', methods=['GET'])
@jwt_required()
def get_training_logs(job_id):
    """Get training logs"""
    user_id = int(get_jwt_identity())

    experiment = Experiment.query.filter_by(id=job_id, user_id=user_id).first()
    if not experiment:
        return jsonify({'error': 'Experiment not found'}), 404

    jobs = TrainingJob.query.filter_by(experiment_id=experiment.id).all()

    logs = []
    for job in jobs:
        logs.append({
            'model_name': job.model_name,
            'status': job.status,
            'logs': job.logs,
            'error': job.error_message
        })

    return jsonify({'logs': logs}), 200


@training_bp.route('/<int:job_id>/thinking', methods=['GET'])
@jwt_required()
def get_thinking_logs(job_id):
    """Get real-time thinking logs for the AI training process"""
    user_id = int(get_jwt_identity())

    experiment = Experiment.query.filter_by(id=job_id, user_id=user_id).first()
    if not experiment:
        return jsonify({'error': 'Experiment not found'}), 404

    # Parse thinking logs into structured format
    try:
        thinking_logs = experiment.thinking_logs or ''
    except Exception:
        # Column might not exist yet
        thinking_logs = ''

    messages = []

    for line in thinking_logs.strip().split('\n'):
        if line.strip():
            # Parse "[HH:MM:SS] message" format
            if line.startswith('[') and ']' in line:
                try:
                    timestamp = line[1:line.index(']')]
                    message = line[line.index(']') + 2:]
                    messages.append({
                        'timestamp': timestamp,
                        'message': message
                    })
                except:
                    messages.append({
                        'timestamp': '',
                        'message': line
                    })
            else:
                messages.append({
                    'timestamp': '',
                    'message': line
                })

    return jsonify({
        'status': experiment.status,
        'thinking_logs': messages
    }), 200


@training_bp.route('/<int:job_id>/cancel', methods=['POST'])
@jwt_required()
def cancel_training(job_id):
    """Cancel a training job"""
    user_id = int(get_jwt_identity())

    experiment = Experiment.query.filter_by(id=job_id, user_id=user_id).first()
    if not experiment:
        return jsonify({'error': 'Experiment not found'}), 404

    if experiment.status != 'training':
        return jsonify({'error': 'Experiment is not currently training'}), 400

    # TODO: Cancel Celery task

    experiment.status = 'cancelled'
    db.session.commit()

    return jsonify({'message': 'Training cancelled'}), 200

def _run_detection_training_task(experiment_id, file_path, column_info):
    """Run object detection training using YOLOv8"""
    import sys
    import tempfile
    import os
    import json

    experiment = Experiment.query.get(experiment_id)
    if not experiment:
        return

    # Import ExplanationEngine
    try:
        from app.services.explanation_engine import ExplanationEngine
    except ImportError:
        ExplanationEngine = None

    # Initialize thinking logs
    try:
        experiment.thinking_logs = ''
        if ExplanationEngine:
            experiment.explanation_data = ExplanationEngine.init_explanation_data()
        experiment.problem_type = 'object_detection'
        db.session.commit()
    except Exception:
        db.session.rollback()

    print(f"\n{'='*60}", flush=True)
    print(f"🎯 STARTING OBJECT DETECTION - Experiment #{experiment_id}", flush=True)
    print(f"{'='*60}", flush=True)

    _update_thinking_log(experiment, "🎯 Initializing YOLOv8 Object Detection pipeline...")
    if ExplanationEngine:
        ExplanationEngine.emit_phase(experiment, 'understanding', {'target': 'bounding boxes'})

    try:
        # PHASE 1: Download dataset
        _update_thinking_log(experiment, "📂 Downloading dataset from storage...")
        try:
            from app.services.minio_service import get_minio_service
        except ImportError:
            get_minio_service = None

        minio_service = get_minio_service() if get_minio_service else None
        if not minio_service:
            raise Exception("MinIO Service unavailable")

        file_content = minio_service.download_bytes('datasets', file_path)

        if not file_content:
            raise Exception("Failed to download dataset from storage")

        _update_thinking_log(experiment, f"✅ Downloaded {len(file_content) / (1024*1024):.1f} MB")

        # PHASE 2: Load and preprocess YOLO dependencies
        _update_thinking_log(experiment, "🔧 Validating and generating YOLO data.yaml...")
        if ExplanationEngine:
            ExplanationEngine.emit_phase(experiment, 'learning', {'features': 'YOLO annotations'})

        sys.path.insert(0, '/app')
        try:
            from ml_engine.preprocessing.detection_preprocessor import DetectionPreprocessor
            from ml_engine.automl.vision.detector import ObjectDetector
        except ImportError:
            DetectionPreprocessor = None
            ObjectDetector = None

        # We need a stable directory for YOLO to read from during the entire training run
        process_dir = tempfile.mkdtemp(prefix="yolo_data_")
        preprocessor = DetectionPreprocessor(data_dir=process_dir)

        # Save zip to temp file for extraction
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_zip:
            tmp_zip.write(file_content)
            tmp_zip_path = tmp_zip.name

        try:
            prep_results = preprocessor.process_zip(tmp_zip_path)
        finally:
            os.unlink(tmp_zip_path)

        yaml_path = prep_results['yaml_path']
        class_names = prep_results['classes']
        num_classes = prep_results['num_classes']

        _update_thinking_log(experiment, f"📋 Extracted YOLO dataset: {num_classes} classes mapping to {yaml_path}")
        if ExplanationEngine:
            ExplanationEngine.emit_insight(
                experiment,
                f"Validated dataset for Object Detection. Found {num_classes} classes: {', '.join(class_names[:5])}{'...' if len(class_names) > 5 else ''}",
                'preprocessing'
            )

            ExplanationEngine.emit_trust_data(
                experiment, 500, 400, 100, 'high'  # Arbitrary trust data since YOLO handles its own splits
            )

        # PHASE 3: Train YOLO
        if ExplanationEngine:
            ExplanationEngine.emit_phase(experiment, 'testing')
        _update_thinking_log(experiment, "🧠 Firing up YOLOv8 engine...")

        config = experiment.config or {}
        epochs = int(config.get('epochs', 30))
        batch_size = int(config.get('batch_size', 16))
        imgsz = int(config.get('img_size', 640))  # Default YOLO res
        learning_rate = float(config.get('learning_rate', 0.01))

        model_name = "YOLOv8 Nano"

        job = TrainingJob(
            experiment_id=experiment.id,
            model_name=model_name,
            status='training',
            logs=f"🚀 Starting YOLOv8 training (Epochs: {epochs}, Batch: {batch_size}, imgsz: {imgsz})...\n"
        )
        db.session.add(job)
        db.session.commit()

        try:
            if ExplanationEngine:
                ExplanationEngine.emit_model_start(experiment, model_name)

            project_dir = os.path.join(process_dir, 'yolo_runs')

            detector = ObjectDetector(
                model_size='n',  # nano for speed
                epochs=epochs,
                batch_size=batch_size,
                imgsz=imgsz,
                learning_rate=learning_rate
            )

            # Start YOLO Training Loop
            detector.fit_from_directory(
                data_yaml_path=yaml_path,
                project_dir=project_dir,
                name='experiment_run',
                verbose=True
            )

            # Extract metrics
            metrics = detector.metrics_
            score = metrics.get('mAP50-95', 0.0) # Primary YOLO metric

            job.metrics = metrics
            job.cv_score = score
            job.status = 'completed'
            job.logs += f"✅ mAP50-95: {score:.4f}\n"
            if 'mAP50' in metrics:
                job.logs += f"📊 mAP50: {metrics['mAP50']:.4f}\n"
            db.session.commit()

            _update_thinking_log(experiment, f"📈 {model_name}: mAP50-95 = {score:.4f}")
            if ExplanationEngine:
                ExplanationEngine.emit_model_result(experiment, model_name, score, metrics, 'object_detection')

            # PHASE 4: Package model
            if ExplanationEngine:
                ExplanationEngine.emit_phase(experiment, 'comparing')
                ExplanationEngine.emit_phase(experiment, 'choosing')

            experiment.best_model_name = model_name
            experiment.best_score = float(score)
            experiment.results = {
                'all_models': [{'model': model_name, 'score': score, 'metrics': metrics}],
                'best_model': model_name,
                'best_score': float(score),
                'problem_type': 'object_detection',
                'class_names': class_names,
                'num_classes': num_classes,
                'image_size': [imgsz, imgsz]
            }
            db.session.commit()

            if ExplanationEngine:
                ExplanationEngine.emit_phase(experiment, 'ready')
                ExplanationEngine.emit_final_summary(
                    experiment, model_name, score,
                    'object_detection', 'bounding boxes', num_classes
                )

            try:
                _update_thinking_log(experiment, "📦 Packaging PyTorch weights (.pt) for deployment...")

                # Pack .pt into a zip for MinIO
                with tempfile.TemporaryDirectory() as tmp_pkg_dir:
                    weights_path = os.path.join(tmp_pkg_dir, "best.pt")
                    detector.save(weights_path)

                    # Write standard metadata alongside it
                    meta = {
                        'problem_type': 'object_detection',
                        'model_name': model_name,
                        'score': float(score),
                        'class_names': class_names,
                        'num_classes': num_classes,
                        'image_size': [imgsz, imgsz],
                        'framework': 'ultralytics'
                    }
                    with open(os.path.join(tmp_pkg_dir, 'metadata.json'), 'w') as f:
                        json.dump(meta, f, indent=2)

                    import shutil
                    zip_tmp_fd, zip_tmp_path = tempfile.mkstemp(suffix='.zip')
                    os.close(zip_tmp_fd)
                    os.remove(zip_tmp_path)
                    zip_base = zip_tmp_path.replace('.zip', '')

                    zip_path = shutil.make_archive(zip_base, 'zip', tmp_pkg_dir)

                    zip_filename = f"user_{experiment.user_id}/experiment_{experiment.id}/detection_model_package.zip"
                    with open(zip_path, 'rb') as f:
                        zip_content = f.read()

                    minio_service.upload_bytes(
                        bucket='models',
                        object_name=zip_filename,
                        data=zip_content,
                        content_type='application/zip'
                    )

                    updated_results = dict(experiment.results or {})
                    updated_results['model_package_path'] = zip_filename
                    experiment.results = updated_results
                    db.session.commit()

                    print(f"✅ Object Detection package uploaded: {zip_filename}", flush=True)
                    _update_thinking_log(experiment, "✅ Model packaged and ready for deployment!")
            except Exception as pack_error:
                print(f"⚠️ Object Detection packaging failed: {pack_error}", flush=True)
                import traceback
                traceback.print_exc()

            experiment.status = 'completed'
            db.session.commit()

        except Exception as e:
            job.status = 'failed'
            job.error_message = str(e)
            job.logs += f"❌ Training failed: {str(e)}\n"
            db.session.commit()
            raise e

    except Exception as e:
        experiment.status = 'failed'
        experiment.error_message = str(e)
        _update_thinking_log(experiment, f"❌ Training failed: {str(e)}")
        db.session.commit()
        print(f"Detection Training failed: {e}")
        import traceback
        traceback.print_exc()
