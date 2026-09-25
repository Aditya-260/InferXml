"""
Dataset Routes
"""
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename
from app import db
from app.models.dataset import Dataset
from app.models.experiment import Experiment, TrainingJob
from app.models.user import User
import io
import json
import re
import requests as http_requests

datasets_bp = Blueprint('datasets', __name__)

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'jpg', 'jpeg', 'png', 'zip'}


def _normalize_minio_object_path(object_path, bucket):
    """Return an object path relative to the bucket."""
    if not object_path:
        return None

    normalized = object_path.strip().lstrip('/')
    bucket_prefix = f'{bucket}/'
    if normalized.startswith(bucket_prefix):
        normalized = normalized[len(bucket_prefix):]

    return normalized or None


def _collect_model_artifacts(experiment):
    """Collect model storage references before the experiment is deleted."""
    results = experiment.results or {}
    objects = set()

    for object_path in (results.get('model_package_path'), experiment.best_model_id):
        normalized = _normalize_minio_object_path(object_path, 'models')
        if normalized:
            objects.add(normalized)

    return {
        'prefix': f'user_{experiment.user_id}/experiment_{experiment.id}/',
        'objects': objects
    }


def _cleanup_model_artifacts(artifact_refs):
    """Best-effort cleanup for model files in MinIO."""
    try:
        from app.services.minio_service import get_minio_service
        minio_service = get_minio_service()

        prefix = artifact_refs.get('prefix')
        if prefix:
            minio_service.delete_objects('models', prefix)

        for object_name in artifact_refs.get('objects', set()):
            if prefix and object_name.startswith(prefix):
                continue
            minio_service.delete_object('models', object_name)
    except Exception as e:
        print(f"Model MinIO cleanup failed: {e}", flush=True)


def _cleanup_dataset_object(object_name):
    """Best-effort cleanup for a dataset file in MinIO."""
    if not object_name:
        return

    try:
        from app.services.minio_service import get_minio_service
        minio_service = get_minio_service()
        minio_service.delete_object('datasets', object_name)
    except Exception as e:
        print(f"Dataset MinIO cleanup failed: {e}", flush=True)


def _is_numeric_like(value):
    """Return True when a value looks like raw numeric data instead of a label."""
    text = str(value).strip()
    if not text:
        return False
    if re.fullmatch(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?', text):
        return True
    parts = re.split(r'\s+', text)
    return len(parts) > 1 and sum(_is_numeric_like(part) for part in parts) / len(parts) >= 0.8


def _snake_case_header(value, fallback):
    text = str(value or '').strip().lower()
    text = re.sub(r'[^a-z0-9]+', '_', text).strip('_')
    text = re.sub(r'_+', '_', text)
    if not text or text[0].isdigit():
        text = fallback
    return text[:64]


def _dedupe_headers(headers):
    seen = {}
    clean_headers = []
    for index, header in enumerate(headers):
        base = _snake_case_header(header, f'col_{index}')
        count = seen.get(base, 0)
        seen[base] = count + 1
        clean_headers.append(base if count == 0 else f'{base}_{count}')
    return clean_headers


def _fallback_headers(column_count):
    return [f'col_{index}' for index in range(column_count)]


def _canonical_header_guess(raw_rows, column_count, dataset=None):
    """Return trusted names for well-known headerless public datasets."""
    name_parts = [
        getattr(dataset, 'name', '') or '',
        getattr(dataset, 'file_path', '') or '',
        getattr(dataset, 'description', '') or ''
    ]
    context = ' '.join(name_parts).lower()

    if column_count == 14 and 'housing' in context and raw_rows:
        return [
            'crime_rate',
            'residential_zone',
            'industrial_area',
            'charles_river_boundary',
            'nitric_oxide_concentration',
            'average_rooms',
            'older_home_percentage',
            'employment_center_distance',
            'highway_access_index',
            'property_tax_rate',
            'pupil_teacher_ratio',
            'black_population_index',
            'lower_status_percentage',
            'median_home_value'
        ]

    return None


def _json_safe_cell(value):
    if value != value:
        return None
    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, 'isoformat'):
        try:
            return value.isoformat()
        except Exception:
            pass
    return value


def _read_tabular_bytes(file_bytes, file_type, header='infer', nrows=None):
    import pandas as pd

    if file_type in ['xlsx', 'xls']:
        return pd.read_excel(io.BytesIO(file_bytes), header=header, nrows=nrows)

    kwargs = {'header': header}
    if nrows is not None:
        kwargs['nrows'] = nrows

    def _unnamed_ratio(dataframe):
        columns = [str(col) for col in dataframe.columns]
        if not columns:
            return 0
        return sum(col.startswith('Unnamed:') for col in columns) / len(columns)

    try:
        df = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine='python', **kwargs)
    except Exception:
        df = pd.read_csv(io.BytesIO(file_bytes), **kwargs)

    if len(df.columns) == 1 or _unnamed_ratio(df) >= 0.25:
        first_line = file_bytes.decode('utf-8', errors='ignore').splitlines()[0:1]
        if first_line and len(re.split(r'\s+', first_line[0].strip())) > 1:
            whitespace_df = pd.read_csv(io.BytesIO(file_bytes), sep=r'\s+', engine='python', **kwargs)
            if len(whitespace_df.columns) >= len(df.columns) or _unnamed_ratio(whitespace_df) < _unnamed_ratio(df):
                df = whitespace_df

    return df


def _download_dataset_bytes(dataset):
    from app.services.minio_service import get_minio_service

    if dataset.file_path.startswith('local/'):
        raise ValueError('Local fallback dataset files cannot be repaired from storage')

    minio_service = get_minio_service()
    object_name = _normalize_minio_object_path(dataset.file_path, 'datasets')
    file_bytes = minio_service.download_bytes('datasets', object_name)
    if file_bytes is None:
        raise ValueError('Failed to download dataset from storage')
    return file_bytes, object_name, minio_service


def _headers_look_missing(headers):
    if not headers:
        return False
    data_like_count = sum(
        1 for header in headers
        if _is_numeric_like(header) or str(header).strip().startswith('Unnamed:')
    )
    return data_like_count / len(headers) >= 0.8


def _headers_are_too_generic(headers):
    generic_patterns = (
        r'^col_?\d+$',
        r'^column_?\d+$',
        r'^field_?\d+$',
        r'^feature_?\d+$',
        r'^value_?\d+$'
    )
    generic_count = 0
    for header in headers:
        text = str(header).strip().lower()
        if any(re.fullmatch(pattern, text) for pattern in generic_patterns):
            generic_count += 1
    return headers and generic_count / len(headers) > 0.5


def _suggest_headers_with_ai(raw_rows, column_count, dataset=None):
    canonical = _canonical_header_guess(raw_rows, column_count, dataset)
    if canonical:
        return canonical

    fallback = _fallback_headers(column_count)
    try:
        from app.services.gemini_service import get_ai_service

        dataset_context = {
            'dataset_name': getattr(dataset, 'name', None),
            'description': getattr(dataset, 'description', None),
            'file_path': getattr(dataset, 'file_path', None),
            'column_count': column_count
        }
        system_prompt = (
            "You are a senior data scientist and data dictionary author. Infer "
            "professional, domain-specific snake_case column headers for a "
            "headerless dataset. Prefer real semantic names over generic names. "
            "If the sample resembles a known public dataset, use the accepted "
            "canonical feature names. Respond with valid JSON only, no markdown."
        )
        user_message = (
            "Return exactly this JSON shape: "
            '{"headers":["descriptive_name_1","descriptive_name_2"]}\n'
            f"Dataset context: {json.dumps(dataset_context, default=str)}\n"
            f"First rows: {json.dumps(raw_rows[:5], default=str)}\n"
            "Rules:\n"
            f"- Return exactly {column_count} headers in the same column order.\n"
            "- Use concise snake_case names.\n"
            "- Do not use generic names like column_1, col_0, field_1, value_1, "
            "feature_1 unless there is truly no evidence.\n"
            "- Include units or meaning when inferable, such as rate, percentage, "
            "count, score, value, date, id.\n"
            "- No duplicates, spaces, punctuation, explanations, or markdown."
        )
        response_text = get_ai_service().call_llm(system_prompt, user_message).strip()
        if response_text.startswith('```'):
            response_text = "\n".join(response_text.splitlines()[1:-1]).strip()
        parsed = json.loads(response_text)
        headers = parsed.get('headers') if isinstance(parsed, dict) else parsed
        if not isinstance(headers, list) or len(headers) != column_count:
            return fallback
        headers = _dedupe_headers(headers)
        if _headers_are_too_generic(headers):
            return fallback
        return headers
    except Exception as e:
        print(f"AI header suggestion failed: {e}", flush=True)
        return fallback

# ── Public download URLs for sample datasets ──
# These are free, auth-free CSV links (GitHub raw, UCI, etc.)
# Datasets NOT listed here will return 404 — the frontend handles this
# by opening the Kaggle page for manual download.
SAMPLE_SOURCES = {
    # ══════════════════════════════════════
    # Medical & Health
    # ══════════════════════════════════════
    # s-med-1 (Lung Cancer): no reliable public CSV mirror — Kaggle only
    's-med-2': {
        'url': 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv',
        'filename': 'pima_indians_diabetes.csv',
        'name': 'Pima Indians Diabetes',
        'description': 'NIDDK diagnostic data for diabetes prediction',
    },
    's-med-3': {
        'url': 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer-wisconsin.csv',
        'filename': 'breast_cancer_wisconsin.csv',
        'name': 'Breast Cancer Wisconsin (Diagnostic)',
        'description': 'FNA cell nuclei features for malignant/benign classification',
    },
    's-med-4': {
        'url': 'https://raw.githubusercontent.com/MainakRepositor/Datasets/master/drug200.csv',
        'filename': 'drug_classification.csv',
        'name': 'Drug Classification',
        'description': 'Patient data to classify suitable drug type',
    },
    # s-med-5 (Stroke Prediction): no reliable public CSV mirror — Kaggle only

    # ══════════════════════════════════════
    # Agriculture
    # ══════════════════════════════════════
    's-agr-1': {
        'url': 'https://raw.githubusercontent.com/AbhishekKandoi/Crop-Yield-Prediction-based-on-Indian-Agriculture/main/Crop%20Recommendation%20dataset.csv',
        'filename': 'crop_recommendation.csv',
        'name': 'Crop Recommendation Dataset',
        'description': 'N, P, K, temperature, humidity, pH & rainfall for 22 crop recommendations',
    },
    # s-agr-2 (Crop Yield): no public CSV mirror — Kaggle only
    # s-agr-3 (Plant Diseases): image dataset / ZIP — Kaggle only
    # s-agr-4 (Rainfall India): no public CSV mirror — Kaggle only
    # s-agr-5 (Fertilizer): no public CSV mirror — Kaggle only

    # ══════════════════════════════════════
    # Finance
    # ══════════════════════════════════════
    # s-fin-1 (Loan Approval): no public CSV mirror — Kaggle only
    # s-fin-2 (Credit Card Fraud): 144 MB — too large, Kaggle only
    # s-fin-3 (Stock Market): no public CSV mirror — Kaggle only
    # s-fin-4 (Bank Churn): no public CSV mirror — Kaggle only
    's-fin-5': {
        'url': 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv',
        'filename': 'insurance_charges.csv',
        'name': 'Medical Insurance Charges',
        'description': 'Age, sex, BMI, children, smoker & region for medical cost prediction',
    },

    # ══════════════════════════════════════
    # E-Commerce
    # ══════════════════════════════════════
    # All e-commerce datasets are Kaggle-only (large or proprietary)

    # ══════════════════════════════════════
    # Education
    # ══════════════════════════════════════
    # All education datasets: no verified public CSV mirrors — Kaggle only

    # ══════════════════════════════════════
    # Logistics
    # ══════════════════════════════════════
    's-log-2': {
        'url': 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv',
        'filename': 'auto_mpg.csv',
        'name': 'Auto MPG Dataset',
        'description': 'Car displacement, horsepower, weight & acceleration for miles-per-gallon prediction',
    },
    # s-log-1 (Delivery Logistics): no public CSV mirror — Kaggle only
    # s-log-3 (Flight Price): no public CSV mirror — Kaggle only
    # s-log-4 (Supply Chain): no public CSV mirror — Kaggle only
    # s-log-5 (Vehicle CarDekho): no public CSV mirror — Kaggle only

    # ══════════════════════════════════════
    # Analytics
    # ══════════════════════════════════════
    's-ana-1': {
        'url': 'https://raw.githubusercontent.com/MainakRepositor/Datasets/master/Mall_Customers.csv',
        'filename': 'mall_customers.csv',
        'name': 'Mall Customer Segmentation',
        'description': 'Customer ID, age, gender, annual income & spending score for clustering',
    },
    's-ana-2': {
        'url': 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv',
        'filename': 'iris_flower_dataset.csv',
        'name': 'Iris Flower Dataset',
        'description': 'Classic Fisher dataset for 3 iris species classification',
    },
    's-ana-3': {
        'url': 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv',
        'filename': 'titanic.csv',
        'name': 'Titanic — Machine Learning',
        'description': 'Passenger data for survival prediction',
    },
    's-ana-4': {
        'url': 'https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv',
        'filename': 'california_housing.csv',
        'name': 'California Housing Prices',
        'description': '1990 census block-group data for median house value regression',
    },
    's-ana-5': {
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',
        'filename': 'red_wine_quality.csv',
        'name': 'Red Wine Quality',
        'description': 'Physicochemical inputs for quality score prediction',
    },

    # ══════════════════════════════════════
    # Entertainment
    # ══════════════════════════════════════
    # All entertainment datasets: large or Kaggle-only
}

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'jpg', 'jpeg', 'png', 'zip'}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@datasets_bp.route('', methods=['GET'])
@jwt_required()
def list_datasets():
    """List all datasets for current user"""
    user_id = int(get_jwt_identity())
    datasets = Dataset.query.filter_by(user_id=user_id).order_by(Dataset.created_at.desc()).all()
    
    return jsonify({
        'datasets': [d.to_dict() for d in datasets],
        'total': len(datasets)
    }), 200


@datasets_bp.route('/upload', methods=['POST'])
@jwt_required()
def upload_dataset():
    """Upload a new dataset"""
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
        
    if user.plan_type == 'free':
        dataset_count = Dataset.query.filter_by(user_id=user_id).count()
        if dataset_count >= 3:
            return jsonify({'error': 'Free plan limit reached (3 datasets max). Upgrade to Pro for unlimited datasets.'}), 403
    
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Allowed: {ALLOWED_EXTENSIONS}'}), 400
    
    filename = secure_filename(file.filename)
    name = request.form.get('name', filename)
    description = request.form.get('description', '')
    
    file_type = filename.rsplit('.', 1)[1].lower()
    
    # Read file content for size calculation
    file_content = file.read()
    file_size = len(file_content)
    file.seek(0)  # Reset file pointer
    
    # Upload to MinIO
    try:
        from app.services.minio_service import get_minio_service
        minio_service = get_minio_service()
        
        file_path = f'{user_id}/{filename}'
        minio_service.upload_bytes(
            bucket='datasets',
            object_name=file_path,
            data=file_content,
            content_type=file.content_type or 'application/octet-stream'
        )
    except Exception as e:
        # If MinIO fails, continue without it (for development)
        print(f"MinIO upload failed: {e}")
        file_path = f'local/{user_id}/{filename}'
    
    # Create database record
    dataset = Dataset(
        name=name,
        description=description,
        file_path=file_path,
        file_type=file_type,
        file_size=file_size,
        user_id=user_id,
        profile_status='pending'
    )
    
    db.session.add(dataset)
    db.session.commit()
    
    # For tabular data, do quick profiling synchronously (for small files)
    if file_type in ['csv', 'xlsx', 'xls'] and file_size < 10 * 1024 * 1024:  # < 10MB
        try:
            import pandas as pd
            import io
            
            if file_type == 'csv':
                df = pd.read_csv(io.BytesIO(file_content))
            else:
                df = pd.read_excel(io.BytesIO(file_content))
            
            # Basic profiling
            dataset.num_rows = len(df)
            dataset.num_columns = len(df.columns)
            dataset.data_type = 'tabular'
            dataset.column_info = {
                col: {'dtype': str(df[col].dtype), 'null_count': int(df[col].isnull().sum())}
                for col in df.columns
            }
            dataset.profile_status = 'completed'
            db.session.commit()
            
            # Trigger advanced profiling in background for Visual Insights
            from app.tasks.training_tasks import profile_dataset_task
            profile_dataset_task.delay(dataset.id)
        except Exception as e:
            print(f"Profiling failed: {e}")
            dataset.profile_status = 'failed'
            db.session.commit()
    
    # For ZIP files, check if it's an image dataset (folder-per-class structure)
    elif file_type == 'zip':
        try:
            import zipfile
            import io
            
            IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff'}
            
            with zipfile.ZipFile(io.BytesIO(file_content), 'r') as zf:
                # Get all file entries (skip directories and hidden files)
                all_entries = [
                    name for name in zf.namelist()
                    if not name.endswith('/') and not name.split('/')[-1].startswith('.')
                ]
                
                # Detect folder-per-class structure:
                # Expected: root_folder/class_name/image.jpg OR class_name/image.jpg
                class_counts = {}
                class_samples = {}
                
                for entry in all_entries:
                    parts = entry.split('/')
                    # Skip files not in a subfolder
                    if len(parts) < 2:
                        continue
                    
                    # Get file extension
                    ext = '.' + parts[-1].rsplit('.', 1)[-1].lower() if '.' in parts[-1] else ''
                    if ext not in IMAGE_EXTENSIONS:
                        continue
                    
                    # Determine class name (handle optional root folder)
                    # If structure is root/class/img.jpg → class is parts[-2]
                    # If structure is class/img.jpg → class is parts[0]
                    if len(parts) == 2:
                        class_name = parts[0]
                    elif len(parts) == 3:
                        class_name = parts[1]
                    else:
                        # Deeper nesting — use second-to-last folder
                        class_name = parts[-2]
                    
                    # Skip __MACOSX and other system folders
                    if class_name.startswith('__') or class_name.startswith('.'):
                        continue
                    
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    if class_name not in class_samples:
                        class_samples[class_name] = []
                    if len(class_samples[class_name]) < 3:
                        class_samples[class_name].append(parts[-1])
                
                if class_counts and len(class_counts) >= 2:
                    # It's an image classification dataset
                    total_images = sum(class_counts.values())
                    dataset.data_type = 'image'
                    dataset.file_type = 'image_zip'
                    dataset.num_rows = total_images
                    dataset.num_columns = len(class_counts)  # num classes
                    dataset.column_info = {
                        'classes': list(class_counts.keys()),
                        'counts': class_counts,
                        'total_images': total_images,
                        'num_classes': len(class_counts),
                        'samples': class_samples,
                    }
                    dataset.profile_status = 'completed'
                    print(f"📸 Image dataset detected: {len(class_counts)} classes, {total_images} images")
                else:
                    # ZIP but not a recognized image dataset
                    dataset.data_type = 'unknown'
                    dataset.profile_status = 'completed'
                    print(f"📦 ZIP uploaded but not recognized as image dataset")
                
                db.session.commit()
        except Exception as e:
            print(f"ZIP profiling failed: {e}")
            dataset.profile_status = 'failed'
            db.session.commit()
    
    
    # Send dataset upload notification
    try:
        from app.services.notification_service import notify_dataset_uploaded
        notify_dataset_uploaded(user, name)
    except Exception as notify_err:
        print(f'[notify] dataset_uploaded dispatch error: {notify_err}', flush=True)

    return jsonify({
        'message': 'Dataset uploaded successfully',
        'dataset': dataset.to_dict()
    }), 201


@datasets_bp.route('/<int:dataset_id>', methods=['GET'])
@jwt_required()
def get_dataset(dataset_id):
    """Get dataset details"""
    user_id = int(get_jwt_identity())
    dataset = Dataset.query.filter_by(id=dataset_id, user_id=user_id).first()
    
    if not dataset:
        return jsonify({'error': 'Dataset not found'}), 404
    
    return jsonify({'dataset': dataset.to_dict()}), 200


@datasets_bp.route('/<int:dataset_id>/profile', methods=['GET'])
@jwt_required()
def get_dataset_profile(dataset_id):
    """Get dataset profile (stats, types, distributions)"""
    user_id = int(get_jwt_identity())
    dataset = Dataset.query.filter_by(id=dataset_id, user_id=user_id).first()
    
    if not dataset:
        return jsonify({'error': 'Dataset not found'}), 404
    
    return jsonify({
        'dataset_id': dataset.id,
        'profile_status': dataset.profile_status,
        'profile_data': dataset.profile_data,
        'column_info': dataset.column_info
    }), 200


@datasets_bp.route('/<int:dataset_id>/detect-headers', methods=['GET'])
@jwt_required()
def detect_dataset_headers(dataset_id):
    """Detect whether a tabular dataset is missing a real header row."""
    user_id = int(get_jwt_identity())
    dataset = Dataset.query.filter_by(id=dataset_id, user_id=user_id).first()

    if not dataset:
        return jsonify({'error': 'Dataset not found'}), 404

    if dataset.file_type not in ['csv', 'xlsx', 'xls']:
        return jsonify({'error': 'Header detection is only available for tabular datasets'}), 400

    try:
        file_bytes, _, _ = _download_dataset_bytes(dataset)
        raw_df = _read_tabular_bytes(file_bytes, dataset.file_type, header=None, nrows=4)
        current_df = _read_tabular_bytes(file_bytes, dataset.file_type, header='infer', nrows=3)

        raw_rows = [
            [_json_safe_cell(value) for value in row]
            for row in raw_df.values.tolist()
        ]
        current_headers = [str(col).strip() for col in current_df.columns]
        column_count = len(raw_df.columns)
        has_headers = not _headers_look_missing(current_headers)
        suggested_headers = [] if has_headers else _suggest_headers_with_ai(raw_rows[:3], column_count, dataset)

        return jsonify({
            'has_headers': has_headers,
            'raw_rows': raw_rows,
            'current_headers': current_headers,
            'column_count': column_count,
            'suggested_headers': suggested_headers
        }), 200
    except Exception as e:
        return jsonify({'error': f'Failed to detect headers: {str(e)}'}), 500


@datasets_bp.route('/<int:dataset_id>/set-headers', methods=['POST'])
@jwt_required()
def set_dataset_headers(dataset_id):
    """Apply confirmed headers to a stored tabular dataset and refresh metadata."""
    user_id = int(get_jwt_identity())
    dataset = Dataset.query.filter_by(id=dataset_id, user_id=user_id).first()

    if not dataset:
        return jsonify({'error': 'Dataset not found'}), 404

    if dataset.file_type not in ['csv', 'xlsx', 'xls']:
        return jsonify({'error': 'Headers can only be set for tabular datasets'}), 400

    data = request.get_json(silent=True) or {}
    requested_headers = data if isinstance(data, list) else data.get('headers')
    if not isinstance(requested_headers, list) or not requested_headers:
        return jsonify({'error': 'headers must be a non-empty array'}), 400

    try:
        file_bytes, object_name, minio_service = _download_dataset_bytes(dataset)
        df = _read_tabular_bytes(file_bytes, dataset.file_type, header=None)

        if len(requested_headers) != len(df.columns):
            return jsonify({
                'error': f'Expected {len(df.columns)} headers, received {len(requested_headers)}'
            }), 400

        headers = _dedupe_headers(requested_headers)
        df.columns = headers

        output = io.BytesIO()
        content_type = 'text/csv'
        if dataset.file_type in ['xlsx', 'xls']:
            df.to_excel(output, index=False)
            content_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        else:
            df.to_csv(output, index=False)
        output_bytes = output.getvalue()

        uploaded = minio_service.upload_bytes(
            bucket='datasets',
            object_name=object_name,
            data=output_bytes,
            content_type=content_type
        )
        if not uploaded:
            return jsonify({'error': 'Failed to write updated dataset to storage'}), 500

        dataset.file_size = len(output_bytes)
        dataset.num_rows = len(df)
        dataset.num_columns = len(df.columns)
        dataset.data_type = 'tabular'
        dataset.column_info = {
            col: {'dtype': str(df[col].dtype), 'null_count': int(df[col].isnull().sum())}
            for col in df.columns
        }
        dataset.profile_status = 'pending'
        db.session.commit()

        try:
            from app.tasks.training_tasks import profile_dataset_task
            profile_dataset_task.delay(dataset.id)
        except Exception as e:
            print(f"Failed to enqueue dataset profiling after header update: {e}", flush=True)

        return jsonify({
            'message': 'Dataset headers updated successfully',
            'dataset': dataset.to_dict(),
            'headers': headers
        }), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Failed to set headers: {str(e)}'}), 500


@datasets_bp.route('/<int:dataset_id>/preview', methods=['GET'])
@jwt_required()
def preview_dataset(dataset_id):
    """Get dataset content for the viewer page (up to 500 rows)"""
    import pandas as pd
    import numpy as np

    user_id = int(get_jwt_identity())
    dataset = Dataset.query.filter_by(id=dataset_id, user_id=user_id).first()

    if not dataset:
        return jsonify({'error': 'Dataset not found'}), 404

    if dataset.data_type == 'image':
        return jsonify({'error': 'Preview not supported for image datasets'}), 400

    try:
        from app.services.minio_service import get_minio_service
        minio_service = get_minio_service()
        file_content = minio_service.download_bytes('datasets', dataset.file_path)

        if not file_content:
            return jsonify({'error': 'Could not download dataset file'}), 500

        # Read into DataFrame
        file_type = dataset.file_type or ''
        if file_type in ['xlsx', 'xls']:
            df = pd.read_excel(io.BytesIO(file_content))
        else:
            df = pd.read_csv(io.BytesIO(file_content))

        total_rows = len(df)
        total_columns = len(df.columns)

        # Build column statistics
        columns_info = []
        for col in df.columns:
            series = df[col]
            col_info = {
                'name': col,
                'dtype': str(series.dtype),
                'null_count': int(series.isnull().sum()),
                'unique_count': int(series.nunique()),
            }
            if np.issubdtype(series.dtype, np.number):
                col_info['min'] = float(series.min()) if not pd.isna(series.min()) else None
                col_info['max'] = float(series.max()) if not pd.isna(series.max()) else None
                col_info['mean'] = round(float(series.mean()), 4) if not pd.isna(series.mean()) else None
                col_info['type'] = 'numeric'
            elif series.dtype == 'object':
                col_info['top_values'] = series.value_counts().head(5).to_dict()
                col_info['type'] = 'categorical'
            else:
                col_info['type'] = 'other'
            columns_info.append(col_info)

        # Cap at 500 rows
        preview_df = df.head(500)
        # Replace NaN/Inf with None for JSON serialization
        preview_df = preview_df.replace([np.inf, -np.inf], None)
        preview_df = preview_df.where(pd.notnull(preview_df), None)

        rows = preview_df.to_dict(orient='records')

        return jsonify({
            'dataset_id': dataset.id,
            'name': dataset.name,
            'file_type': dataset.file_type,
            'total_rows': total_rows,
            'total_columns': total_columns,
            'preview_rows': len(rows),
            'truncated': total_rows > 500,
            'columns': columns_info,
            'rows': rows
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Preview failed: {str(e)}'}), 500


@datasets_bp.route('/<int:dataset_id>', methods=['DELETE'])
@jwt_required()
def delete_dataset(dataset_id):
    """Delete a dataset and related experiments"""
    user_id = int(get_jwt_identity())
    dataset = Dataset.query.filter_by(id=dataset_id, user_id=user_id).first()

    if not dataset:
        return jsonify({'error': 'Dataset not found'}), 404

    dataset_object = None
    if dataset.file_path and not dataset.file_path.startswith('local/'):
        shared_dataset = Dataset.query.filter(
            Dataset.id != dataset.id,
            Dataset.file_path == dataset.file_path
        ).first()
        if not shared_dataset:
            dataset_object = _normalize_minio_object_path(dataset.file_path, 'datasets')

    experiments = Experiment.query.filter_by(dataset_id=dataset.id, user_id=user_id).all()
    model_artifacts = [_collect_model_artifacts(experiment) for experiment in experiments]

    try:
        for experiment in experiments:
            TrainingJob.query.filter_by(experiment_id=experiment.id).delete(synchronize_session=False)
            db.session.delete(experiment)

        db.session.delete(dataset)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"Error deleting dataset {dataset_id}: {e}", flush=True)
        return jsonify({'error': f'Failed to delete dataset: {str(e)}'}), 500

    for artifact_refs in model_artifacts:
        _cleanup_model_artifacts(artifact_refs)
    _cleanup_dataset_object(dataset_object)

    return jsonify({'message': 'Dataset deleted successfully'}), 200


@datasets_bp.route('/use-sample', methods=['POST'])
@jwt_required()
def use_sample_dataset():
    """Download a sample dataset and add it to the user's account"""
    user_id = int(get_jwt_identity())
    
    data = request.get_json()
    sample_id = data.get('sample_id')
    
    if not sample_id:
        return jsonify({'error': 'sample_id is required'}), 400
    
    # Look up the sample source
    source = SAMPLE_SOURCES.get(sample_id)
    if not source:
        return jsonify({'error': f'Sample dataset "{sample_id}" is not available for direct download yet. Try "View on Kaggle" to download manually.'}), 404
    
    # Check if user already has this dataset
    existing = Dataset.query.filter_by(
        user_id=user_id,
        name=source['name']
    ).first()
    if existing:
        return jsonify({
            'message': 'Dataset already in your uploads',
            'dataset': existing.to_dict()
        }), 200
    
    # Download the CSV from public URL
    try:
        resp = http_requests.get(source['url'], timeout=60)
        resp.raise_for_status()
        file_content = resp.content
    except Exception as e:
        return jsonify({'error': f'Failed to download dataset: {str(e)}'}), 502
    
    filename = source['filename']
    file_type = filename.rsplit('.', 1)[1].lower()
    file_size = len(file_content)
    
    # Upload to MinIO
    try:
        from app.services.minio_service import get_minio_service
        minio_service = get_minio_service()
        
        file_path = f'{user_id}/{filename}'
        minio_service.upload_bytes(
            bucket='datasets',
            object_name=file_path,
            data=file_content,
            content_type='text/csv'
        )
    except Exception as e:
        print(f"MinIO upload failed: {e}")
        file_path = f'local/{user_id}/{filename}'
    
    # Create database record
    dataset = Dataset(
        name=source['name'],
        description=source.get('description', ''),
        file_path=file_path,
        file_type=file_type,
        file_size=file_size,
        user_id=user_id,
        profile_status='pending'
    )
    
    db.session.add(dataset)
    db.session.commit()
    
    # Quick profiling for tabular data
    if file_type in ['csv', 'xlsx', 'xls'] and file_size < 50 * 1024 * 1024:
        try:
            import pandas as pd
            
            if file_type == 'csv':
                df = pd.read_csv(io.BytesIO(file_content))
            else:
                df = pd.read_excel(io.BytesIO(file_content))
            
            dataset.num_rows = len(df)
            dataset.num_columns = len(df.columns)
            dataset.data_type = 'tabular'
            dataset.column_info = {
                col: {'dtype': str(df[col].dtype), 'null_count': int(df[col].isnull().sum())}
                for col in df.columns
            }
            dataset.profile_status = 'completed'
            db.session.commit()
            
            # Trigger advanced profiling in background for Visual Insights
            from app.tasks.training_tasks import profile_dataset_task
            profile_dataset_task.delay(dataset.id)
        except Exception as e:
            print(f"Profiling failed: {e}")
            dataset.profile_status = 'failed'
            db.session.commit()
    
    return jsonify({
        'message': 'Sample dataset added to your uploads',
        'dataset': dataset.to_dict()
    }), 201


# ══════════════════════════════════════════════════════════
# Kaggle API Integration — Search & Download
# ══════════════════════════════════════════════════════════
import os
import tempfile
import glob as _glob

def _get_kaggle_api():
    """Initialise and return an authenticated KaggleApi instance."""
    # Try standard Kaggle env vars first
    username = os.environ.get('KAGGLE_USERNAME')
    key = os.environ.get('KAGGLE_KEY')

    # If not found, try combined token username:key
    if not username or not key:
        token = os.environ.get('KAGGLE_API_TOKEN', '')
        if ':' in token:
            username, key = token.split(':', 1)
            os.environ['KAGGLE_USERNAME'] = username
            os.environ['KAGGLE_KEY'] = key
        else:
            # Fallback for dev if nothing else is provided
            os.environ['KAGGLE_USERNAME'] = 'alok920'
            os.environ['KAGGLE_KEY'] = '5547498f46540188897a92e3222effbe'

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


@datasets_bp.route('/kaggle/search', methods=['GET'])
@jwt_required()
def kaggle_search():
    """Search Kaggle datasets by query string."""
    query = request.args.get('query', '').strip()
    if not query:
        return jsonify({'error': 'Search query is required'}), 400

    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)

    try:
        api = _get_kaggle_api()
        # Search datasets
        datasets = api.dataset_list(search=query, max_size=1000000)
    except Exception as e:
        return jsonify({'error': f'Kaggle API error: {str(e)}'}), 502

    is_limited = False
    if user and user.plan_type == 'free':
        datasets = datasets[:5]
        is_limited = True

    results = []
    for ds in datasets:
        # ds is a dict-like object from kaggle SDK
        ref = ds.ref if hasattr(ds, 'ref') else (ds.get('ref') if isinstance(ds, dict) else str(ds))
        title = ds.title if hasattr(ds, 'title') else (ds.get('title') if isinstance(ds, dict) else ref)
        subtitle = ds.subtitle if hasattr(ds, 'subtitle') else (ds.get('subtitle') if isinstance(ds, dict) else '')
        total_bytes = ds.totalBytes if hasattr(ds, 'totalBytes') else (ds.get('totalBytes') if isinstance(ds, dict) else 0)
        download_count = ds.downloadCount if hasattr(ds, 'downloadCount') else (ds.get('downloadCount') if isinstance(ds, dict) else 0)
        vote_count = ds.voteCount if hasattr(ds, 'voteCount') else (ds.get('voteCount') if isinstance(ds, dict) else 0)
        last_updated = ds.lastUpdated if hasattr(ds, 'lastUpdated') else (ds.get('lastUpdated') if isinstance(ds, dict) else '')
        usability = ds.usabilityRating if hasattr(ds, 'usabilityRating') else (ds.get('usabilityRating') if isinstance(ds, dict) else 0)

        results.append({
            'ref': ref,
            'title': title,
            'subtitle': subtitle or '',
            'totalBytes': total_bytes or 0,
            'downloadCount': download_count or 0,
            'voteCount': vote_count or 0,
            'lastUpdated': str(last_updated) if last_updated else '',
            'usabilityRating': round(float(usability or 0), 2),
            'url': f'https://www.kaggle.com/datasets/{ref}',
        })

    return jsonify({
        'query': query,
        'is_limited': is_limited,
        'results': results
    }), 200


@datasets_bp.route('/kaggle/download', methods=['POST'])
@jwt_required()
def kaggle_download():
    """Download a Kaggle dataset by ref and add it to the user's library.

    Body JSON:
        ref   – Kaggle dataset ref, e.g. "heptapod/titanic" (required)
        name  – optional display name
    """
    user_id = int(get_jwt_identity())
    data = request.get_json()
    ref = data.get('ref', '').strip()

    if not ref:
        return jsonify({'error': 'Dataset "ref" is required'}), 400

    display_name = data.get('name', ref.split('/')[-1].replace('-', ' ').title())

    # Check duplicate
    existing = Dataset.query.filter_by(user_id=user_id, name=display_name).first()
    if existing:
        return jsonify({'message': 'Dataset already in your uploads', 'dataset': existing.to_dict()}), 200

    # Download to temp dir
    tmp_dir = tempfile.mkdtemp(prefix='kaggle_')
    try:
        api = _get_kaggle_api()
        api.dataset_download_files(ref, path=tmp_dir, unzip=True)
    except Exception as e:
        return jsonify({'error': f'Kaggle download failed: {str(e)}'}), 502

    # Find first CSV / Excel file in downloaded contents
    patterns = ['*.csv', '*.xlsx', '*.xls']
    found_file = None
    for pattern in patterns:
        matches = _glob.glob(os.path.join(tmp_dir, '**', pattern), recursive=True)
        if matches:
            # Prefer the largest file (likely the main dataset)
            found_file = max(matches, key=os.path.getsize)
            break

    if not found_file:
        # Cleanup
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return jsonify({'error': 'No CSV or Excel file found in the downloaded dataset'}), 422

    # Read file content
    with open(found_file, 'rb') as f:
        file_content = f.read()

    filename = secure_filename(os.path.basename(found_file))
    file_type = filename.rsplit('.', 1)[1].lower()
    file_size = len(file_content)

    # Upload to MinIO
    try:
        from app.services.minio_service import get_minio_service
        minio_service = get_minio_service()
        file_path = f'{user_id}/{filename}'
        minio_service.upload_bytes(
            bucket='datasets',
            object_name=file_path,
            data=file_content,
            content_type='text/csv' if file_type == 'csv' else 'application/octet-stream'
        )
    except Exception as e:
        print(f"MinIO upload failed: {e}")
        file_path = f'local/{user_id}/{filename}'

    # Create DB record
    dataset = Dataset(
        name=display_name,
        description=f'Imported from Kaggle: {ref}',
        file_path=file_path,
        file_type=file_type,
        file_size=file_size,
        user_id=user_id,
        profile_status='pending'
    )
    db.session.add(dataset)
    db.session.commit()

    # Quick profiling
    if file_type in ['csv', 'xlsx', 'xls'] and file_size < 50 * 1024 * 1024:
        try:
            import pandas as pd
            if file_type == 'csv':
                df = pd.read_csv(io.BytesIO(file_content))
            else:
                df = pd.read_excel(io.BytesIO(file_content))

            dataset.num_rows = len(df)
            dataset.num_columns = len(df.columns)
            dataset.data_type = 'tabular'
            dataset.column_info = {
                col: {'dtype': str(df[col].dtype), 'null_count': int(df[col].isnull().sum())}
                for col in df.columns
            }
            dataset.profile_status = 'completed'
            db.session.commit()

            from app.tasks.training_tasks import profile_dataset_task
            profile_dataset_task.delay(dataset.id)
        except Exception as e:
            print(f"Profiling failed: {e}")
            dataset.profile_status = 'failed'
            db.session.commit()

    # Cleanup temp dir
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return jsonify({
        'message': f'Kaggle dataset "{display_name}" added to your uploads',
        'dataset': dataset.to_dict()
    }), 201


# ═══════════════════════════════════════════════════════
# AI-POWERED SYNTHETIC DATA GENERATION
# ═══════════════════════════════════════════════════════

@datasets_bp.route('/ai/schema', methods=['POST'])
@jwt_required()
def ai_generate_schema():
    """Use AI to generate a dataset schema from a natural language prompt"""
    from app.services.gemini_service import get_ai_service

    data = request.get_json()
    prompt = data.get('prompt', '').strip()

    if not prompt:
        return jsonify({'error': 'Please describe the dataset you want to create'}), 400

    try:
        ai = get_ai_service()
        schema = ai.generate_dataset_schema(prompt, num_columns=data.get('num_columns', 8))

        if 'error' in schema:
            return jsonify({'error': schema['error']}), 500

        return jsonify({'schema': schema}), 200

    except Exception as e:
        print(f"AI schema generation error: {e}", flush=True)
        return jsonify({'error': f'AI service error: {str(e)}'}), 500


@datasets_bp.route('/ai/generate', methods=['POST'])
@jwt_required()
def ai_generate_data():
    """Generate realistic data using AI seed rows + programmatic extension"""
    import pandas as pd
    import numpy as np
    import random
    import string
    from datetime import datetime, timedelta

    user_id = int(get_jwt_identity())
    data = request.get_json()

    name = data.get('name', 'ai_dataset').strip()
    num_rows = min(int(data.get('num_rows', 1000)), 100000)
    columns = data.get('columns', [])
    save = data.get('save', False)

    if not columns:
        return jsonify({'error': 'Schema columns are required'}), 400

    # Step 1: Get AI seed rows (30 rows for pattern)
    try:
        from app.services.gemini_service import get_ai_service
        ai = get_ai_service()
        seed_rows = ai.generate_dataset_rows(columns, num_sample=min(30, num_rows))
    except Exception as e:
        print(f"AI row generation error: {e}", flush=True)
        seed_rows = []

    # Step 2: Build full dataset
    if seed_rows and len(seed_rows) >= 5:
        # Use AI seed data as base, extend programmatically if needed
        seed_df = pd.DataFrame(seed_rows)

        if num_rows <= len(seed_rows):
            df = seed_df.head(num_rows)
        else:
            # Extend by sampling distributions from seed data
            extended_rows = []
            for _ in range(num_rows - len(seed_rows)):
                row = {}
                for col_def in columns:
                    col_name = col_def['name']
                    col_type = col_def.get('type', 'text')
                    config = col_def.get('config', {})

                    if col_name in seed_df.columns:
                        seed_values = seed_df[col_name].dropna().tolist()
                    else:
                        seed_values = []

                    if col_type == 'id':
                        prefix = config.get('prefix', 'ID')
                        row[col_name] = f"{prefix}_{len(seed_rows) + len(extended_rows) + 1:05d}"
                    elif col_type in ('numeric', 'integer') and seed_values:
                        numeric_vals = [v for v in seed_values if isinstance(v, (int, float))]
                        if numeric_vals:
                            mean_val = np.mean(numeric_vals)
                            std_val = max(np.std(numeric_vals), 0.1)
                            val = np.random.normal(mean_val, std_val)
                            low = float(config.get('min', min(numeric_vals)))
                            high = float(config.get('max', max(numeric_vals)))
                            val = np.clip(val, low, high)
                            row[col_name] = int(round(val)) if col_type == 'integer' else round(float(val), 2)
                        else:
                            row[col_name] = 0
                    elif col_type == 'categorical' and seed_values:
                        row[col_name] = random.choice(seed_values)
                    elif col_type == 'boolean':
                        prob = float(config.get('true_probability', 0.5))
                        row[col_name] = random.random() < prob
                    elif col_type == 'date':
                        start = datetime(2020, 1, 1)
                        end = datetime(2025, 12, 31)
                        delta = (end - start).days
                        row[col_name] = (start + timedelta(days=random.randint(0, delta))).strftime('%Y-%m-%d')
                    elif seed_values:
                        row[col_name] = random.choice(seed_values)
                    else:
                        row[col_name] = ''

                extended_rows.append(row)

            df = pd.concat([seed_df, pd.DataFrame(extended_rows)], ignore_index=True)
    else:
        # Fallback: pure programmatic generation (no AI seed)
        generated = {}
        for col_def in columns:
            col_name = col_def['name']
            col_type = col_def.get('type', 'text')
            config = col_def.get('config', {})

            if col_type == 'numeric':
                low = float(config.get('min', 0))
                high = float(config.get('max', 100))
                generated[col_name] = np.round(np.random.uniform(low, high, num_rows), 2)
            elif col_type == 'integer':
                low = int(config.get('min', 0))
                high = int(config.get('max', 100))
                generated[col_name] = np.random.randint(low, high + 1, num_rows)
            elif col_type == 'categorical':
                cats = config.get('categories', ['A', 'B', 'C'])
                generated[col_name] = np.random.choice(cats, num_rows)
            elif col_type == 'boolean':
                prob = float(config.get('true_probability', 0.5))
                generated[col_name] = np.random.choice([True, False], num_rows, p=[prob, 1 - prob])
            elif col_type == 'date':
                start = datetime(2020, 1, 1)
                end = datetime(2025, 12, 31)
                delta = (end - start).days
                generated[col_name] = [(start + timedelta(days=random.randint(0, delta))).strftime('%Y-%m-%d') for _ in range(num_rows)]
            elif col_type == 'id':
                prefix = config.get('prefix', 'ID')
                generated[col_name] = [f'{prefix}_{i+1:05d}' for i in range(num_rows)]
            else:
                length = int(config.get('length', 10))
                generated[col_name] = [''.join(random.choices(string.ascii_lowercase + ' ', k=length)).strip() for _ in range(num_rows)]

        df = pd.DataFrame(generated)

    # Build preview (first 10 rows)
    preview = df.head(10).to_dict(orient='records')

    # If save=True, persist to MinIO + DB
    if save:
        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue()
        file_size = len(csv_bytes)

        filename = f'{name.replace(" ", "_").lower()}.csv'
        file_path = f'{user_id}/ai_{filename}'

        try:
            from app.services.minio_service import get_minio_service
            minio_service = get_minio_service()
            minio_service.upload_bytes(
                bucket='datasets',
                object_name=file_path,
                data=csv_bytes,
                content_type='text/csv'
            )
        except Exception as e:
            print(f"MinIO upload failed: {e}")
            file_path = f'local/{user_id}/ai_{filename}'

        dataset_obj = Dataset(
            name=name,
            description=f'AI-generated dataset with {num_rows} rows and {len(columns)} columns',
            file_path=file_path,
            file_type='csv',
            file_size=file_size,
            data_type='tabular',
            num_rows=num_rows,
            num_columns=len(columns),
            column_info={
                col_name: {'dtype': str(df[col_name].dtype), 'null_count': 0}
                for col_name in df.columns
            },
            user_id=user_id,
            profile_status='completed'
        )

        db.session.add(dataset_obj)
        db.session.commit()

        return jsonify({
            'message': f'AI dataset "{name}" saved with {num_rows} rows',
            'dataset': dataset_obj.to_dict(),
            'preview': preview,
            'saved': True
        }), 201

    # Preview only (not saved yet)
    return jsonify({
        'preview': preview,
        'total_rows': num_rows,
        'columns': [c['name'] for c in columns],
        'saved': False
    }), 200


@datasets_bp.route('/generate-synthetic', methods=['POST'])
@jwt_required()
def generate_synthetic():
    """Generate a synthetic dataset from a user-defined schema"""
    import pandas as pd
    import numpy as np
    import io
    import random
    import string
    from datetime import datetime, timedelta

    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404

    if user.plan_type == 'free':
        return jsonify({'error': 'Synthetic data generation is locked on the Free tier. Please upgrade.'}), 403

    data = request.get_json()

    name = data.get('name', 'synthetic_data').strip()
    num_rows = min(int(data.get('num_rows', 1000)), 100000)  # Cap at 100k
    columns = data.get('columns', [])

    if not columns:
        return jsonify({'error': 'At least one column is required'}), 400
    if not name:
        return jsonify({'error': 'Dataset name is required'}), 400

    # Generate data for each column
    generated = {}
    for col in columns:
        col_name = col.get('name', 'column').strip()
        col_type = col.get('type', 'numeric')

        if col_type == 'numeric':
            low = float(col.get('min', 0))
            high = float(col.get('max', 100))
            decimals = int(col.get('decimals', 2))
            generated[col_name] = np.round(np.random.uniform(low, high, num_rows), decimals)

        elif col_type == 'integer':
            low = int(col.get('min', 0))
            high = int(col.get('max', 100))
            generated[col_name] = np.random.randint(low, high + 1, num_rows)

        elif col_type == 'categorical':
            categories = col.get('categories', ['A', 'B', 'C'])
            if isinstance(categories, str):
                categories = [c.strip() for c in categories.split(',')]
            generated[col_name] = np.random.choice(categories, num_rows)

        elif col_type == 'boolean':
            prob = float(col.get('true_probability', 0.5))
            generated[col_name] = np.random.choice([True, False], num_rows, p=[prob, 1 - prob])

        elif col_type == 'date':
            start = datetime(2020, 1, 1)
            end = datetime(2025, 12, 31)
            delta = (end - start).days
            dates = [start + timedelta(days=random.randint(0, delta)) for _ in range(num_rows)]
            generated[col_name] = [d.strftime('%Y-%m-%d') for d in dates]

        elif col_type == 'text':
            length = int(col.get('length', 10))
            generated[col_name] = [
                ''.join(random.choices(string.ascii_lowercase + ' ', k=length)).strip()
                for _ in range(num_rows)
            ]

        elif col_type == 'id':
            prefix = col.get('prefix', 'ID')
            generated[col_name] = [f'{prefix}_{i+1:05d}' for i in range(num_rows)]

        else:
            generated[col_name] = np.random.uniform(0, 100, num_rows)

    df = pd.DataFrame(generated)

    # Convert to CSV bytes
    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue()
    file_size = len(csv_bytes)

    filename = f'{name.replace(" ", "_").lower()}.csv'
    file_path = f'{user_id}/synthetic_{filename}'

    # Upload to MinIO
    try:
        from app.services.minio_service import get_minio_service
        minio_service = get_minio_service()
        minio_service.upload_bytes(
            bucket='datasets',
            object_name=file_path,
            data=csv_bytes,
            content_type='text/csv'
        )
    except Exception as e:
        print(f"MinIO upload failed: {e}")
        file_path = f'local/{user_id}/synthetic_{filename}'

    # Create database record
    dataset_obj = Dataset(
        name=name,
        description=f'Synthetic dataset with {num_rows} rows and {len(columns)} columns',
        file_path=file_path,
        file_type='csv',
        file_size=file_size,
        data_type='tabular',
        num_rows=num_rows,
        num_columns=len(columns),
        column_info={
            col_name: {'dtype': str(df[col_name].dtype), 'null_count': 0}
            for col_name in df.columns
        },
        user_id=user_id,
        profile_status='completed'
    )

    db.session.add(dataset_obj)
    db.session.commit()

    return jsonify({
        'message': f'Synthetic dataset "{name}" generated with {num_rows} rows',
        'dataset': dataset_obj.to_dict()
    }), 201
