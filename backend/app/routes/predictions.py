"""
Prediction Routes
"""
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.models.experiment import Experiment

predictions_bp = Blueprint('predictions', __name__)


@predictions_bp.route('/<int:model_id>', methods=['POST'])
@jwt_required()
def predict(model_id):
    """Make a single prediction"""
    from app.services.minio_service import get_minio_service
    import zipfile
    import io
    import json
    import joblib
    import tempfile
    import pandas as pd
    import os
    
    user_id = int(get_jwt_identity())
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    # Find the experiment/model
    experiment = Experiment.query.filter_by(id=model_id, user_id=user_id).first()
    if not experiment:
        return jsonify({'error': 'Model not found'}), 404
    
    if experiment.status != 'completed':
        return jsonify({'error': 'Model training not completed'}), 400
    
    input_data = data.get('input')
    if not input_data:
        return jsonify({'error': 'Input data is required'}), 400
    
    # Get model package path
    results = experiment.results or {}
    model_package_path = results.get('model_package_path')
    
    if not model_package_path:
        return jsonify({'error': 'Model package not available'}), 404
    
    try:
        # Download model package from MinIO
        minio_service = get_minio_service()
        # Handle legacy paths with incorrect 'models/' prefix
        corrected_path = model_package_path[7:] if model_package_path.startswith('models/') else model_package_path
        zip_content = minio_service.download_bytes('models', corrected_path)
        
        if not zip_content:
            return jsonify({'error': 'Failed to download model'}), 500
        
        # Extract and load model
        zip_buffer = io.BytesIO(zip_content)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Load model
            model_path = os.path.join(temp_dir, 'model.pkl')
            model = joblib.load(model_path)
            
            # Load preprocessor if exists
            preprocessor_path = os.path.join(temp_dir, 'preprocessor.pkl')
            preprocessor = None
            if os.path.exists(preprocessor_path):
                preprocessor = joblib.load(preprocessor_path)
            
            # Load target classes mapping (for classification label decoding)
            target_classes = None
            tc_path = os.path.join(temp_dir, 'target_classes.json')
            if os.path.exists(tc_path):
                with open(tc_path, 'r') as f:
                    target_classes = json.load(f)
            
            # Prepare input as DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Apply preprocessor if available
            if preprocessor is not None:
                input_processed = preprocessor.transform(input_df)
            else:
                input_processed = input_df
            
            # Make prediction
            prediction = model.predict(input_processed)[0]
            
            # Decode classification labels
            if target_classes is not None and hasattr(prediction, '__int__'):
                try:
                    idx = int(prediction)
                    decoded_prediction = target_classes[idx] if idx < len(target_classes) else prediction
                except (ValueError, IndexError):
                    decoded_prediction = prediction
            else:
                decoded_prediction = prediction
            
            # Round regression floats
            if isinstance(decoded_prediction, float):
                decoded_prediction = round(decoded_prediction, 4)
            
            # Convert to native Python type
            if hasattr(decoded_prediction, 'item'):
                decoded_prediction = decoded_prediction.item()
            
            # Get all class probabilities (for classification)
            probability = None
            class_probabilities = None
            if hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(input_processed)[0]
                    probability = float(max(proba))
                    
                    # Build class probability dict
                    if target_classes and len(target_classes) == len(proba):
                        class_probabilities = {
                            name: round(float(p), 4)
                            for name, p in zip(target_classes, proba)
                        }
                    else:
                        class_probabilities = {
                            f'class_{i}': round(float(p), 4)
                            for i, p in enumerate(proba)
                        }
                    # Sort by probability descending
                    class_probabilities = dict(
                        sorted(class_probabilities.items(), key=lambda x: x[1], reverse=True)
                    )
                except:
                    pass
            
            response = {
                'prediction': decoded_prediction,
                'probability': probability,
                'model_name': experiment.best_model_name,
                'input': input_data
            }
            if class_probabilities:
                response['class_probabilities'] = class_probabilities
            
            return jsonify(response), 200
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@predictions_bp.route('/<int:model_id>/batch', methods=['POST'])
@jwt_required()
def batch_predict(model_id):
    """Make batch predictions from uploaded file"""
    from app.services.minio_service import get_minio_service
    import zipfile
    import io
    import json
    import joblib
    import tempfile
    import pandas as pd
    import numpy as np
    
    user_id = int(get_jwt_identity())
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    # Find the experiment/model
    experiment = Experiment.query.filter_by(id=model_id, user_id=user_id).first()
    if not experiment:
        return jsonify({'error': 'Model not found'}), 404
    
    if experiment.status != 'completed':
        return jsonify({'error': 'Model training not completed'}), 400
    
    try:
        # Read uploaded CSV
        file = request.files['file']
        batch_df = pd.read_csv(file)
        
        if len(batch_df) == 0:
            return jsonify({'error': 'Uploaded file is empty'}), 400
        
        if len(batch_df) > 10000:
            return jsonify({'error': 'Maximum 10,000 rows allowed per batch'}), 400
        
        # Download and extract model package
        minio = get_minio_service()
        model_data = minio.download_model(experiment.model_path)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_buffer = io.BytesIO(model_data)
            with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            import os
            model = joblib.load(os.path.join(temp_dir, 'model.pkl'))
            
            preprocessor = None
            preprocessor_path = os.path.join(temp_dir, 'preprocessor.pkl')
            if os.path.exists(preprocessor_path):
                preprocessor = joblib.load(preprocessor_path)
            
            target_classes = None
            tc_path = os.path.join(temp_dir, 'target_classes.json')
            if os.path.exists(tc_path):
                with open(tc_path, 'r') as f:
                    target_classes = json.load(f)
            
            # Apply preprocessing
            if preprocessor is not None:
                batch_processed = preprocessor.transform(batch_df)
            else:
                batch_processed = batch_df
            
            # Predict
            raw_predictions = model.predict(batch_processed)
            
            # Decode predictions
            decoded = []
            for pred in raw_predictions:
                if target_classes is not None and hasattr(pred, '__int__'):
                    try:
                        idx = int(pred)
                        decoded.append(target_classes[idx] if idx < len(target_classes) else str(pred))
                    except (ValueError, IndexError):
                        decoded.append(str(pred))
                elif isinstance(pred, (float, np.floating)):
                    decoded.append(round(float(pred), 4))
                else:
                    if hasattr(pred, 'item'):
                        decoded.append(pred.item())
                    else:
                        decoded.append(pred)
            
            # Get confidence if classification
            confidences = None
            if hasattr(model, 'predict_proba'):
                try:
                    probas = model.predict_proba(batch_processed)
                    confidences = [round(float(max(p)), 4) for p in probas]
                except:
                    pass
            
            # Build results
            results = []
            for i, pred in enumerate(decoded):
                row = {
                    'row_index': i,
                    'prediction': pred,
                }
                if confidences:
                    row['confidence'] = confidences[i]
                results.append(row)
            
            return jsonify({
                'predictions': results,
                'total_rows': len(results),
                'model_name': experiment.best_model_name
            }), 200
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500


@predictions_bp.route('/<int:model_id>/explain', methods=['POST'])
@jwt_required()
def explain_prediction(model_id):
    """Get explanation for a prediction using SHAP"""
    from app.services.minio_service import get_minio_service
    import zipfile
    import io
    import json
    import joblib
    import tempfile
    import pandas as pd
    import numpy as np
    import os
    
    user_id = int(get_jwt_identity())
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    # Find the experiment/model
    experiment = Experiment.query.filter_by(id=model_id, user_id=user_id).first()
    if not experiment:
        return jsonify({'error': 'Model not found'}), 404
    
    input_data = data.get('input')
    if not input_data:
        return jsonify({'error': 'Input data is required'}), 400
    
    # Get model package path
    results = experiment.results or {}
    model_package_path = results.get('model_package_path')
    
    if not model_package_path:
        return jsonify({'error': 'Model package not available'}), 404
    
    try:
        # Download model package from MinIO
        minio_service = get_minio_service()
        # Handle legacy paths with incorrect 'models/' prefix
        corrected_path = model_package_path[7:] if model_package_path.startswith('models/') else model_package_path
        zip_content = minio_service.download_bytes('models', corrected_path)
        
        if not zip_content:
            return jsonify({'error': 'Failed to download model'}), 500
        
        zip_buffer = io.BytesIO(zip_content)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Load model
            model_path = os.path.join(temp_dir, 'model.pkl')
            model = joblib.load(model_path)
            
            # Load preprocessor if exists
            preprocessor_path = os.path.join(temp_dir, 'preprocessor.pkl')
            preprocessor = None
            if os.path.exists(preprocessor_path):
                preprocessor = joblib.load(preprocessor_path)
            
            # Load target classes mapping (for classification label decoding)
            target_classes = None
            tc_path = os.path.join(temp_dir, 'target_classes.json')
            if os.path.exists(tc_path):
                with open(tc_path, 'r') as f:
                    target_classes = json.load(f)
            
            # Prepare input as DataFrame
            input_df = pd.DataFrame([input_data])
            feature_names = list(input_data.keys())
            
            # Apply preprocessor if available
            if preprocessor is not None:
                input_processed = preprocessor.transform(input_df)
            else:
                input_processed = input_df.values if hasattr(input_df, 'values') else input_df
            
            # Make prediction
            prediction = model.predict(input_processed)[0]
            
            # Decode classification labels
            if target_classes is not None and hasattr(prediction, '__int__'):
                try:
                    idx = int(prediction)
                    prediction = target_classes[idx] if idx < len(target_classes) else prediction
                except (ValueError, IndexError):
                    pass
            
            # Round regression floats
            if isinstance(prediction, float):
                prediction = round(prediction, 4)
            
            if hasattr(prediction, 'item'):
                prediction = prediction.item()
            
            # Try to get feature importance
            feature_importance = {}
            
            # Try SHAP first
            try:
                import shap
                
                # Use TreeExplainer for tree-based models
                if hasattr(model, 'feature_importances_'):
                    # Use model's feature importance
                    importances = model.feature_importances_
                    for i, name in enumerate(feature_names):
                        if i < len(importances):
                            feature_importance[name] = float(importances[i])
                else:
                    # Try SHAP
                    explainer = shap.Explainer(model)
                    shap_values = explainer(input_processed)
                    
                    for i, name in enumerate(feature_names):
                        if i < len(shap_values.values[0]):
                            feature_importance[name] = float(abs(shap_values.values[0][i]))
                            
            except Exception as shap_error:
                print(f"SHAP failed: {shap_error}")
                # Fallback to model's feature importance if available
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    for i, name in enumerate(feature_names):
                        if i < len(importances):
                            feature_importance[name] = float(importances[i])
            
            # Sort by importance
            feature_importance = dict(
                sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
            )
            
            return jsonify({
                'prediction': prediction,
                'explanation': {
                    'feature_importance': feature_importance,
                    'summary': f'Top contributing feature: {list(feature_importance.keys())[0] if feature_importance else "Unknown"}'
                }
            }), 200
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Explanation failed: {str(e)}'}), 500
