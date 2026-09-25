
"""
InferX-ML Streamlit Application
Dynamic model loading and prediction UI
Version: 1.1 (Dynamic Execution)
"""
import sys
sys.path.append('/app')

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import io
import joblib
import tempfile
import requests
from typing import Dict, Any, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ─── CombinedPreprocessor must be defined here for joblib unpickling ───
# When the model was trained in the backend container, the preprocessor was
# pickled as app.routes.training.CombinedPreprocessor. The Streamlit container
# only mounts streamlit_app/ at /app, so the original module is unavailable.
# We define the class here and register it in the expected module path.
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
        for col in self.categorical_columns:
            self.label_encoders[col] = LabelEncoder()
            X_processed[col] = self.label_encoders[col].fit_transform(X_processed[col].astype(str))
        X_processed = X_processed.fillna(X_processed.median())
        return self.scaler.fit_transform(X_processed)
    
    def transform(self, X):
        X_processed = X.copy()
        for col in self.categorical_columns:
            if col in X_processed.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                X_processed[col] = X_processed[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0
                )
        for col in self.feature_columns:
            if col not in X_processed.columns:
                X_processed[col] = 0
        X_processed = X_processed[self.feature_columns].fillna(0)
        return self.scaler.transform(X_processed)


# Register in the module path that pickle expects
import types
_fake_app = types.ModuleType('app')
_fake_routes = types.ModuleType('app.routes')
_fake_training = types.ModuleType('app.routes.training')
_fake_training.CombinedPreprocessor = CombinedPreprocessor
_fake_app.routes = _fake_routes
_fake_routes.training = _fake_training
import sys as _sys
_sys.modules.setdefault('app', _fake_app)
_sys.modules.setdefault('app.routes', _fake_routes)
_sys.modules.setdefault('app.routes.training', _fake_training)

# Page configuration
st.set_page_config(
    page_title="InferX-ML Predictions",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# API URL - internal Docker network
API_URL = os.environ.get('API_URL', 'http://backend:5000/api')
INTERNAL_SECRET = os.environ.get('INTERNAL_API_SECRET', 'inferx-internal-2024')

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #f97316;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-result {
        background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        margin: 1rem 0;
    }
    .confidence-badge {
        background: rgba(255,255,255,0.2);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin-top: 0.5rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        font-weight: bold;
        background-color: #10b981;
        color: white;
    }
    .stButton>button:hover {
        background-color: #059669;
    }
</style>
""", unsafe_allow_html=True)


def get_model_id_from_url():
    """Get model ID from URL query parameters"""
    # Use experimental API for compatibility with older Streamlit versions
    try:
        params = st.query_params
        return params.get('model', None)
    except AttributeError:
        # Fallback for older Streamlit versions
        params = st.experimental_get_query_params()
        model_list = params.get('model', [])
        return model_list[0] if model_list else None



def load_model_package(model_id: int):
    """
    Download and extract model package from MinIO.
    Returns: (temp_dir_object, temp_dir_path, error_message)
    Caller is responsible for cleaning up temp_dir_object.
    """
    try:
        # Use internal endpoint (no JWT required)
        headers = {'X-Internal-Secret': INTERNAL_SECRET}
        response = requests.get(
            f"{API_URL}/models/internal/{model_id}/download", 
            headers=headers,
            timeout=30
        )
        
        if response.status_code != 200:
            return None, None, f"Failed to load model: {response.status_code}"
        
        # Extract ZIP in memory
        import zipfile
        zip_buffer = io.BytesIO(response.content)
        
        temp_dir_obj = tempfile.TemporaryDirectory()
        temp_dir_path = temp_dir_obj.name
        
        with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
            zip_ref.extractall(temp_dir_path)
            
        return temp_dir_obj, temp_dir_path, None
            
    except Exception as e:
        return None, None, str(e)


@st.cache_resource
def load_cached_model(_model_id_str: str):
    """
    Download, extract, and load model components — cached by model_id.
    Uses @st.cache_resource so the model stays in memory across re-runs.
    """
    model_id = int(_model_id_str)
    temp_dir_obj, temp_dir, error = load_model_package(model_id)
    if error:
        return None, None, None, None, None, None, error
    
    model = None
    preprocessor = None
    schema = None
    model_info = {}
    target_classes = None
    
    try:
        # Load model
        model_path = os.path.join(temp_dir, 'model.pkl')
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        
        # Load preprocessor
        preprocessor_path = os.path.join(temp_dir, 'preprocessor.pkl')
        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)
        
        # Load UI schema
        schema_path = os.path.join(temp_dir, 'ui_schema.json')
        if os.path.exists(schema_path):
            with open(schema_path, 'r') as f:
                schema = json.load(f)
        
        # Load model info
        info_path = os.path.join(temp_dir, 'model_info.json')
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                model_info = json.load(f)
        
        # Load target classes
        tc_path = os.path.join(temp_dir, 'target_classes.json')
        if os.path.exists(tc_path):
            with open(tc_path, 'r') as f:
                target_classes = json.load(f)
        
        return model, preprocessor, schema, model_info, target_classes, temp_dir_obj, None
    except Exception as e:
        return model, preprocessor, schema, model_info, target_classes, temp_dir_obj, str(e)


def load_legacy_components(temp_dir):
    """Load components for legacy/fallback UI"""
    model = None
    preprocessor = None
    schema = None
    model_info = {}
    target_classes = None
    
    try:
        # Debug: List files in temp_dir
        print(f"--- DEBUG: Loading components from {temp_dir} ---", flush=True)
        if os.path.exists(temp_dir):
            files = os.listdir(temp_dir)
            print(f"Files in temp_dir: {files}", flush=True)
        else:
            print(f"ERROR: temp_dir does not exist!", flush=True)
            return None, None, None, None, None
        
        # Load model - with specific error handling
        model_path = os.path.join(temp_dir, 'model.pkl')
        print(f"Looking for model at: {model_path}, exists: {os.path.exists(model_path)}", flush=True)
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                print(f"Model loaded successfully: {type(model)}", flush=True)
            except Exception as e:
                print(f"Failed to load model: {e}", flush=True)
                # Try loading with pickle directly as fallback
                try:
                    import pickle
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    print(f"Model loaded via pickle fallback: {type(model)}", flush=True)
                except Exception as pe:
                    print(f"Model pickle fallback also failed: {pe}", flush=True)
        
        # Load preprocessor
        preprocessor_path = os.path.join(temp_dir, 'preprocessor.pkl')
        print(f"Looking for preprocessor at: {preprocessor_path}, exists: {os.path.exists(preprocessor_path)}", flush=True)
        if os.path.exists(preprocessor_path):
            try:
                preprocessor = joblib.load(preprocessor_path)
                print(f"Preprocessor loaded successfully: {type(preprocessor)}", flush=True)
            except Exception as e:
                print(f"Failed to load preprocessor: {e}", flush=True)
        
        # Load UI schema
        schema_path = os.path.join(temp_dir, 'ui_schema.json')
        print(f"Looking for schema at: {schema_path}, exists: {os.path.exists(schema_path)}", flush=True)
        if os.path.exists(schema_path):
            with open(schema_path, 'r') as f:
                schema = json.load(f)
        
        # Load model info
        info_path = os.path.join(temp_dir, 'model_info.json')
        print(f"Looking for model_info at: {info_path}, exists: {os.path.exists(info_path)}", flush=True)
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                model_info = json.load(f)
        
        # Load target classes (for classification label decoding)
        tc_path = os.path.join(temp_dir, 'target_classes.json')
        if os.path.exists(tc_path):
            with open(tc_path, 'r') as f:
                target_classes = json.load(f)
            print(f"Target classes loaded: {target_classes}", flush=True)
        
        print(f"Loaded: model={model is not None}, preprocessor={preprocessor is not None}, schema={schema is not None}", flush=True)
        return model, preprocessor, schema, model_info, target_classes
    except Exception as e:
        print(f"ERROR loading components: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return model, preprocessor, schema, model_info, target_classes


def fetch_models_list():
    """Fetch available models from internal API"""
    try:
        headers = {'X-Internal-Secret': INTERNAL_SECRET}
        response = requests.get(f"{API_URL}/models/internal/list", headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get('models', [])
    except:
        pass
    return []


def generate_form_from_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Generate Streamlit form elements from UI schema"""
    form_values = {}
    
    if not schema or 'fields' not in schema:
        return form_values
    
    fields = schema.get('fields', [])
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    for i, field in enumerate(fields):
        name = field['name']
        label = field.get('label', name.replace('_', ' ').title())
        input_type = field.get('input_type', 'text')
        
        # Alternate between columns
        with col1 if i % 2 == 0 else col2:
            if input_type == 'number':
                min_val = field.get('min', 0.0)
                max_val = field.get('max', 1000.0)
                default_val = field.get('default', min_val)
                
                # Ensure values are valid floats
                try:
                    min_val = float(min_val) if min_val is not None else 0.0
                    max_val = float(max_val) if max_val is not None else 1000.0
                    default_val = float(default_val) if default_val is not None else min_val
                    default_val = max(min_val, min(max_val, default_val))
                except:
                    min_val, max_val, default_val = 0.0, 1000.0, 0.0
                
                form_values[name] = st.number_input(
                    label,
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    key=f"field_{name}"
                )
            
            elif input_type == 'dropdown':
                options = field.get('options', [])
                if options:
                    form_values[name] = st.selectbox(label, options, key=f"field_{name}")
                else:
                    form_values[name] = st.text_input(label, key=f"field_{name}")
            
            elif input_type == 'slider':
                min_val = field.get('min', 0)
                max_val = field.get('max', 100)
                default_val = field.get('default', 50)
                form_values[name] = st.slider(
                    label,
                    min_value=int(min_val),
                    max_value=int(max_val),
                    value=int(default_val),
                    key=f"field_{name}"
                )
            
            elif input_type == 'checkbox':
                form_values[name] = st.checkbox(label, value=field.get('default', False), key=f"field_{name}")
            
            else:  # text input
                form_values[name] = st.text_input(label, value=str(field.get('default', '')), key=f"field_{name}")
    
    return form_values


def run_legacy_ui(temp_dir):
    """Run the legacy/generic UI for older models"""
    model, preprocessor, schema, model_info, target_classes = load_legacy_components(temp_dir)
    
    if model is None:
        st.error("Failed to load model components.")
        return
    
    # Determine problem type from model_info
    problem_type = model_info.get('problem_type', 'classification')
    target_col = model_info.get('target_column', 'target')
    
    # Model info in sidebar
    with st.sidebar:
        st.header("📊 Model Info")
        if model_info:
            st.metric("Algorithm", model_info.get('best_algorithm', model_info.get('model_name', 'Unknown')))
            st.metric("Score", f"{model_info.get('best_score', model_info.get('accuracy', 0)):.2%}")
            st.metric("Target", target_col)
            
            if 'num_features' in model_info:
                st.metric("Features", model_info.get('num_features', 0))
            elif 'num_classes' in model_info:
                 st.metric("Classes", model_info.get('num_classes', 0))
            
            if target_classes:
                st.metric("Classes", len(target_classes))
        
        st.divider()
        if st.button("🔄 Choose Different Model"):
            try:
                st.query_params.clear()
            except AttributeError:
                st.experimental_set_query_params()
            st.rerun()

    # CHECK FOR IMAGE CLASSIFIER
    # We can check for specific methods or metadata
    is_image_model = False
    if hasattr(model, 'predict_with_label') or \
       (model_info and model_info.get('problem_type') == 'image_classification'):
        is_image_model = True

    if is_image_model:
        st.subheader("🖼️ Image Classification")
        st.info(f"Model trained to recognize: {', '.join(model_info.get('class_names', []))}")
        
        uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Display image
            from PIL import Image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            if st.button("🚀 Classify Image", use_container_width=True):
                with st.spinner("Analyzing..."):
                    try:
                        # Save to temp file for preprocessor
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                            image.convert('RGB').save(tmp.name)
                            tmp_path = tmp.name
                        
                        # Preprocess
                        # If we have the original ImagePreprocessor class instance
                        if hasattr(model, 'input_size_'):
                            # It's likely our ImageClassifier class
                            import numpy as np
                            # Re-implement simple preprocessing if preprocessor missing
                            img_rgb = image.convert('RGB')
                            img_resized = img_rgb.resize(model.input_size_)
                            img_array = np.array(img_resized) / 255.0
                            img_array = (img_array - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                            img_batch = np.expand_dims(img_array, axis=0)
                            
                            # Predict
                            predictions = model.predict_with_label(img_batch)
                            result = predictions[0]
                            
                            # Display result
                            st.markdown(f'''
                                <div class="prediction-result">
                                    <div>Predicted: <strong>{result['class']}</strong></div>
                                    <div class="confidence-badge">Confidence: {result['confidence']:.1%}</div>
                                </div>
                            ''', unsafe_allow_html=True)
                            
                            # Show all probabilities
                            st.bar_chart(result['all_probabilities'])
                            
                        else:
                            st.error("Model format not recognized for image classification")
                            
                        # Cleanup
                        os.unlink(tmp_path)
                        
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
                        import traceback
                        st.text(traceback.format_exc())

    else:
        # === TABULAR PREDICTION UI ===
        if not schema:
            st.warning("No schema available. Cannot generate form.")
            return
        
        # Create tabs for Single and Batch prediction
        tab_single, tab_batch = st.tabs(["🎯 Single Prediction", "📁 Batch Prediction"])
        
        with tab_single:
            st.subheader("📝 Enter Feature Values")
            form_values = generate_form_from_schema(schema)
            
            st.divider()
            
            # Predict button
            if st.button("✨ Make Prediction", use_container_width=True, key="single_predict"):
                try:
                    # --- Input validation: warn if outside training range ---
                    out_of_range = []
                    for field in schema.get('fields', []):
                        if field.get('input_type') == 'number':
                            name = field['name']
                            val = form_values.get(name)
                            fmin = field.get('min')
                            fmax = field.get('max')
                            if val is not None and fmin is not None and fmax is not None:
                                if val < fmin or val > fmax:
                                    out_of_range.append(f"**{field.get('label', name)}**: {val} (training range: {fmin:.2f} – {fmax:.2f})")
                    
                    if out_of_range:
                        st.warning("⚠️ Some inputs are outside the training data range. Predictions may be unreliable:\n\n" + "\n\n".join(out_of_range))
                    
                    # Prepare input data
                    input_df = pd.DataFrame([form_values])
                    
                    # Handle categorical encoding for older models
                    def encode_categorical(df):
                        df_encoded = df.copy()
                        for col in df_encoded.columns:
                            if df_encoded[col].dtype == 'object' or isinstance(df_encoded[col].iloc[0], str):
                                try:
                                    df_encoded[col] = pd.Categorical(df_encoded[col]).codes
                                except:
                                    df_encoded[col] = 0
                        return df_encoded
                    
                    # Apply preprocessor if available
                    if preprocessor is not None:
                        try:
                            input_processed = preprocessor.transform(input_df)
                        except (ValueError, TypeError):
                            input_encoded = encode_categorical(input_df)
                            try:
                                input_processed = preprocessor.transform(input_encoded)
                            except:
                                input_processed = input_encoded.values
                    else:
                        input_processed = encode_categorical(input_df).values
                    
                    # Make prediction
                    raw_prediction = model.predict(input_processed)[0]
                    
                    # ═══ Format and display result ═══
                    st.divider()
                    
                    if problem_type == 'classification' or (target_classes is not None):
                        # Decode classification label
                        if target_classes is not None and hasattr(raw_prediction, '__int__'):
                            try:
                                idx = int(raw_prediction)
                                display_prediction = target_classes[idx] if idx < len(target_classes) else str(raw_prediction)
                            except (ValueError, IndexError):
                                display_prediction = str(raw_prediction)
                        else:
                            display_prediction = str(raw_prediction)
                        
                        # Get all class probabilities
                        probability = None
                        class_probs = None
                        if hasattr(model, 'predict_proba'):
                            try:
                                proba = model.predict_proba(input_processed)[0]
                                probability = float(max(proba))
                                
                                if target_classes and len(target_classes) == len(proba):
                                    class_probs = {name: float(p) for name, p in zip(target_classes, proba)}
                                else:
                                    class_probs = {f'Class {i}': float(p) for i, p in enumerate(proba)}
                            except:
                                pass
                        
                        # Display prediction result
                        st.markdown(f'''
                            <div class="prediction-result">
                                <div>Predicted {target_col}: <strong>{display_prediction}</strong></div>
                                {f'<div class="confidence-badge">Confidence: {probability:.1%}</div>' if probability else ''}
                            </div>
                        ''', unsafe_allow_html=True)
                        
                        # Show class probability chart
                        if class_probs:
                            st.subheader("📊 Class Probabilities")
                            prob_df = pd.DataFrame({
                                'Class': list(class_probs.keys()),
                                'Probability (%)': [round(p * 100, 2) for p in class_probs.values()]
                            }).sort_values('Probability (%)', ascending=False)
                            
                            st.bar_chart(prob_df.set_index('Class'))
                            
                            with st.expander("📋 Detailed Probabilities", expanded=False):
                                st.dataframe(prob_df.reset_index(drop=True), use_container_width=True)
                    
                    else:
                        # Regression — round to 2 decimal places
                        if isinstance(raw_prediction, (float, np.floating)):
                            display_prediction = round(float(raw_prediction), 2)
                        else:
                            display_prediction = raw_prediction
                        
                        st.markdown(f'''
                            <div class="prediction-result">
                                <div>Predicted {target_col}: <strong>{display_prediction}</strong></div>
                            </div>
                        ''', unsafe_allow_html=True)
                    
                    # Show input summary
                    with st.expander("📋 Input Summary", expanded=True):
                        summary_df = pd.DataFrame([form_values]).T
                        summary_df.columns = ['Value']
                        summary_df.index.name = 'Feature'
                        st.dataframe(summary_df, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
        
        with tab_batch:
            st.subheader("📁 Upload CSV for Batch Predictions")
            st.info("Upload a CSV file with the same feature columns used during training. The model will predict each row.")
            
            uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], key="batch_upload")
            
            if uploaded_file is not None:
                try:
                    batch_df = pd.read_csv(uploaded_file)
                    st.write(f"📊 **{len(batch_df)} rows** loaded with columns: {', '.join(batch_df.columns.tolist())}")
                    
                    with st.expander("🔍 Preview Input Data", expanded=False):
                        st.dataframe(batch_df.head(10), use_container_width=True)
                    
                    if st.button("✨ Run Batch Prediction", use_container_width=True, key="batch_predict"):
                        with st.spinner(f"Predicting {len(batch_df)} rows..."):
                            try:
                                # Handle categorical encoding
                                def encode_categorical_batch(df):
                                    df_encoded = df.copy()
                                    for col in df_encoded.columns:
                                        if df_encoded[col].dtype == 'object' or (len(df_encoded) > 0 and isinstance(df_encoded[col].iloc[0], str)):
                                            try:
                                                df_encoded[col] = pd.Categorical(df_encoded[col]).codes
                                            except:
                                                df_encoded[col] = 0
                                    return df_encoded
                                
                                # Apply preprocessing
                                if preprocessor is not None:
                                    try:
                                        batch_processed = preprocessor.transform(batch_df)
                                    except:
                                        batch_processed = encode_categorical_batch(batch_df).values
                                else:
                                    batch_processed = encode_categorical_batch(batch_df).values
                                
                                # Predict
                                raw_predictions = model.predict(batch_processed)
                                
                                # Decode predictions
                                decoded_predictions = []
                                for pred in raw_predictions:
                                    if target_classes is not None and hasattr(pred, '__int__'):
                                        try:
                                            idx = int(pred)
                                            decoded_predictions.append(target_classes[idx] if idx < len(target_classes) else str(pred))
                                        except (ValueError, IndexError):
                                            decoded_predictions.append(str(pred))
                                    elif isinstance(pred, (float, np.floating)):
                                        decoded_predictions.append(round(float(pred), 2))
                                    else:
                                        decoded_predictions.append(pred)
                                
                                # Build results DataFrame
                                results_df = batch_df.copy()
                                results_df[f'Predicted_{target_col}'] = decoded_predictions
                                
                                # Add confidence if classification
                                if hasattr(model, 'predict_proba'):
                                    try:
                                        probas = model.predict_proba(batch_processed)
                                        results_df['Confidence (%)'] = [round(float(max(p)) * 100, 1) for p in probas]
                                    except:
                                        pass
                                
                                st.success(f"✅ Predictions complete for {len(results_df)} rows!")
                                st.dataframe(results_df, use_container_width=True)
                                
                                # Download button
                                csv_output = results_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Results as CSV",
                                    data=csv_output,
                                    file_name="batch_predictions.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                                
                            except Exception as e:
                                st.error(f"Batch prediction failed: {str(e)}")
                
                except Exception as e:
                    st.error(f"Failed to read CSV: {str(e)}")


def run_cached_ui(model_id):
    """Load model from cache and run the prediction UI"""
    model, preprocessor, schema, model_info, target_classes, _temp_dir_obj, error = load_cached_model(str(model_id))
    
    if error:
        st.error(f"Failed to load model: {error}")
        return
    if model is None:
        st.error("Failed to load model components.")
        return
    
    # Determine problem type
    problem_type = model_info.get('problem_type', 'classification') if model_info else 'classification'
    target_col = model_info.get('target_column', 'target') if model_info else 'target'
    
    # Model info in sidebar
    with st.sidebar:
        st.header("📊 Model Info")
        if model_info:
            st.metric("Algorithm", model_info.get('best_algorithm', model_info.get('model_name', 'Unknown')))
            st.metric("Score", f"{model_info.get('best_score', model_info.get('accuracy', 0)):.2%}")
            st.metric("Target", target_col)
            if 'num_features' in model_info:
                st.metric("Features", model_info.get('num_features', 0))
            if target_classes:
                st.metric("Classes", len(target_classes))
        
        st.divider()
        if st.button("🔄 Choose Different Model"):
            try:
                st.query_params.clear()
            except AttributeError:
                st.experimental_set_query_params()
            # Clear cache for this model so it reloads fresh next time
            load_cached_model.clear()
            st.rerun()
    
    # Image classifier check
    is_image_model = False
    if hasattr(model, 'predict_with_label') or \
       (model_info and model_info.get('problem_type') == 'image_classification'):
        is_image_model = True
    
    if is_image_model:
        # Delegate to legacy UI for image models
        run_legacy_ui_image(model, model_info)
    else:
        # Tabular prediction with tabs
        run_tabular_ui(model, preprocessor, schema, model_info, target_classes, problem_type, target_col)


def run_legacy_ui_image(model, model_info):
    """Handle image classification models"""
    st.subheader("🖼️ Image Classification")
    st.info(f"Model trained to recognize: {', '.join(model_info.get('class_names', []))}")
    
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        from PIL import Image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if st.button("🚀 Classify Image", use_container_width=True):
            with st.spinner("Analyzing..."):
                try:
                    if hasattr(model, 'input_size_'):
                        img_rgb = image.convert('RGB')
                        img_resized = img_rgb.resize(model.input_size_)
                        img_array = np.array(img_resized) / 255.0
                        img_array = (img_array - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                        img_batch = np.expand_dims(img_array, axis=0)
                        predictions = model.predict_with_label(img_batch)
                        result = predictions[0]
                        st.markdown(f'''
                            <div class="prediction-result">
                                <div>Predicted: <strong>{result['class']}</strong></div>
                                <div class="confidence-badge">Confidence: {result['confidence']:.1%}</div>
                            </div>
                        ''', unsafe_allow_html=True)
                        st.bar_chart(result['all_probabilities'])
                    else:
                        st.error("Model format not recognized for image classification")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")


def run_tabular_ui(model, preprocessor, schema, model_info, target_classes, problem_type, target_col):
    """Tabular prediction UI with single and batch tabs"""
    if not schema:
        st.warning("No schema available. Cannot generate form.")
        return
    
    tab_single, tab_batch = st.tabs(["🎯 Single Prediction", "📁 Batch Prediction"])
    
    with tab_single:
        st.subheader("📝 Enter Feature Values")
        form_values = generate_form_from_schema(schema)
        st.divider()
        
        if st.button("✨ Make Prediction", use_container_width=True, key="single_predict"):
            try:
                # Input validation
                out_of_range = []
                for field in schema.get('fields', []):
                    if field.get('input_type') == 'number':
                        name = field['name']
                        val = form_values.get(name)
                        fmin = field.get('min')
                        fmax = field.get('max')
                        if val is not None and fmin is not None and fmax is not None:
                            if val < fmin or val > fmax:
                                out_of_range.append(f"**{field.get('label', name)}**: {val} (training range: {fmin:.2f} – {fmax:.2f})")
                
                if out_of_range:
                    st.warning("⚠️ Some inputs are outside the training data range. Predictions may be unreliable:\n\n" + "\n\n".join(out_of_range))
                
                input_df = pd.DataFrame([form_values])
                input_processed = _preprocess_input(input_df, preprocessor)
                raw_prediction = model.predict(input_processed)[0]
                
                st.divider()
                _display_single_result(model, input_processed, raw_prediction, target_classes, problem_type, target_col)
                
                with st.expander("📋 Input Summary", expanded=True):
                    summary_df = pd.DataFrame([form_values]).T
                    summary_df.columns = ['Value']
                    summary_df.index.name = 'Feature'
                    st.dataframe(summary_df, use_container_width=True)
            
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
    
    with tab_batch:
        st.subheader("📁 Upload CSV for Batch Predictions")
        st.info("Upload a CSV file with the same feature columns used during training. The model will predict each row.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], key="batch_upload")
        
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.write(f"📊 **{len(batch_df)} rows** loaded with columns: {', '.join(batch_df.columns.tolist())}")
                
                with st.expander("🔍 Preview Input Data", expanded=False):
                    st.dataframe(batch_df.head(10), use_container_width=True)
                
                if st.button("✨ Run Batch Prediction", use_container_width=True, key="batch_predict"):
                    with st.spinner(f"Predicting {len(batch_df)} rows..."):
                        try:
                            batch_processed = _preprocess_input(batch_df, preprocessor)
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
                                    decoded.append(round(float(pred), 2))
                                else:
                                    decoded.append(pred)
                            
                            results_df = batch_df.copy()
                            results_df[f'Predicted_{target_col}'] = decoded
                            
                            if hasattr(model, 'predict_proba'):
                                try:
                                    probas = model.predict_proba(batch_processed)
                                    results_df['Confidence (%)'] = [round(float(max(p)) * 100, 1) for p in probas]
                                except:
                                    pass
                            
                            st.success(f"✅ Predictions complete for {len(results_df)} rows!")
                            st.dataframe(results_df, use_container_width=True)
                            
                            csv_output = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv_output,
                                file_name="batch_predictions.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        except Exception as e:
                            st.error(f"Batch prediction failed: {str(e)}")
            except Exception as e:
                st.error(f"Failed to read CSV: {str(e)}")


def _preprocess_input(input_df, preprocessor):
    """Apply preprocessor or fallback encoding"""
    def encode_categorical(df):
        df_encoded = df.copy()
        for col in df_encoded.columns:
            if df_encoded[col].dtype == 'object' or (len(df_encoded) > 0 and isinstance(df_encoded[col].iloc[0], str)):
                try:
                    df_encoded[col] = pd.Categorical(df_encoded[col]).codes
                except:
                    df_encoded[col] = 0
        return df_encoded
    
    if preprocessor is not None:
        try:
            return preprocessor.transform(input_df)
        except (ValueError, TypeError):
            encoded = encode_categorical(input_df)
            try:
                return preprocessor.transform(encoded)
            except:
                return encoded.values
    return encode_categorical(input_df).values


def _display_single_result(model, input_processed, raw_prediction, target_classes, problem_type, target_col):
    """Display formatted prediction result"""
    if problem_type == 'classification' or (target_classes is not None):
        if target_classes is not None and hasattr(raw_prediction, '__int__'):
            try:
                idx = int(raw_prediction)
                display_prediction = target_classes[idx] if idx < len(target_classes) else str(raw_prediction)
            except (ValueError, IndexError):
                display_prediction = str(raw_prediction)
        else:
            display_prediction = str(raw_prediction)
        
        probability = None
        class_probs = None
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(input_processed)[0]
                probability = float(max(proba))
                if target_classes and len(target_classes) == len(proba):
                    class_probs = {name: float(p) for name, p in zip(target_classes, proba)}
                else:
                    class_probs = {f'Class {i}': float(p) for i, p in enumerate(proba)}
            except:
                pass
        
        st.markdown(f'''
            <div class="prediction-result">
                <div>Predicted {target_col}: <strong>{display_prediction}</strong></div>
                {f'<div class="confidence-badge">Confidence: {probability:.1%}</div>' if probability else ''}
            </div>
        ''', unsafe_allow_html=True)
        
        if class_probs:
            st.subheader("📊 Class Probabilities")
            prob_df = pd.DataFrame({
                'Class': list(class_probs.keys()),
                'Probability (%)': [round(p * 100, 2) for p in class_probs.values()]
            }).sort_values('Probability (%)', ascending=False)
            st.bar_chart(prob_df.set_index('Class'))
            with st.expander("📋 Detailed Probabilities", expanded=False):
                st.dataframe(prob_df.reset_index(drop=True), use_container_width=True)
    else:
        if isinstance(raw_prediction, (float, np.floating)):
            display_prediction = round(float(raw_prediction), 2)
        else:
            display_prediction = raw_prediction
        st.markdown(f'''
            <div class="prediction-result">
                <div>Predicted {target_col}: <strong>{display_prediction}</strong></div>
            </div>
        ''', unsafe_allow_html=True)


def main():
    """Main application entry point"""
    # Get model ID from URL or show selector
    model_id = get_model_id_from_url()
    
    if not model_id:
        st.markdown('<h1 class="main-header">🤖 Make Predictions</h1>', unsafe_allow_html=True)
        st.info("Select a model to make predictions")
        models = fetch_models_list()
        
        if models:
            model_names = {str(m['id']): m['name'] for m in models if m.get('has_package')}
            if model_names:
                selected = st.selectbox(
                    "Choose Model",
                    options=list(model_names.keys()),
                    format_func=lambda x: model_names[x]
                )
                if st.button("Load Model"):
                    try:
                        st.query_params['model'] = selected
                    except AttributeError:
                        st.experimental_set_query_params(model=selected)
                    st.rerun()
            else:
                st.warning("No models with prediction packages available. Train a new model first.")
        else:
            st.warning("No models available. Train your first model to get started!")
        return
    
    # Load and run with caching
    st.markdown('<h1 class="main-header">🤖 Make Predictions</h1>', unsafe_allow_html=True)
    run_cached_ui(model_id)


if __name__ == "__main__":
    main()
