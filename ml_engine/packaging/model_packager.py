"""
Model Packaging Module
Creates self-contained model packages with preprocessing, schema, and metadata
"""
import json
import os
import shutil
import tempfile
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib


class ModelPackager:
    """Package trained models with all dependencies for deployment"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def package(
        self,
        model: Any,
        preprocessor: Any,
        feature_schema: Dict[str, Any],
        metadata: Dict[str, Any],
        model_format: str = 'pkl',
        graphs_dir: Optional[str] = None
    ) -> str:
        """
        Create a complete model package
        
        Args:
            model: Trained model object
            preprocessor: Fitted preprocessing pipeline
            feature_schema: Schema describing input features
            metadata: Model metadata (name, version, metrics, etc.)
            model_format: 'pkl' for sklearn, 'h5' for keras
            
        Returns:
            Path to the package directory
        """
        # Create package directory
        package_name = f"{metadata.get('name', 'model')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        package_dir = os.path.join(self.output_dir, package_name)
        os.makedirs(package_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(package_dir, f'model.{model_format}')
        if model_format == 'pkl':
            joblib.dump(model, model_path)
        elif model_format == 'h5':
            model.save(model_path)
        
        # Save preprocessor
        if preprocessor is not None:
            preprocessor_path = os.path.join(package_dir, 'preprocessor.pkl')
            joblib.dump(preprocessor, preprocessor_path)
        
        # Save feature schema
        feature_schema_path = os.path.join(package_dir, 'feature_schema.json')
        with open(feature_schema_path, 'w') as f:
            json.dump(feature_schema, f, indent=2)
        
        # Generate and save UI schema
        ui_schema = self._generate_ui_schema(feature_schema, metadata)
        ui_schema_path = os.path.join(package_dir, 'ui_schema.json')
        with open(ui_schema_path, 'w') as f:
            json.dump(ui_schema, f, indent=2)
        
        # Save metadata
        metadata['packaged_at'] = datetime.now().isoformat()
        metadata['package_version'] = '1.0.0'
        metadata_path = os.path.join(package_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save target class mapping (for decoding classification predictions)
        target_classes = metadata.get('target_classes')
        if target_classes:
            tc_path = os.path.join(package_dir, 'target_classes.json')
            with open(tc_path, 'w') as f:
                json.dump(target_classes, f, indent=2)
        
        # Generate model_info.json (headline/summary of trained model)
        model_info = self._generate_model_info(metadata, feature_schema)
        model_info_path = os.path.join(package_dir, 'model_info.json')
        with open(model_info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        # Generate loader.py (script to load and use the model)
        loader_code = self._generate_loader_script(model_format, metadata)
        loader_path = os.path.join(package_dir, 'loader.py')
        with open(loader_path, 'w', encoding='utf-8') as f:
            f.write(loader_code)
        
        # Generate Streamlit app (streamlit_app.py)
        streamlit_code = self._generate_streamlit_app(metadata, feature_schema)
        streamlit_path = os.path.join(package_dir, 'streamlit_app.py')
        with open(streamlit_path, 'w', encoding='utf-8') as f:
            f.write(streamlit_code)
        
        # Generate requirements.txt (includes streamlit)
        # Copy graphs folder if provided
        if graphs_dir and os.path.exists(graphs_dir):
            target_graphs_dir = os.path.join(package_dir, 'graphs')
            if not os.path.exists(target_graphs_dir):
                shutil.copytree(graphs_dir, target_graphs_dir)
        
        return package_dir
    
    def _generate_model_info(self, metadata: Dict[str, Any], feature_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generate model_info.json with headline/summary information"""
        return {
            'model_name': metadata.get('name', 'Unnamed Model'),
            'description': f"Trained {metadata.get('problem_type', 'ML')} model for predicting {metadata.get('target_column', 'target')}",
            'problem_type': metadata.get('problem_type'),
            'target_column': metadata.get('target_column'),
            'best_algorithm': metadata.get('best_model'),
            'best_score': metadata.get('best_score'),
            'training_date': metadata.get('packaged_at'),
            'features': list(feature_schema.get('features', {}).keys()),
            'num_features': len(feature_schema.get('features', {})),
            'usage': 'Load this model using loader.py or directly with joblib.load("model.pkl")'
        }
    
    def _generate_loader_script(self, model_format: str, metadata: Dict[str, Any]) -> str:
        """Generate loader.py script for loading and using the model"""
        
        model_name = metadata.get('name', 'model')
        target_col = metadata.get('target_column', 'target')
        problem_type = metadata.get('problem_type', 'classification')
        
        loader_code = f'''"""
Model Loader - {model_name}
Auto-generated script to load and use the trained model

Usage:
    from loader import load_model, predict
    
    model, preprocessor = load_model()
    result = predict(model, preprocessor, your_data)
"""
import os
import json
import joblib
import pandas as pd
import numpy as np

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_model():
    """
    Load the trained model and preprocessor
    
    Returns:
        tuple: (model, preprocessor)
    """
    model_path = os.path.join(SCRIPT_DIR, 'model.{model_format}')
    preprocessor_path = os.path.join(SCRIPT_DIR, 'preprocessor.pkl')
    
    model = joblib.load(model_path)
    
    preprocessor = None
    if os.path.exists(preprocessor_path):
        preprocessor = joblib.load(preprocessor_path)
    
    return model, preprocessor


def load_model_info():
    """Load model information/metadata"""
    info_path = os.path.join(SCRIPT_DIR, 'model_info.json')
    with open(info_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_feature_schema():
    """Load feature schema for input validation"""
    schema_path = os.path.join(SCRIPT_DIR, 'feature_schema.json')
    with open(schema_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def predict(model, preprocessor, data):
    """
    Make predictions using the trained model
    
    Args:
        model: Loaded model object
        preprocessor: Loaded preprocessor (can be None)
        data: Input data (dict, DataFrame, or numpy array)
        
    Returns:
        Prediction result
    """
    # Convert dict to DataFrame if needed
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    elif isinstance(data, list):
        data = pd.DataFrame(data)
    
    # Apply preprocessing if available
    if preprocessor is not None:
        data = preprocessor.transform(data)
    
    # Make prediction
    prediction = model.predict(data)
    
    return prediction


def predict_proba(model, preprocessor, data):
    """
    Get prediction probabilities (for classification models)
    
    Args:
        model: Loaded model object
        preprocessor: Loaded preprocessor (can be None)
        data: Input data
        
    Returns:
        Probability predictions
    """
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    elif isinstance(data, list):
        data = pd.DataFrame(data)
    
    if preprocessor is not None:
        data = preprocessor.transform(data)
    
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(data)
    else:
        return model.predict(data)


# Example usage
if __name__ == '__main__':
    # Load model info
    info = load_model_info()
    print(f"Model: {{info['model_name']}}")
    print(f"Type: {{info['problem_type']}}")
    print(f"Target: {{info['target_column']}}")
    print(f"Best Score: {{info['best_score']}}")
    print(f"Features: {{info['num_features']}}")
    print()
    
    # Load model
    model, preprocessor = load_model()
    print("Model loaded successfully!")
    print()
    
    # Show example usage
    schema = load_feature_schema()
    print("Required features:")
    for feature in schema.get('features', {{}}).keys():
        print(f"  - {{feature}}")
'''
        return loader_code
    
    def _generate_ui_schema(
        self,
        feature_schema: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate UI schema for automatic form generation"""
        
        ui_schema = {
            'model_name': metadata.get('name', 'Unnamed Model'),
            'version': metadata.get('version', '1.0.0'),
            'target_column': metadata.get('target_column'),
            'problem_type': metadata.get('problem_type'),
            'fields': []
        }
        
        for feature_name, feature_info in feature_schema.get('features', {}).items():
            field = self._create_ui_field(feature_name, feature_info)
            ui_schema['fields'].append(field)
        
        return ui_schema
    
    def _create_ui_field(self, name: str, info: Dict[str, Any]) -> Dict[str, Any]:
        """Create UI field definition from feature info"""
        
        field = {
            'name': name,
            'label': name.replace('_', ' ').title(),
            'required': True
        }
        
        dtype = info.get('dtype', 'object')
        semantic_type = info.get('semantic_type', 'unknown')
        
        if semantic_type == 'categorical' or dtype in ['object', 'category']:
            field['type'] = 'categorical'
            field['input_type'] = 'dropdown'
            field['options'] = info.get('categories', [])
        
        elif semantic_type == 'numeric' or dtype in ['int64', 'float64', 'int32', 'float32']:
            field['type'] = 'number'
            field['input_type'] = 'number'
            field['min'] = info.get('min')
            field['max'] = info.get('max')
            field['default'] = info.get('mean') or info.get('median')
        
        elif semantic_type == 'boolean':
            field['type'] = 'boolean'
            field['input_type'] = 'checkbox'
        
        else:
            field['type'] = 'text'
            field['input_type'] = 'text'
        
        return field
    
    def _generate_requirements(self, package_dir: str, model_format: str):
        """Generate requirements.txt for the package"""
        
        requirements = [
            'pandas>=2.0.0',
            'numpy>=1.24.0',
            'scikit-learn>=1.3.0',
            'joblib>=1.3.0',
            'streamlit>=1.28.0'
        ]
        
        if model_format == 'h5':
            requirements.append('tensorflow>=2.15.0')
        
        requirements_path = os.path.join(package_dir, 'requirements.txt')
        with open(requirements_path, 'w') as f:
            f.write('\n'.join(requirements))
    
    def _generate_streamlit_app(self, metadata: Dict[str, Any], feature_schema: Dict[str, Any]) -> str:
        """Generate a Streamlit app for model predictions"""
        
        model_name = metadata.get('name', 'ML Model')
        target_col = metadata.get('target_column', 'target')
        problem_type = metadata.get('problem_type', 'classification')
        best_model = metadata.get('best_model', 'Unknown')
        best_score = metadata.get('best_score', 0)
        
        # Build form fields dynamically

        # Build form fields dynamically
        features = feature_schema.get('features', {})
        feature_fields = []
        
        for feature_name, feature_info in features.items():
            dtype = feature_info.get('dtype', 'object')
            semantic_type = feature_info.get('semantic_type', 'unknown')
            label = feature_name.replace('_', ' ').title()
            
            field_code = ""
            if semantic_type == 'categorical':
                categories = feature_info.get('categories', [])
                if categories:
                    cats_str = str(categories)
                    field_code = f'''
        st.session_state.input_data["{feature_name}"] = st.selectbox(
            "{label}",
            options={cats_str},
            key="{feature_name}"
        )'''
                else:
                    field_code = f'''
        st.session_state.input_data["{feature_name}"] = st.text_input("{label}", key="{feature_name}")'''
            
            elif semantic_type == 'numeric':
                min_val = feature_info.get('min', 0)
                max_val = feature_info.get('max', 100)
                default_val = feature_info.get('mean') or feature_info.get('median') or 0
                
                # Clean up values
                min_val = float(min_val) if min_val is not None else 0.0
                max_val = float(max_val) if max_val is not None else 100.0
                default_val = float(default_val) if default_val is not None else min_val
                
                field_code = f'''
        st.session_state.input_data["{feature_name}"] = st.number_input(
            "{label}",
            min_value={min_val},
            max_value={max_val},
            value={default_val},
            key="{feature_name}"
        )'''
            
            elif semantic_type == 'boolean':
                field_code = f'''
        st.session_state.input_data["{feature_name}"] = st.checkbox("{label}", key="{feature_name}")'''
            
            else:
                field_code = f'''
        st.session_state.input_data["{feature_name}"] = st.text_input("{label}", key="{feature_name}")'''
            
            feature_fields.append(field_code)

        # Split fields into two columns
        mid_idx = (len(feature_fields) + 1) // 2
        col1_fields = feature_fields[:mid_idx]
        col2_fields = feature_fields[mid_idx:]
        
        col1_code = "".join(col1_fields) if col1_fields else "        pass"
        col2_code = "".join(col2_fields) if col2_fields else "        pass"
        
        streamlit_code = f'''"""
Streamlit Prediction App - {model_name}
Auto-generated Streamlit UI for making predictions

Usage:
    streamlit run streamlit_app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import os

# Page config
st.set_page_config(
    page_title="{model_name} - Predictions",
    page_icon="🤖",
    layout="wide"
)

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and preprocessor
@st.cache_resource
def load_model():
    import joblib
    model_path = os.path.join(SCRIPT_DIR, 'model.pkl')
    preprocessor_path = os.path.join(SCRIPT_DIR, 'preprocessor.pkl')
    
    model = joblib.load(model_path)
    preprocessor = None
    if os.path.exists(preprocessor_path):
        preprocessor = joblib.load(preprocessor_path)
    
    return model, preprocessor

@st.cache_data
def load_model_info():
    info_path = os.path.join(SCRIPT_DIR, 'model_info.json')
    with open(info_path, 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_data
def load_target_classes():
    tc_path = os.path.join(SCRIPT_DIR, 'target_classes.json')
    if os.path.exists(tc_path):
        with open(tc_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

# Load model
model, preprocessor = load_model()
model_info = load_model_info()
target_classes = load_target_classes()
target_col = "{target_col}"
problem_type = "{problem_type}"

# Header
st.title("🤖 " + model_info.get('model_name', '{model_name}'))
st.markdown(model_info.get('description', 'Make predictions using the trained model'))

# Sidebar with model info
with st.sidebar:
    st.header("📊 Model Information")
    st.metric("Algorithm", "{best_model}")
    st.metric("Score", f"{{model_info.get('best_score', 0):.2%}}")
    st.metric("Target", target_col)
    st.metric("Features", model_info.get('num_features', 0))
    if target_classes:
        st.metric("Classes", len(target_classes))
    
    st.divider()
    st.caption("Model trained with InferX-ML")

# Initialize session state for input data
if 'input_data' not in st.session_state:
    st.session_state.input_data = {{}}

# Main content
st.header("🎯 Make Predictions")

# Create input form
with st.form("prediction_form"):
    st.subheader("📝 Enter Feature Values")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:{col1_code}
    
    with col2:{col2_code}
    
    submitted = st.form_submit_button("✨ Make Prediction", use_container_width=True)

if submitted:
    try:
        # Create DataFrame from input
        input_df = pd.DataFrame([st.session_state.input_data])
        
        # Apply preprocessing if available
        if preprocessor is not None:
            input_processed = preprocessor.transform(input_df)
        else:
            input_processed = input_df
        
        # Make prediction
        raw_prediction = model.predict(input_processed)
        raw_value = raw_prediction[0]
        
        # ═══ Display Result ═══
        st.divider()
        
        if problem_type == 'classification':
            # Decode label if target_classes mapping exists
            if target_classes is not None and isinstance(raw_value, (int, np.integer)):
                predicted_label = target_classes[int(raw_value)] if int(raw_value) < len(target_classes) else str(raw_value)
            else:
                predicted_label = str(raw_value)
            
            st.success(f"**Predicted {{target_col}}:** {{predicted_label}}")
            
            # Show all class probabilities
            if hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(input_processed)[0]
                    confidence = float(max(proba)) * 100
                    
                    st.metric("Confidence", f"{{confidence:.1f}}%")
                    
                    # Build probability DataFrame for chart
                    if target_classes and len(target_classes) == len(proba):
                        prob_data = pd.DataFrame({{
                            'Class': target_classes,
                            'Probability (%)': [round(p * 100, 2) for p in proba]
                        }}).sort_values('Probability (%)', ascending=False)
                    else:
                        prob_data = pd.DataFrame({{
                            'Class': [f'Class {{i}}' for i in range(len(proba))],
                            'Probability (%)': [round(p * 100, 2) for p in proba]
                        }}).sort_values('Probability (%)', ascending=False)
                    
                    st.subheader("📊 Class Probabilities")
                    st.bar_chart(prob_data.set_index('Class'))
                    
                    # Show top 5 as a table
                    with st.expander("📋 Detailed Probabilities", expanded=False):
                        st.dataframe(prob_data.reset_index(drop=True), use_container_width=True)
                except Exception:
                    pass
        
        else:
            # Regression — round to 2 decimal places
            if isinstance(raw_value, (float, np.floating)):
                display_value = round(float(raw_value), 2)
            else:
                display_value = raw_value
            
            st.success(f"**Predicted {{target_col}}:** {{display_value}}")
        
        # Show input summary
        with st.expander("📝 Input Summary", expanded=True):
            summary_df = pd.DataFrame([st.session_state.input_data]).T
            summary_df.columns = ['Value']
            summary_df.index.name = 'Feature'
            st.dataframe(summary_df, use_container_width=True)
            
    except Exception as e:
        st.error(f"Prediction failed: {{str(e)}}")

# Footer
st.divider()
st.caption("Generated by InferX-ML | Powered by Streamlit")
'''
        return streamlit_code
    
    def create_zip(self, package_dir: str) -> str:
        """Create a zip file of the package"""
        zip_path = shutil.make_archive(package_dir, 'zip', package_dir)
        return zip_path


def create_feature_schema(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """
    Create feature schema from a DataFrame
    
    Args:
        df: Training DataFrame
        target_column: Name of the target column
        
    Returns:
        Feature schema dictionary
    """
    schema = {
        'target_column': target_column,
        'features': {}
    }
    
    for col in df.columns:
        if col == target_column:
            continue
        
        series = df[col]
        feature_info = {
            'dtype': str(series.dtype),
            'missing_count': int(series.isna().sum())
        }
        
        # Determine semantic type and add relevant stats
        if series.dtype in ['object', 'category']:
            feature_info['semantic_type'] = 'categorical'
            feature_info['categories'] = series.dropna().unique().tolist()[:50]
        
        elif np.issubdtype(series.dtype, np.number):
            feature_info['semantic_type'] = 'numeric'
            feature_info['min'] = float(series.min()) if not pd.isna(series.min()) else None
            feature_info['max'] = float(series.max()) if not pd.isna(series.max()) else None
            feature_info['mean'] = float(series.mean()) if not pd.isna(series.mean()) else None
            feature_info['median'] = float(series.median()) if not pd.isna(series.median()) else None
        
        elif np.issubdtype(series.dtype, np.bool_):
            feature_info['semantic_type'] = 'boolean'
        
        else:
            feature_info['semantic_type'] = 'unknown'
        
        schema['features'][col] = feature_info
    
    return schema
