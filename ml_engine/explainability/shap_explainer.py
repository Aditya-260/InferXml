"""
SHAP Explainability Module
Feature importance and model explanations using SHAP
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import io
import base64

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class SHAPExplainer:
    """
    SHAP-based model explainability
    Provides feature importance and prediction explanations
    """
    
    def __init__(self, model, model_type: str = 'tree'):
        """
        Initialize SHAP explainer
        
        Args:
            model: Trained model object
            model_type: 'tree', 'linear', 'kernel', or 'deep'
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required. Install with: pip install shap")
        
        self.model = model
        self.model_type = model_type
        self.explainer_ = None
        self.shap_values_ = None
        self.expected_value_ = None
        self.feature_names_ = None
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        feature_names: List[str] = None
    ) -> 'SHAPExplainer':
        """
        Fit the explainer
        
        Args:
            X: Background data (training data sample)
            feature_names: Feature names
            
        Returns:
            self
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_array = X.values
        else:
            self.feature_names_ = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
            X_array = X
        
        # Choose explainer based on model type
        if self.model_type == 'tree':
            self.explainer_ = shap.TreeExplainer(self.model)
        elif self.model_type == 'linear':
            self.explainer_ = shap.LinearExplainer(self.model, X_array)
        elif self.model_type == 'kernel':
            # Sample for kernel explainer (slower)
            if len(X_array) > 100:
                sample_idx = np.random.choice(len(X_array), 100, replace=False)
                X_sample = X_array[sample_idx]
            else:
                X_sample = X_array
            
            self.explainer_ = shap.KernelExplainer(self.model.predict, X_sample)
        elif self.model_type == 'deep':
            self.explainer_ = shap.DeepExplainer(self.model, X_array)
        else:
            # Try tree explainer first, fall back to kernel
            try:
                self.explainer_ = shap.TreeExplainer(self.model)
            except:
                self.explainer_ = shap.KernelExplainer(self.model.predict, X_array[:100])
        
        self.expected_value_ = self.explainer_.expected_value
        
        return self
    
    def explain(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        max_samples: int = None
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanations
        
        Args:
            X: Data to explain
            max_samples: Maximum samples to explain
            
        Returns:
            Dictionary with SHAP values and visualizations
        """
        if self.explainer_ is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        if max_samples and len(X_array) > max_samples:
            X_array = X_array[:max_samples]
        
        # Calculate SHAP values
        self.shap_values_ = self.explainer_.shap_values(X_array)
        
        # Handle multi-class output
        if isinstance(self.shap_values_, list):
            # For binary classification, use class 1
            shap_values = self.shap_values_[1] if len(self.shap_values_) == 2 else self.shap_values_[0]
        else:
            shap_values = self.shap_values_
        
        # Calculate feature importance
        feature_importance = self._calculate_feature_importance(shap_values)
        
        return {
            'shap_values': shap_values,
            'expected_value': self.expected_value_,
            'feature_importance': feature_importance,
            'feature_names': self.feature_names_
        }
    
    def explain_single(
        self,
        x: Union[pd.Series, np.ndarray],
        include_plot: bool = True
    ) -> Dict[str, Any]:
        """
        Explain a single prediction
        
        Args:
            x: Single instance to explain
            include_plot: Include base64 encoded plot
            
        Returns:
            Explanation dictionary
        """
        if isinstance(x, pd.Series):
            x_array = x.values.reshape(1, -1)
        elif len(x.shape) == 1:
            x_array = x.reshape(1, -1)
        else:
            x_array = x
        
        # Get SHAP values
        shap_values = self.explainer_.shap_values(x_array)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
        
        shap_values = shap_values.flatten()
        
        # Get contribution for each feature
        contributions = []
        for i, (name, value, shap_val) in enumerate(zip(
            self.feature_names_,
            x_array.flatten(),
            shap_values
        )):
            contributions.append({
                'feature': name,
                'value': float(value),
                'shap_value': float(shap_val),
                'impact': 'positive' if shap_val > 0 else 'negative',
                'abs_impact': abs(float(shap_val))
            })
        
        # Sort by absolute impact
        contributions.sort(key=lambda x: x['abs_impact'], reverse=True)
        
        result = {
            'contributions': contributions,
            'expected_value': float(self.expected_value_) if not isinstance(self.expected_value_, list) else float(self.expected_value_[0]),
            'prediction_offset': float(np.sum(shap_values))
        }
        
        # Generate plot
        if include_plot:
            result['plot_base64'] = self._generate_waterfall_plot(x_array, shap_values)
        
        return result
    
    def _calculate_feature_importance(
        self,
        shap_values: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Calculate feature importance from SHAP values"""
        # Mean absolute SHAP value per feature
        importance = np.abs(shap_values).mean(axis=0)
        
        # Normalize
        total = importance.sum()
        if total > 0:
            importance_normalized = importance / total
        else:
            importance_normalized = importance
        
        # Create sorted list
        feature_importance = []
        for i, name in enumerate(self.feature_names_):
            feature_importance.append({
                'feature': name,
                'importance': float(importance[i]),
                'importance_normalized': float(importance_normalized[i]),
                'rank': 0  # Will be set after sorting
            })
        
        # Sort and add rank
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        for i, item in enumerate(feature_importance):
            item['rank'] = i + 1
        
        return feature_importance
    
    def _generate_waterfall_plot(
        self,
        x: np.ndarray,
        shap_values: np.ndarray
    ) -> str:
        """Generate waterfall plot as base64"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create explanation object
        explanation = shap.Explanation(
            values=shap_values,
            base_values=self.expected_value_ if not isinstance(self.expected_value_, list) else self.expected_value_[0],
            data=x.flatten(),
            feature_names=self.feature_names_
        )
        
        shap.waterfall_plot(explanation, show=False)
        
        # Save to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return img_base64
    
    def get_summary_plot(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        plot_type: str = 'bar'
    ) -> str:
        """
        Generate summary plot as base64
        
        Args:
            X: Data to explain
            plot_type: 'bar', 'beeswarm', or 'violin'
            
        Returns:
            Base64 encoded plot
        """
        if self.shap_values_ is None:
            self.explain(X)
        
        shap_values = self.shap_values_
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if plot_type == 'bar':
            shap.summary_plot(
                shap_values,
                X_array,
                feature_names=self.feature_names_,
                plot_type='bar',
                show=False
            )
        else:
            shap.summary_plot(
                shap_values,
                X_array,
                feature_names=self.feature_names_,
                show=False
            )
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close('all')
        
        return img_base64
    
    def generate_text_explanation(
        self,
        contributions: List[Dict[str, Any]],
        top_k: int = 5
    ) -> str:
        """
        Generate human-readable text explanation
        
        Args:
            contributions: Feature contributions from explain_single
            top_k: Number of top features to include
            
        Returns:
            Text explanation
        """
        top_features = contributions[:top_k]
        
        lines = ["The prediction was primarily influenced by:"]
        
        for i, feat in enumerate(top_features, 1):
            direction = "increased" if feat['shap_value'] > 0 else "decreased"
            lines.append(
                f"{i}. **{feat['feature']}** (value: {feat['value']:.2f}) "
                f"{direction} the prediction by {abs(feat['shap_value']):.4f}"
            )
        
        return "\n".join(lines)


class FeatureImportanceCalculator:
    """
    Calculate feature importance using various methods
    """
    
    @staticmethod
    def from_tree_model(model, feature_names: List[str]) -> List[Dict[str, Any]]:
        """
        Get feature importance from tree-based models
        
        Args:
            model: Tree model (RF, XGBoost, etc.)
            feature_names: Feature names
            
        Returns:
            Sorted feature importance list
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            raise ValueError("Model doesn't have feature_importances_ attribute")
        
        result = []
        for name, imp in zip(feature_names, importances):
            result.append({
                'feature': name,
                'importance': float(imp)
            })
        
        result.sort(key=lambda x: x['importance'], reverse=True)
        return result
    
    @staticmethod
    def from_permutation(
        model,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        n_repeats: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Calculate permutation importance
        
        Args:
            model: Any model with predict method
            X, y: Test data
            feature_names: Feature names
            n_repeats: Number of permutations
            
        Returns:
            Sorted feature importance list
        """
        from sklearn.inspection import permutation_importance
        
        result = permutation_importance(model, X, y, n_repeats=n_repeats)
        
        importance_list = []
        for name, mean, std in zip(feature_names, result.importances_mean, result.importances_std):
            importance_list.append({
                'feature': name,
                'importance': float(mean),
                'std': float(std)
            })
        
        importance_list.sort(key=lambda x: x['importance'], reverse=True)
        return importance_list
