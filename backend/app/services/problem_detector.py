"""
Problem Detector Service
Automatically detects the ML problem type from data and goals
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple


class ProblemDetector:
    """Detect ML problem type from data characteristics"""
    
    def __init__(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        goal_description: str = ""
    ):
        self.df = df
        self.target_column = target_column
        self.goal_description = goal_description or ""
        self.target = self.df[target_column] if target_column and target_column in self.df.columns else None
    
    def detect(self) -> Dict[str, Any]:
        """Detect problem type and return recommendations"""
        if self.target_column is None:
            # No target = clustering or unsupervised
            return {
                'problem_type': 'clustering',
                'confidence': 0.8,
                'reason': 'No target column specified',
                'recommended_algorithms': ['kmeans', 'dbscan']
            }

        if self.target is None:
            return {
                'problem_type': 'unknown',
                'confidence': 0.0,
                'reason': f'Target column "{self.target_column}" not found',
                'target_column': self.target_column,
                'recommended_algorithms': []
            }
        
        problem_type, confidence, reason = self._determine_problem_type()
        
        return {
            'problem_type': problem_type,
            'confidence': confidence,
            'reason': reason,
            'target_column': self.target_column,
            'target_info': self._analyze_target(),
            'data_type_signals': self._get_data_type_signals(),
            'recommended_algorithms': self._get_recommended_algorithms(problem_type),
            'preprocessing_suggestions': self._get_preprocessing_suggestions()
        }
    
    def _determine_problem_type(self) -> Tuple[str, float, str]:
        """Determine if classification, regression, or timeseries"""
        target = self.target.dropna()
        if target.empty:
            return 'unknown', 0.0, 'Target column contains no usable values'
        
        # Check for timeseries
        if self._is_timeseries_forecast_goal():
            return 'timeseries', 0.9, 'Date/time column found and goal indicates forecasting'
        
        # Check target column characteristics
        unique_count = target.nunique()
        unique_ratio = unique_count / len(target)
        is_numeric = np.issubdtype(target.dtype, np.number)
        is_bool = pd.api.types.is_bool_dtype(target)
        is_integer_like = is_numeric and np.all(np.isclose(target, target.astype(int)))
        target_name = (self.target_column or '').lower()

        if is_bool:
            return 'binary_classification', 0.98, 'Boolean target has two possible labels'

        if self._looks_like_identifier(target):
            return 'classification', 0.45, 'Target looks like an identifier; verify this is really what you want to predict'

        if self._looks_like_datetime_target(target):
            return 'timeseries', 0.75, 'Target values look like dates or timestamps'
        
        # Classification: few repeated labels. This covers strings, booleans, and coded numeric classes.
        if unique_count == 2:
            return 'binary_classification', 0.95, 'Target has exactly two distinct labels'

        classification_name_hints = [
            'class', 'category', 'label', 'type', 'status', 'segment', 'churn',
            'fraud', 'approved', 'default', 'outcome', 'result'
        ]
        has_class_name_hint = any(hint in target_name for hint in classification_name_hints)

        if unique_count <= 20 and (unique_ratio <= 0.2 or has_class_name_hint):
            if unique_count == 2:
                return 'binary_classification', 0.95, 'Target has exactly two distinct labels'
            return 'multiclass_classification', 0.9, f'Target has {unique_count} repeated categories'

        if not is_numeric and unique_count <= min(100, max(20, len(target) * 0.5)):
            return 'multiclass_classification', 0.75, 'Non-numeric target is best handled as categories'
        
        # Regression: numeric with many unique values
        if is_numeric:
            if is_integer_like and unique_count <= 50 and unique_ratio <= 0.1:
                return 'multiclass_classification', 0.75, 'Integer target has repeated coded classes'
            return 'regression', 0.9, 'Numeric target has many distinct continuous values'
        
        # Text target might be classification
        if target.dtype == 'object':
            if unique_count <= 50:
                return 'multiclass_classification', 0.7, 'Text target has a manageable number of labels'
        
        return 'regression', 0.5, 'Fallback choice; target pattern is ambiguous'
    
    def _is_timeseries(self) -> bool:
        """Check if data is timeseries"""
        for col in self.df.columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in ['date', 'time', 'datetime', 'timestamp']):
                try:
                    pd.to_datetime(self.df[col].head(100))
                    return True
                except:
                    pass
        return False

    def _is_timeseries_forecast_goal(self) -> bool:
        """Only select timeseries when the data and goal both point that way."""
        goal = self.goal_description.lower()
        forecast_terms = ['forecast', 'future', 'trend', 'seasonal', 'time series', 'timeseries', 'next']
        return self._is_timeseries() and any(term in goal for term in forecast_terms)

    def _looks_like_identifier(self, values: pd.Series) -> bool:
        """Detect ID-like targets that should not be predicted accidentally."""
        name = (self.target_column or '').lower()
        unique_ratio = values.nunique() / len(values)
        return (
            unique_ratio > 0.9 and
            (name == 'id' or name.endswith('_id') or 'uuid' in name or 'identifier' in name)
        )

    def _looks_like_datetime_target(self, values: pd.Series) -> bool:
        if pd.api.types.is_datetime64_any_dtype(values):
            return True
        if values.dtype != 'object':
            return False
        try:
            parsed = pd.to_datetime(values.head(100), errors='coerce')
            return parsed.notna().mean() >= 0.8
        except Exception:
            return False

    def _get_data_type_signals(self) -> Dict[str, Any]:
        """Return compact signals used by deterministic and AI verification."""
        target = self.target.dropna() if self.target is not None else pd.Series(dtype=object)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        datetime_like_cols = []
        for col in self.df.columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in ['date', 'time', 'datetime', 'timestamp']):
                datetime_like_cols.append(col)

        return {
            'rows': int(len(self.df)),
            'columns': int(len(self.df.columns)),
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'datetime_like_columns': datetime_like_cols,
            'target_unique_count': int(target.nunique()) if not target.empty else 0,
            'target_unique_ratio': float(target.nunique() / len(target)) if not target.empty else 0.0
        }
    
    def _analyze_target(self) -> Dict[str, Any]:
        """Analyze target column"""
        if self.target is None:
            return {}
        
        analysis = {
            'dtype': str(self.target.dtype),
            'unique_count': int(self.target.nunique()),
            'missing_count': int(self.target.isna().sum()),
            'value_distribution': self.target.value_counts().head(10).to_dict()
        }
        
        if np.issubdtype(self.target.dtype, np.number):
            analysis.update({
                'min': float(self.target.min()),
                'max': float(self.target.max()),
                'mean': float(self.target.mean()),
                'std': float(self.target.std())
            })
        
        return analysis
    
    def _get_recommended_algorithms(self, problem_type: str) -> list:
        """Get recommended algorithms based on problem type"""
        algorithms = {
            'binary_classification': ['logistic_regression', 'random_forest', 'xgboost'],
            'multiclass_classification': ['random_forest', 'xgboost', 'logistic_regression'],
            'regression': ['linear_regression', 'random_forest', 'xgboost'],
            'clustering': ['kmeans', 'dbscan'],
            'timeseries': ['arima', 'prophet', 'lstm']
        }
        return algorithms.get(problem_type, ['random_forest'])
    
    def _get_preprocessing_suggestions(self) -> list:
        """Get preprocessing suggestions based on data"""
        suggestions = []
        
        # Check for missing values
        missing = self.df.isna().sum().sum()
        if missing > 0:
            suggestions.append('Handle missing values (imputation or removal)')
        
        # Check for categorical columns
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            suggestions.append(f'Encode categorical columns: {list(cat_cols)}')
        
        # Check for numeric scaling
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            ranges = self.df[num_cols].max() - self.df[num_cols].min()
            if ranges.max() > 1000:
                suggestions.append('Consider feature scaling for numeric columns')
        
        # Check for high cardinality
        for col in cat_cols:
            if self.df[col].nunique() > 50:
                suggestions.append(f'High cardinality in {col} - consider grouping or target encoding')
        
        return suggestions
