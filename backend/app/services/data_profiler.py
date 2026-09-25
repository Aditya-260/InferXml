"""
Data Profiler Service
Automatically analyzes and profiles uploaded datasets
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


class DataProfiler:
    """Automatic data profiling for uploaded datasets"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.profile = {}
    
    def profile_dataset(self) -> Dict[str, Any]:
        """Generate complete data profile"""
        self.profile = {
            'basic_info': self._get_basic_info(),
            'column_profiles': self._profile_columns(),
            'missing_values': self._analyze_missing(),
            'correlations': self._compute_correlations(),
            'data_quality_score': self._calculate_quality_score()
        }
        return self.profile
    
    def _get_basic_info(self) -> Dict[str, Any]:
        """Get basic dataset information"""
        return {
            'num_rows': len(self.df),
            'num_columns': len(self.df.columns),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024 / 1024,
            'duplicate_rows': self.df.duplicated().sum(),
            'column_names': list(self.df.columns)
        }
    
    def _profile_columns(self) -> Dict[str, Dict]:
        """Profile each column"""
        profiles = {}
        
        for col in self.df.columns:
            profiles[col] = self._profile_single_column(col)
        
        return profiles
    
    def _profile_single_column(self, column: str) -> Dict[str, Any]:
        """Profile a single column"""
        series = self.df[column]
        dtype = str(series.dtype)
        
        profile = {
            'dtype': dtype,
            'missing_count': series.isna().sum(),
            'missing_pct': (series.isna().sum() / len(series)) * 100,
            'unique_count': series.nunique()
        }
        
        # Infer semantic type
        profile['semantic_type'] = self._infer_semantic_type(series)
        
        # Add type-specific stats
        if np.issubdtype(series.dtype, np.number):
            profile.update(self._numeric_stats(series))
        elif series.dtype == 'object' or str(series.dtype) == 'category':
            profile.update(self._categorical_stats(series))
        elif np.issubdtype(series.dtype, np.datetime64):
            profile.update(self._datetime_stats(series))
        
        return profile
    
    def _infer_semantic_type(self, series: pd.Series) -> str:
        """Infer semantic type of column"""
        col_lower = series.name.lower()
        
        # Check for timestamp columns
        timestamp_keywords = ['date', 'time', 'datetime', 'timestamp', 'created', 'updated']
        if any(kw in col_lower for kw in timestamp_keywords):
            return 'timestamp'
        
        # Check for ID columns
        if 'id' in col_lower or col_lower.endswith('_id'):
            return 'identifier'
        
        # Check for categorical
        if series.dtype == 'object':
            unique_ratio = series.nunique() / len(series)
            if unique_ratio < 0.05:  # Less than 5% unique = categorical
                return 'categorical'
            return 'text'
        
        # Numeric
        if np.issubdtype(series.dtype, np.number):
            if series.nunique() <= 10:
                return 'categorical'
            return 'numeric'
        
        return 'unknown'
    
    def _numeric_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Statistics for numeric columns"""
        return {
            'min': float(series.min()) if not pd.isna(series.min()) else None,
            'max': float(series.max()) if not pd.isna(series.max()) else None,
            'mean': float(series.mean()) if not pd.isna(series.mean()) else None,
            'median': float(series.median()) if not pd.isna(series.median()) else None,
            'std': float(series.std()) if not pd.isna(series.std()) else None,
            'skewness': float(series.skew()) if not pd.isna(series.skew()) else None,
            'zeros_count': int((series == 0).sum()),
            'negative_count': int((series < 0).sum())
        }
    
    def _categorical_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Statistics for categorical columns"""
        value_counts = series.value_counts().head(10).to_dict()
        return {
            'top_values': {str(k): int(v) for k, v in value_counts.items()},
            'categories': list(series.dropna().unique()[:20])
        }
    
    def _datetime_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Statistics for datetime columns"""
        return {
            'min_date': str(series.min()),
            'max_date': str(series.max()),
            'date_range_days': (series.max() - series.min()).days if not pd.isna(series.min()) else None
        }
    
    def _analyze_missing(self) -> Dict[str, Any]:
        """Analyze missing values"""
        missing = self.df.isna().sum()
        missing_pct = (missing / len(self.df)) * 100
        
        return {
            'total_missing': int(missing.sum()),
            'columns_with_missing': int((missing > 0).sum()),
            'missing_by_column': missing.to_dict(),
            'missing_pct_by_column': missing_pct.to_dict()
        }
    
    def _compute_correlations(self) -> Optional[Dict[str, Dict]]:
        """Compute correlation matrix for numeric columns"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return None
        
        corr_matrix = self.df[numeric_cols].corr()
        return corr_matrix.to_dict()
    
    def _calculate_quality_score(self) -> float:
        """Calculate overall data quality score (0-100)"""
        score = 100.0
        
        # Penalize for missing values
        missing_pct = (self.df.isna().sum().sum() / self.df.size) * 100
        score -= min(missing_pct, 30)  # Max 30 point penalty
        
        # Penalize for duplicates
        duplicate_pct = (self.df.duplicated().sum() / len(self.df)) * 100
        score -= min(duplicate_pct, 20)  # Max 20 point penalty
        
        return max(0, round(score, 2))
    
    def detect_data_type(self) -> str:
        """Detect if dataset is tabular, timeseries, or needs special handling"""
        # Check for timestamp column
        for col in self.df.columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in ['date', 'time', 'datetime', 'timestamp']):
                # Try to parse as datetime
                try:
                    pd.to_datetime(self.df[col].head(100))
                    return 'timeseries'
                except:
                    pass
        
        return 'tabular'
