"""
Time-Series Data Preprocessor
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib


class TimeSeriesPreprocessor:
    """
    Preprocessing for time-series data
    Handles: datetime parsing, resampling, feature engineering, scaling
    """
    
    def __init__(
        self,
        datetime_column: str = None,
        target_column: str = None,
        freq: str = None,  # 'D', 'H', 'W', 'M', etc.
        fill_method: str = 'ffill',
        scale: bool = True,
        use_robust_scaler: bool = True,
        outlier_clip: bool = False
    ):
        """
        Initialize preprocessor
        
        Args:
            datetime_column: Name of datetime column
            target_column: Name of target column
            freq: Resampling frequency
            fill_method: Method to fill missing values
            scale: Whether to scale the data
            use_robust_scaler: Uses RobustScaler if True, otherwise StandardScaler
            outlier_clip: Whether to apply 1st/99th percentile clipping
        """
        self.datetime_column = datetime_column
        self.target_column = target_column
        self.freq = freq
        self.fill_method = fill_method
        self.scale = scale
        self.use_robust_scaler = use_robust_scaler
        self.outlier_clip = outlier_clip
        
        # Fitted attributes
        self.scaler_ = None
        self.datetime_col_detected_ = None
        self.feature_columns_ = []
    
    def fit(self, df: pd.DataFrame) -> 'TimeSeriesPreprocessor':
        """
        Fit the preprocessor
        
        Args:
            df: Input DataFrame
            
        Returns:
            self
        """
        # Detect datetime column if not specified
        if self.datetime_column is None:
            self.datetime_col_detected_ = self._detect_datetime_column(df)
        else:
            self.datetime_col_detected_ = self.datetime_column
        
        # Get feature columns
        self.feature_columns_ = [
            col for col in df.columns 
            if col not in [self.datetime_col_detected_, self.target_column]
            and np.issubdtype(df[col].dtype, np.number)
        ]
        
        # Fit scaler
        if self.scale:
            self.scaler_ = RobustScaler() if self.use_robust_scaler else StandardScaler()
            numeric_cols = self.feature_columns_.copy()
            if self.target_column:
                numeric_cols.append(self.target_column)
            
            numeric_data = df[numeric_cols].values
            self.scaler_.fit(numeric_data)
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data
        
        Args:
            df: Input DataFrame
            
        Returns:
            Processed DataFrame
        """
        df = df.copy()
        
        # Parse datetime
        if self.datetime_col_detected_:
            df[self.datetime_col_detected_] = pd.to_datetime(df[self.datetime_col_detected_])
            df = df.sort_values(self.datetime_col_detected_)
            df = df.set_index(self.datetime_col_detected_)
        
        # Resample if frequency specified
        if self.freq:
            df = df.resample(self.freq).mean()
        
        # Fill missing values
        if self.fill_method == 'ffill':
            df = df.fillna(method='ffill')
        elif self.fill_method == 'bfill':
            df = df.fillna(method='bfill')
        elif self.fill_method == 'interpolate':
            df = df.interpolate(method='linear')
        else:
            df = df.fillna(0)
        
        # Outlier Clipping (Winsorization)
        if hasattr(self, 'outlier_clip') and self.outlier_clip:
            numeric_cols = self.feature_columns_.copy()
            if self.target_column and self.target_column in df.columns:
                numeric_cols.append(self.target_column)
            
            # Use 1st and 99th percentiles
            for col in numeric_cols:
                lower = df[col].quantile(0.01)
                upper = df[col].quantile(0.99)
                df[col] = df[col].clip(lower, upper)
                
        # Scale
        if self.scale and self.scaler_ is not None:
            numeric_cols = self.feature_columns_.copy()
            if self.target_column and self.target_column in df.columns:
                numeric_cols.append(self.target_column)
            
            df[numeric_cols] = self.scaler_.transform(df[numeric_cols])
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform"""
        self.fit(df)
        return self.transform(df)
    
    def _detect_datetime_column(self, df: pd.DataFrame) -> Optional[str]:
        """Detect datetime column by name or content"""
        # Check by name
        datetime_keywords = ['date', 'time', 'datetime', 'timestamp', 'created', 'updated']
        for col in df.columns:
            if any(kw in col.lower() for kw in datetime_keywords):
                return col
        
        # Check by content
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]':
                return col
            
            # Try parsing
            try:
                pd.to_datetime(df[col].head(100))
                return col
            except:
                pass
        
        return None
    
    def create_time_features(self, df: pd.DataFrame, encode_cyclical: bool = True) -> pd.DataFrame:
        """
        Create time-based features from datetime index
        
        Args:
            df: DataFrame with datetime index
            encode_cyclical: Whether to sine/cosine encode cyclical features (industry best practice)
            
        Returns:
            DataFrame with additional time features
        """
        df = df.copy()
        
        if isinstance(df.index, pd.DatetimeIndex):
            # Extract basic components
            hour = df.index.hour
            day_of_week = df.index.dayofweek
            day_of_month = df.index.day
            month = df.index.month
            
            if encode_cyclical:
                # Sine/Cosine encoding to preserve relative ordering (e.g. Hour 23 is next to Hour 0)
                df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
                df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
                
                df['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7.0)
                df['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7.0)
                
                # Assume 31 days roughly for generic normalization, better than not encoding
                df['day_of_month_sin'] = np.sin(2 * np.pi * day_of_month / 31.0)
                df['day_of_month_cos'] = np.cos(2 * np.pi * day_of_month / 31.0)
                
                df['month_sin'] = np.sin(2 * np.pi * month / 12.0)
                df['month_cos'] = np.cos(2 * np.pi * month / 12.0)
            else:
                df['hour'] = hour
                df['day_of_week'] = day_of_week
                df['day_of_month'] = day_of_month
                df['month'] = month
                
            df['quarter'] = df.index.quarter
            df['year'] = df.index.year
            df['is_weekend'] = day_of_week >= 5
            df['is_month_start'] = df.index.is_month_start
            df['is_month_end'] = df.index.is_month_end
        
        return df
    
    def create_lag_features(
        self,
        df: pd.DataFrame,
        column: str,
        lags: List[int] = [1, 7, 14, 30]
    ) -> pd.DataFrame:
        """
        Create lag features
        
        Args:
            df: Input DataFrame
            column: Column to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features
        """
        df = df.copy()
        
        for lag in lags:
            df[f'{column}_lag_{lag}'] = df[column].shift(lag)
        
        return df
    
    def create_rolling_features(
        self,
        df: pd.DataFrame,
        column: str,
        windows: List[int] = [7, 14, 30]
    ) -> pd.DataFrame:
        """
        Create rolling window features
        
        Args:
            df: Input DataFrame
            column: Column to create features for
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features
        """
        df = df.copy()
        
        for window in windows:
            df[f'{column}_rolling_mean_{window}'] = df[column].rolling(window=window).mean()
            df[f'{column}_rolling_std_{window}'] = df[column].rolling(window=window).std()
            df[f'{column}_rolling_min_{window}'] = df[column].rolling(window=window).min()
            df[f'{column}_rolling_max_{window}'] = df[column].rolling(window=window).max()
        
        return df
    
    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """Inverse transform scaled values"""
        if self.scaler_ is not None:
            return self.scaler_.inverse_transform(values.reshape(-1, 1)).flatten()
        return values
    
    def save(self, path: str):
        """Save preprocessor"""
        joblib.dump(self, path)
    
    @classmethod
    def load(cls, path: str) -> 'TimeSeriesPreprocessor':
        """Load preprocessor"""
        return joblib.load(path)
