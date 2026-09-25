"""
Prophet Time-Series Forecasting
Facebook's Prophet for time-series with seasonality
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
import warnings
import joblib

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


class ProphetForecaster:
    """
    Prophet model for time-series forecasting
    Good for data with strong seasonal patterns and holidays
    """
    
    def __init__(
        self,
        yearly_seasonality: bool = 'auto',
        weekly_seasonality: bool = 'auto',
        daily_seasonality: bool = 'auto',
        seasonality_mode: str = 'additive',
        changepoint_prior_scale: float = 0.05,
        holidays: pd.DataFrame = None
    ):
        """
        Initialize Prophet forecaster
        
        Args:
            yearly_seasonality: Include yearly seasonality
            weekly_seasonality: Include weekly seasonality
            daily_seasonality: Include daily seasonality
            seasonality_mode: 'additive' or 'multiplicative'
            changepoint_prior_scale: Flexibility of trend
            holidays: DataFrame of holidays
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is required. Install with: pip install prophet")
        
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.holidays = holidays
        
        # Fitted attributes
        self.model_ = None
        self.training_df_ = None
    
    def fit(
        self,
        df: pd.DataFrame,
        datetime_col: str = None,
        target_col: str = None
    ) -> 'ProphetForecaster':
        """
        Fit Prophet model
        
        Args:
            df: DataFrame with datetime and target columns
            datetime_col: Name of datetime column (auto-detected if None)
            target_col: Name of target column (auto-detected if None)
            
        Returns:
            self
        """
        # Prepare data for Prophet (requires 'ds' and 'y' columns)
        prophet_df = self._prepare_data(df, datetime_col, target_col)
        self.training_df_ = prophet_df.copy()
        
        # Create and fit model
        self.model_ = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=self.changepoint_prior_scale,
            holidays=self.holidays
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_.fit(prophet_df)
        
        return self
    
    def predict(
        self,
        periods: int = 30,
        freq: str = 'D',
        include_history: bool = False
    ) -> pd.DataFrame:
        """
        Make predictions
        
        Args:
            periods: Number of periods to forecast
            freq: Frequency ('D', 'H', 'W', 'M')
            include_history: Include historical predictions
            
        Returns:
            DataFrame with predictions
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Create future dataframe
        future = self.model_.make_future_dataframe(periods=periods, freq=freq)
        
        # Make predictions
        forecast = self.model_.predict(future)
        
        if not include_history:
            forecast = forecast.tail(periods)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    def predict_at_dates(self, dates: pd.DataFrame) -> pd.DataFrame:
        """
        Predict at specific dates
        
        Args:
            dates: DataFrame with 'ds' column
            
        Returns:
            Predictions DataFrame
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        forecast = self.model_.predict(dates)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    def _prepare_data(
        self,
        df: pd.DataFrame,
        datetime_col: str = None,
        target_col: str = None
    ) -> pd.DataFrame:
        """
        Prepare data for Prophet (needs 'ds' and 'y' columns)
        """
        df = df.copy()
        
        # Detect datetime column
        if datetime_col is None:
            for col in df.columns:
                if df[col].dtype == 'datetime64[ns]' or 'date' in col.lower():
                    datetime_col = col
                    break
        
        # Detect target column
        if target_col is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            target_col = numeric_cols[0] if len(numeric_cols) > 0 else None
        
        # Create Prophet DataFrame
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(df[datetime_col]),
            'y': df[target_col]
        })
        
        return prophet_df
    
    def get_components(self) -> pd.DataFrame:
        """Get model components (trend, seasonality)"""
        if self.model_ is None:
            return pd.DataFrame()
        
        future = self.model_.make_future_dataframe(periods=0)
        forecast = self.model_.predict(future)
        
        components = ['ds', 'trend']
        
        if 'yearly' in forecast.columns:
            components.append('yearly')
        if 'weekly' in forecast.columns:
            components.append('weekly')
        if 'daily' in forecast.columns:
            components.append('daily')
        
        return forecast[components]
    
    def add_regressor(self, df: pd.DataFrame, name: str):
        """Add external regressor"""
        if self.model_ is None:
            self.model_ = Prophet(
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality,
                seasonality_mode=self.seasonality_mode
            )
        
        self.model_.add_regressor(name)
    
    def cross_validate(
        self,
        initial: str = '365 days',
        period: str = '30 days',
        horizon: str = '30 days'
    ) -> pd.DataFrame:
        """
        Perform cross-validation
        
        Args:
            initial: Initial training period
            period: Period between cutoff dates
            horizon: Forecast horizon
            
        Returns:
            Cross-validation results
        """
        from prophet.diagnostics import cross_validation, performance_metrics
        
        cv_results = cross_validation(
            self.model_,
            initial=initial,
            period=period,
            horizon=horizon
        )
        
        return cv_results
    
    def get_metrics(
        self,
        initial: str = '365 days',
        period: str = '30 days',
        horizon: str = '30 days'
    ) -> Dict[str, float]:
        """Get performance metrics via cross-validation"""
        from prophet.diagnostics import cross_validation, performance_metrics
        
        cv_results = cross_validation(
            self.model_,
            initial=initial,
            period=period,
            horizon=horizon
        )
        
        metrics = performance_metrics(cv_results)
        
        return {
            'mape': float(metrics['mape'].mean()),
            'rmse': float(metrics['rmse'].mean()),
            'mae': float(metrics['mae'].mean())
        }
    
    def save(self, path: str):
        """Save model"""
        joblib.dump(self.model_, path)
    
    @classmethod
    def load(cls, path: str) -> 'ProphetForecaster':
        """Load model"""
        forecaster = cls()
        forecaster.model_ = joblib.load(path)
        return forecaster
