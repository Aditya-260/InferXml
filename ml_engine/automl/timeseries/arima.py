"""
ARIMA Time-Series Forecasting
Statistical approach for time-series prediction
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import warnings

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

import joblib


class ARIMAForecaster:
    """
    ARIMA model for time-series forecasting
    Automatically determines optimal (p, d, q) parameters
    """
    
    def __init__(
        self,
        order: Tuple[int, int, int] = None,
        seasonal_order: Tuple[int, int, int, int] = None,
        auto_order: bool = True,
        max_p: int = 5,
        max_d: int = 2,
        max_q: int = 5
    ):
        """
        Initialize ARIMA forecaster
        
        Args:
            order: (p, d, q) - AR, differencing, MA orders
            seasonal_order: (P, D, Q, s) - Seasonal parameters
            auto_order: Automatically find optimal order
            max_p, max_d, max_q: Max values for auto order search
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for ARIMA. Install with: pip install statsmodels")
        
        self.order = order
        self.seasonal_order = seasonal_order
        self.auto_order = auto_order
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        
        # Fitted attributes
        self.model_ = None
        self.fitted_model_ = None
        self.best_order_ = None
    
    def fit(
        self,
        y: pd.Series,
        exog: pd.DataFrame = None
    ) -> 'ARIMAForecaster':
        """
        Fit ARIMA model
        
        Args:
            y: Time series values
            exog: Exogenous variables (optional)
            
        Returns:
            self
        """
        # Determine order
        if self.auto_order and self.order is None:
            self.best_order_ = self._find_best_order(y)
        else:
            self.best_order_ = self.order or (1, 1, 1)
        
        # Fit model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if self.seasonal_order:
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                self.model_ = SARIMAX(
                    y,
                    exog=exog,
                    order=self.best_order_,
                    seasonal_order=self.seasonal_order
                )
            else:
                self.model_ = ARIMA(
                    y,
                    exog=exog,
                    order=self.best_order_
                )
            
            self.fitted_model_ = self.model_.fit()
        
        return self
    
    def predict(
        self,
        steps: int = 1,
        exog: pd.DataFrame = None
    ) -> pd.Series:
        """
        Make predictions
        
        Args:
            steps: Number of steps to forecast
            exog: Exogenous variables for forecast period
            
        Returns:
            Forecasted values
        """
        if self.fitted_model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        forecast = self.fitted_model_.forecast(steps=steps, exog=exog)
        return forecast
    
    def get_forecast_with_intervals(
        self,
        steps: int = 1,
        alpha: float = 0.05
    ) -> Dict[str, np.ndarray]:
        """
        Get forecast with confidence intervals
        
        Args:
            steps: Number of steps to forecast
            alpha: Significance level (0.05 = 95% CI)
            
        Returns:
            Dict with 'mean', 'lower', 'upper' keys
        """
        forecast = self.fitted_model_.get_forecast(steps=steps)
        conf_int = forecast.conf_int(alpha=alpha)
        
        return {
            'mean': forecast.predicted_mean.values,
            'lower': conf_int.iloc[:, 0].values,
            'upper': conf_int.iloc[:, 1].values
        }
    
    def _find_best_order(self, y: pd.Series) -> Tuple[int, int, int]:
        """
        Find optimal ARIMA order using grid search
        
        Args:
            y: Time series
            
        Returns:
            Best (p, d, q) order
        """
        # Determine d (differencing order) using ADF test
        d = self._find_d(y)
        
        # Grid search for p and q
        best_aic = float('inf')
        best_order = (1, d, 1)
        
        for p in range(self.max_p + 1):
            for q in range(self.max_q + 1):
                if p == 0 and q == 0:
                    continue
                
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = ARIMA(y, order=(p, d, q))
                        fitted = model.fit()
                        
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                except:
                    continue
        
        return best_order
    
    def _find_d(self, y: pd.Series) -> int:
        """
        Find differencing order using ADF test
        
        Args:
            y: Time series
            
        Returns:
            Optimal differencing order
        """
        for d in range(self.max_d + 1):
            if d == 0:
                test_series = y
            else:
                test_series = y.diff(d).dropna()
            
            result = adfuller(test_series, autolag='AIC')
            p_value = result[1]
            
            if p_value < 0.05:  # Stationary
                return d
        
        return self.max_d
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get model diagnostics"""
        if self.fitted_model_ is None:
            return {}
        
        return {
            'aic': self.fitted_model_.aic,
            'bic': self.fitted_model_.bic,
            'order': self.best_order_,
            'seasonal_order': self.seasonal_order,
            'n_observations': self.fitted_model_.nobs
        }
    
    def save(self, path: str):
        """Save model"""
        joblib.dump({
            'fitted_model': self.fitted_model_,
            'best_order': self.best_order_,
            'seasonal_order': self.seasonal_order
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'ARIMAForecaster':
        """Load model"""
        data = joblib.load(path)
        forecaster = cls()
        forecaster.fitted_model_ = data['fitted_model']
        forecaster.best_order_ = data['best_order']
        forecaster.seasonal_order = data['seasonal_order']
        return forecaster
