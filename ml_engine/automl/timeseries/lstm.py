"""
LSTM Time-Series Forecasting
Deep learning approach using LSTM networks
GPU: Automatically uses GPU if available, falls back to CPU.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import warnings

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# ── GPU auto-configuration ──────────────────────────────────────
try:
    from ml_engine.utils.gpu_utils import configure_tensorflow_gpu, log_gpu_info
    if TENSORFLOW_AVAILABLE:
        configure_tensorflow_gpu()
except ImportError:
    pass  # utils not available (e.g. standalone use)


class LSTMForecaster:
    """
    LSTM model for time-series forecasting
    Good for complex, non-linear patterns
    """
    
    def __init__(
        self,
        lookback: int = 30,
        forecast_horizon: int = 1,
        lstm_units: List[int] = [64, 32],
        dropout: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        bidirectional: bool = False
    ):
        """
        Initialize LSTM forecaster
        
        Args:
            lookback: Number of past time steps to use
            forecast_horizon: Number of steps to forecast
            lstm_units: List of units for each LSTM layer
            dropout: Dropout rate
            epochs: Training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            bidirectional: Use bidirectional LSTM
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required. Install with: pip install tensorflow")
        
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.bidirectional = bidirectional
        
        # Fitted attributes
        self.model_ = None
        self.scaler_mean_ = None
        self.scaler_std_ = None
        self.history_ = None
        self.n_features_ = 1
    
    def fit(
        self,
        data: np.ndarray,
        validation_split: float = 0.2,
        verbose: int = 0
    ) -> 'LSTMForecaster':
        """
        Fit LSTM model
        
        Args:
            data: 1D or 2D array of time series values
            validation_split: Fraction for validation
            verbose: Verbosity level
            
        Returns:
            self
        """
        # Log GPU status at training start
        try:
            log_gpu_info()
        except Exception:
            pass

        # Ensure 2D
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        self.n_features_ = data.shape[1]
        
        # Scale data
        self.scaler_mean_ = np.mean(data, axis=0)
        self.scaler_std_ = np.std(data, axis=0) + 1e-8
        data_scaled = (data - self.scaler_mean_) / self.scaler_std_
        
        # Create sequences
        X, y = self._create_sequences(data_scaled)
        
        # Build model
        self.model_ = self._build_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Fit
        self.history_ = self.model_.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self
    
    def predict(
        self,
        last_sequence: np.ndarray,
        steps: int = None
    ) -> np.ndarray:
        """
        Make predictions
        
        Args:
            last_sequence: Last 'lookback' values
            steps: Number of steps to predict (default: forecast_horizon)
            
        Returns:
            Predicted values
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        steps = steps or self.forecast_horizon
        
        # Ensure shape
        if len(last_sequence.shape) == 1:
            last_sequence = last_sequence.reshape(-1, 1)
        
        # Scale
        sequence_scaled = (last_sequence - self.scaler_mean_) / self.scaler_std_
        
        # Predict iteratively
        predictions = []
        current_sequence = sequence_scaled[-self.lookback:].copy()
        
        for _ in range(steps):
            # Reshape for LSTM: (1, lookback, features)
            X = current_sequence.reshape(1, self.lookback, self.n_features_)
            
            # Predict next step
            pred = self.model_.predict(X, verbose=0)
            predictions.append(pred[0])
            
            # Update sequence
            current_sequence = np.roll(current_sequence, -1, axis=0)
            if self.forecast_horizon == 1:
                current_sequence[-1] = pred[0]
            else:
                current_sequence[-1] = pred[0, 0]
        
        predictions = np.array(predictions)
        
        # Inverse scale
        if len(predictions.shape) == 1:
            predictions = predictions.reshape(-1, 1)
        
        predictions_unscaled = predictions * self.scaler_std_[:1] + self.scaler_mean_[:1]
        
        return predictions_unscaled.flatten()
    
    def _create_sequences(
        self,
        data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training
        
        Args:
            data: Scaled data array
            
        Returns:
            X, y arrays
        """
        X, y = [], []
        
        for i in range(len(data) - self.lookback - self.forecast_horizon + 1):
            X.append(data[i:i + self.lookback])
            
            if self.forecast_horizon == 1:
                y.append(data[i + self.lookback, 0])
            else:
                y.append(data[i + self.lookback:i + self.lookback + self.forecast_horizon, 0])
        
        return np.array(X), np.array(y)
    
    def _build_model(self) -> 'keras.Model':
        """Build LSTM model"""
        model = Sequential()
        
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            
            if i == 0:
                # First layer with input shape
                if self.bidirectional:
                    model.add(Bidirectional(
                        LSTM(units, return_sequences=return_sequences),
                        input_shape=(self.lookback, self.n_features_)
                    ))
                else:
                    model.add(LSTM(
                        units,
                        return_sequences=return_sequences,
                        input_shape=(self.lookback, self.n_features_)
                    ))
            else:
                if self.bidirectional:
                    model.add(Bidirectional(LSTM(units, return_sequences=return_sequences)))
                else:
                    model.add(LSTM(units, return_sequences=return_sequences))
            
            model.add(Dropout(self.dropout))
        
        # Output layer
        model.add(Dense(self.forecast_horizon))
        
        # Compile
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def evaluate(
        self,
        data: np.ndarray,
        test_size: int = None
    ) -> Dict[str, float]:
        """
        Evaluate model on test data
        
        Args:
            data: Full data array
            test_size: Number of test samples
            
        Returns:
            Dictionary of metrics
        """
        if test_size is None:
            test_size = len(data) // 5
        
        # Get test predictions
        predictions = []
        actuals = []
        
        for i in range(test_size):
            idx = len(data) - test_size - self.lookback + i
            sequence = data[idx:idx + self.lookback]
            pred = self.predict(sequence, steps=1)
            
            predictions.append(pred[0])
            actuals.append(data[idx + self.lookback])
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - actuals))
        mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-8))) * 100
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape)
        }
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """Get training history"""
        if self.history_ is None:
            return {}
        
        return {
            'loss': self.history_.history['loss'],
            'val_loss': self.history_.history.get('val_loss', []),
            'mae': self.history_.history.get('mae', []),
            'val_mae': self.history_.history.get('val_mae', [])
        }
    
    def save(self, path: str):
        """Save model"""
        if self.model_ is not None:
            self.model_.save(path)
            
            # Save scalers separately
            import json
            scaler_path = path.replace('.h5', '_scaler.json').replace('.keras', '_scaler.json')
            with open(scaler_path, 'w') as f:
                json.dump({
                    'mean': self.scaler_mean_.tolist(),
                    'std': self.scaler_std_.tolist(),
                    'lookback': self.lookback,
                    'forecast_horizon': self.forecast_horizon,
                    'n_features': self.n_features_
                }, f)
    
    @classmethod
    def load(cls, path: str) -> 'LSTMForecaster':
        """Load model"""
        import json
        
        forecaster = cls()
        forecaster.model_ = keras.models.load_model(path)
        
        # Load scalers
        scaler_path = path.replace('.h5', '_scaler.json').replace('.keras', '_scaler.json')
        with open(scaler_path, 'r') as f:
            scaler_data = json.load(f)
        
        forecaster.scaler_mean_ = np.array(scaler_data['mean'])
        forecaster.scaler_std_ = np.array(scaler_data['std'])
        forecaster.lookback = scaler_data['lookback']
        forecaster.forecast_horizon = scaler_data['forecast_horizon']
        forecaster.n_features_ = scaler_data['n_features']
        
        return forecaster
