import numpy as np
import pandas as pd
import pytest
from ml_engine.preprocessing.timeseries_preprocessor import TimeSeriesPreprocessor

def test_cyclical_encoding():
    """Test that sine and cosine features are generated correctly."""
    dates = pd.date_range("2023-01-01", periods=24, freq="H")
    df = pd.DataFrame({"target": np.random.rand(24)}, index=dates)
    
    preprocessor = TimeSeriesPreprocessor(scale=False)
    processed_df = preprocessor.create_time_features(df, encode_cyclical=True)
    
    # Must contain Sin/Cos columns
    assert "hour_sin" in processed_df.columns
    assert "hour_cos" in processed_df.columns
    
    # Values should be bounded [-1, 1]
    assert processed_df["hour_sin"].max() <= 1.0
    assert processed_df["hour_sin"].min() >= -1.0

def test_outlier_clipping():
    """Test that outlier values are clipped accurately."""
    dates = pd.date_range("2023-01-01", periods=1000, freq="D")
    data = np.random.normal(0, 1, 1000)
    
    # Inject massive outliers
    data[0] = 99999.0
    data[-1] = -99999.0
    
    df = pd.DataFrame({"value": data}, index=dates)
    
    # First, test without clipping
    preprocessor_no_clip = TimeSeriesPreprocessor(scale=False)
    preprocessor_no_clip.feature_columns_ = ["value"]
    
    # manually add date column so index parsing won't complain or break
    df["date"] = df.index
    preprocessor_no_clip.fit(df)
    transformed_no_clip = preprocessor_no_clip.transform(df)
    
    # Then test with clipping
    preprocessor_clip = TimeSeriesPreprocessor(scale=False, outlier_clip=True)
    preprocessor_clip.feature_columns_ = ["value"]
    preprocessor_clip.fit(df)
    transformed_clip = preprocessor_clip.transform(df)
    
    assert transformed_no_clip["value"].max() > 90000
    assert transformed_clip["value"].max() < 1000 # Clipped to 99th percentile

def test_robust_scaling():
    """Test that robust scaler is used when flagged and outputs different range than standard scaler."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "feature1": np.random.normal(10, 2, 100)
    })
    
    # Add an outlier to heavily affect StandardScale mean, but not RobustScale median
    df.loc[0, "feature1"] = 1000.0
    
    preprocessor_robust = TimeSeriesPreprocessor(
        datetime_column="date",
        scale=True,
        use_robust_scaler=True
    )
    
    preprocessor_standard = TimeSeriesPreprocessor(
        datetime_column="date", 
        scale=True,
        use_robust_scaler=False
    )
    
    robust_df = preprocessor_robust.fit_transform(df)
    standard_df = preprocessor_standard.fit_transform(df)
    
    # The output shapes and features should be the same
    assert "feature1" in robust_df.columns
    assert "feature1" in standard_df.columns
    
    # But values should differ because RobustScaler uses IQR, StandardScaler uses Mean/Var
    assert not np.allclose(robust_df["feature1"].values, standard_df["feature1"].values)
