# InferX-ML Data Preprocessing Guide

This document outlines the advanced, industrial best practices currently implemented across the `ml_engine/preprocessing` pipelines. The preprocessing engine is split into three core modules: Tabular, Image, and Time-Series data.

---

## 1. Tabular Data Preprocessing (`TabularPreprocessor`)

Tabular data often contains messy, missing, or highly skewed values. The tabular pipeline handles these automatically:

- **Missing Target Handling**: Rows with missing target values are explicitly dropped before training to prevent model confusion.
- **Missing Value Indicators**: When imputing missing features, `add_indicator=True` is used. This creates a boolean column indicating *where* values were originally missing, as the sheer fact that data is missing can reflect an important pattern.
- **High Missing Ratio Filter**: Automatically drops any feature columns where the missing value ratio exceeds 60%, preventing the model from learning on mostly imputed noise.
- **Constant Feature Removal**: Uses `VarianceThreshold` to completely remove features with zero variance (features where every row has the exact same value).
- **Robust Scaling**: Uses `RobustScaler` (median and IQR) instead of traditional `StandardScaler` (mean and variance) to prevent extreme outlier values from skewing the normalized distributions.

---

## 2. Image Data Preprocessing (`ImagePreprocessor`)

Image models, particularly deep neural networks and transfer learning models, are very prone to overfitting and pipeline crashes.

- **Robust Corruption Handling**: Users often accidentally upload directories containing hidden system files (like `.DS_Store` or `thumbs.db`) or corrupted JPEGs. The image loader wraps parsing in a robust `try...except` block to log warnings and silently skip bad files rather than crashing an entire distributed training job.
- **Advanced Augmentations**: Instead of simple flips, the pipeline uses randomized $\pm15^\circ$ rotations and random zooms ($0.9\times$ to $1.1\times$). This actively prevents the network from memorizing exact pixel placements, vastly improving real-world generalization. 
- **Balanced Class Weighting**: Automatically computes a dynamic weight dictionary if the dataset is heavily imbalanced (e.g., 500 "Normal" images vs. 10 "Defect" images). This instructs the model's loss function to penalize mistakes on the minority class much more heavily.

---

## 3. Time-Series Data Preprocessing (`TimeSeriesPreprocessor`)

Time-series forecasting often involves noisy sensor/IoT data and repeating seasonal or temporal trends.

- **Cyclical Feature Encoding**: Machine learning models do not naturally understand that time is a circle (e.g., that Hour 23 is right next to Hour 0). The pipeline automatically extracts features like `hour`, `month`, and `day_of_week`, and encodes them into mathematical `sine` and `cosine` waves to perfectly preserve their circular relationships.
- **Outlier Clipping (Winsorization)**: Time-series sensors are notoriously prone to transient spikes (e.g., a sensor briefly reading $99,999$ due to interference). The pipeline caps all numeric values to their $1^{st}$ and $99^{th}$ percentiles globally. This isolates extreme spikes and prevents them from destroying rolling averages and normalization mathematics.
- **Robust Scaling (Time-Series)**: Just like Tabular data, Time-Series implements togglable `RobustScaler` logic to ensure large spikes do not pull the mathematical mean of an entire sensor's history up or down.

---

*For detailed test coverage of these mechanisms, see the respective unit tests in the `tests/unit/` directory.*
