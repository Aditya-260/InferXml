"""
Unit Tests for AutoML Modules
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression


class TestTabularClassifier:
    """Test suite for TabularClassifier"""
    
    @pytest.fixture
    def classification_data(self):
        """Generate classification dataset"""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_classes=2,
            random_state=42
        )
        return X, y
    
    def test_classifier_training(self, classification_data):
        """Test classifier training"""
        from ml_engine.automl.tabular.classifier import TabularClassifier
        
        X, y = classification_data
        classifier = TabularClassifier()
        result = classifier.train(X, y)
        
        assert 'best_model' in result
        assert 'best_score' in result
        assert result['best_score'] > 0.5  # Should be better than random
    
    def test_classifier_prediction(self, classification_data):
        """Test classifier prediction"""
        from ml_engine.automl.tabular.classifier import TabularClassifier
        
        X, y = classification_data
        classifier = TabularClassifier()
        classifier.train(X, y)
        
        predictions = classifier.predict(X[:10])
        
        assert len(predictions) == 10
        assert all(p in [0, 1] for p in predictions)
    
    def test_classifier_proba(self, classification_data):
        """Test probability predictions"""
        from ml_engine.automl.tabular.classifier import TabularClassifier
        
        X, y = classification_data
        classifier = TabularClassifier()
        classifier.train(X, y)
        
        probas = classifier.predict_proba(X[:10])
        
        assert probas.shape == (10, 2)
        assert np.allclose(probas.sum(axis=1), 1.0)


class TestTabularRegressor:
    """Test suite for TabularRegressor"""
    
    @pytest.fixture
    def regression_data(self):
        """Generate regression dataset"""
        X, y = make_regression(
            n_samples=200,
            n_features=10,
            n_informative=5,
            noise=0.1,
            random_state=42
        )
        return X, y
    
    def test_regressor_training(self, regression_data):
        """Test regressor training"""
        from ml_engine.automl.tabular.regressor import TabularRegressor
        
        X, y = regression_data
        regressor = TabularRegressor()
        result = regressor.train(X, y)
        
        assert 'best_model' in result
        assert 'best_score' in result
    
    def test_regressor_prediction(self, regression_data):
        """Test regressor prediction"""
        from ml_engine.automl.tabular.regressor import TabularRegressor
        
        X, y = regression_data
        regressor = TabularRegressor()
        regressor.train(X, y)
        
        predictions = regressor.predict(X[:10])
        
        assert len(predictions) == 10
        assert all(isinstance(p, (int, float, np.floating)) for p in predictions)


class TestTabularClusterer:
    """Test suite for TabularClusterer"""
    
    @pytest.fixture
    def clustering_data(self):
        """Generate clustering dataset"""
        np.random.seed(42)
        n_samples = 150
        
        # Create 3 clusters
        cluster1 = np.random.randn(50, 2) + np.array([0, 0])
        cluster2 = np.random.randn(50, 2) + np.array([5, 5])
        cluster3 = np.random.randn(50, 2) + np.array([0, 5])
        
        X = np.vstack([cluster1, cluster2, cluster3])
        return X
    
    def test_clusterer_training(self, clustering_data):
        """Test clusterer training"""
        from ml_engine.automl.tabular.clusterer import TabularClusterer
        
        X = clustering_data
        clusterer = TabularClusterer()
        result = clusterer.train(X)
        
        assert 'best_model' in result
        assert 'n_clusters' in result
    
    def test_clusterer_prediction(self, clustering_data):
        """Test cluster assignment"""
        from ml_engine.automl.tabular.clusterer import TabularClusterer
        
        X = clustering_data
        clusterer = TabularClusterer()
        clusterer.train(X)
        
        labels = clusterer.predict(X[:10])
        
        assert len(labels) == 10


class TestPreprocessor:
    """Test suite for TabularPreprocessor"""
    
    @pytest.fixture
    def mixed_data(self):
        """Create mixed-type dataset"""
        return pd.DataFrame({
            'numeric1': np.random.randn(100),
            'numeric2': np.random.randint(0, 100, 100),
            'categorical': np.random.choice(['A', 'B', 'C'], 100),
            'target': np.random.choice([0, 1], 100)
        })
    
    def test_preprocessor_fit_transform(self, mixed_data):
        """Test fit_transform"""
        from ml_engine.preprocessing.tabular_preprocessor import TabularPreprocessor
        
        X = mixed_data.drop(columns=['target'])
        preprocessor = TabularPreprocessor()
        X_transformed = preprocessor.fit_transform(X)
        
        assert X_transformed is not None
        assert len(X_transformed) == len(X)
    
    def test_preprocessor_column_detection(self, mixed_data):
        """Test automatic column type detection"""
        from ml_engine.preprocessing.tabular_preprocessor import TabularPreprocessor
        
        X = mixed_data.drop(columns=['target'])
        preprocessor = TabularPreprocessor()
        preprocessor.fit(X)
        
        assert 'numeric1' in preprocessor.numeric_columns_
        assert 'categorical' in preprocessor.categorical_columns_
