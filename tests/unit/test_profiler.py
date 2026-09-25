"""
Unit Tests for Data Profiler
"""
import pytest
import pandas as pd
import numpy as np
from app.services.data_profiler import DataProfiler


class TestDataProfiler:
    """Test suite for DataProfiler service"""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing"""
        return pd.DataFrame({
            'id': range(1, 101),
            'name': [f'Person {i}' for i in range(1, 101)],
            'age': np.random.randint(18, 80, 100),
            'salary': np.random.uniform(30000, 150000, 100),
            'department': np.random.choice(['Sales', 'Engineering', 'Marketing'], 100),
            'is_active': np.random.choice([True, False], 100),
            'hire_date': pd.date_range('2020-01-01', periods=100, freq='D')
        })
    
    @pytest.fixture
    def profiler(self, sample_df):
        """Create profiler instance"""
        return DataProfiler(sample_df)
    
    def test_profile_dataset(self, profiler):
        """Test basic profiling"""
        profile = profiler.profile_dataset()
        
        assert 'basic_info' in profile
        assert 'column_profiles' in profile
        assert 'missing_values' in profile
        assert 'data_quality_score' in profile
    
    def test_basic_info(self, profiler):
        """Test basic info extraction"""
        profile = profiler.profile_dataset()
        basic_info = profile['basic_info']
        
        assert basic_info['num_rows'] == 100
        assert basic_info['num_columns'] == 7
    
    def test_column_profiles(self, profiler):
        """Test column profiling"""
        profile = profiler.profile_dataset()
        column_profiles = profile['column_profiles']
        
        # Test numeric column
        assert 'age' in column_profiles
        assert column_profiles['age']['dtype'] in ['int64', 'int32']
        assert 'mean' in column_profiles['age']
        assert 'min' in column_profiles['age']
        assert 'max' in column_profiles['age']
        
        # Test categorical column
        assert 'department' in column_profiles
        assert column_profiles['department']['dtype'] == 'object'
    
    def test_detect_data_type_tabular(self, sample_df):
        """Test tabular data type detection"""
        profiler = DataProfiler(sample_df)
        data_type = profiler.detect_data_type()
        
        assert data_type == 'tabular'
    
    def test_detect_data_type_timeseries(self):
        """Test time-series data type detection"""
        ts_df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=365, freq='D'),
            'value': np.random.randn(365).cumsum()
        })
        
        profiler = DataProfiler(ts_df)
        data_type = profiler.detect_data_type()
        
        assert data_type == 'timeseries'
    
    def test_data_quality_score(self, profiler):
        """Test data quality score calculation"""
        profile = profiler.profile_dataset()
        
        assert 'data_quality_score' in profile
        assert 0 <= profile['data_quality_score'] <= 100
    
    def test_missing_values_handling(self):
        """Test handling of missing values"""
        df_with_missing = pd.DataFrame({
            'a': [1, 2, None, 4, 5],
            'b': ['x', None, 'y', 'z', None]
        })
        
        profiler = DataProfiler(df_with_missing)
        profile = profiler.profile_dataset()
        
        assert profile['missing_values']['total_missing'] > 0


class TestProblemDetector:
    """Test suite for ProblemDetector"""
    
    def test_binary_classification_detection(self):
        """Test detection of binary classification problem"""
        from app.services.problem_detector import ProblemDetector
        
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.choice(['Yes', 'No'], 100)
        })
        
        detector = ProblemDetector(df, 'target')
        result = detector.detect()
        
        assert result['problem_type'] == 'binary_classification'
    
    def test_regression_detection(self):
        """Test detection of regression problem"""
        from app.services.problem_detector import ProblemDetector
        
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.uniform(0, 100, 100)
        })
        
        detector = ProblemDetector(df, 'target')
        result = detector.detect()
        
        assert result['problem_type'] == 'regression'
    
    def test_multiclass_classification_detection(self):
        """Test detection of multiclass classification"""
        from app.services.problem_detector import ProblemDetector
        
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'target': np.random.choice(['A', 'B', 'C', 'D'], 100)
        })
        
        detector = ProblemDetector(df, 'target')
        result = detector.detect()
        
        assert result['problem_type'] == 'multiclass_classification'

    def test_numeric_class_codes_detection(self):
        """Test numeric class codes are not mistaken for regression"""
        from app.services.problem_detector import ProblemDetector

        df = pd.DataFrame({
            'feature1': np.random.randn(200),
            'target': np.random.choice([0, 1, 2], 200)
        })

        detector = ProblemDetector(df, 'target')
        result = detector.detect()

        assert result['problem_type'] == 'multiclass_classification'

    def test_timeseries_requires_forecast_goal(self):
        """Test date columns alone do not force a timeseries problem"""
        from app.services.problem_detector import ProblemDetector

        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100, freq='D'),
            'feature1': np.random.randn(100),
            'target': np.random.uniform(0, 100, 100)
        })

        detector = ProblemDetector(df, 'target')
        result = detector.detect()

        assert result['problem_type'] == 'regression'

    def test_timeseries_forecast_goal_detection(self):
        """Test date columns plus forecasting goal select timeseries"""
        from app.services.problem_detector import ProblemDetector

        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100, freq='D'),
            'target': np.random.uniform(0, 100, 100)
        })

        detector = ProblemDetector(df, 'target', goal_description='forecast next month demand')
        result = detector.detect()

        assert result['problem_type'] == 'timeseries'
    
    def test_clustering_no_target(self):
        """Test clustering detection when no target specified"""
        from app.services.problem_detector import ProblemDetector
        
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        
        detector = ProblemDetector(df, target_column=None)
        result = detector.detect()
        
        assert result['problem_type'] == 'clustering'
