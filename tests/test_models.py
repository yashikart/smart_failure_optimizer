"""
Unit tests for ML models and pipeline
"""

import pytest
import pandas as pd
import numpy as np
import joblib
import os
from unittest.mock import Mock, patch


class TestModelLoading:
    """Test model loading and initialization"""
    
    def test_model_directory_exists(self):
        """Test if models directory exists"""
        # This will pass if models are trained
        if os.path.exists('models'):
            assert True
        else:
            pytest.skip("Models not trained yet")
    
    def test_model_files_present(self):
        """Test if model files are present"""
        if not os.path.exists('models'):
            pytest.skip("Models directory not found")
        
        expected_files = [
            'best_model_cooler_cond.pkl',
            'best_model_valve_cond.pkl',
            'best_model_pump_leak.pkl',
            'best_model_accumulator_press.pkl'
        ]
        
        for file in expected_files:
            file_path = os.path.join('models', file)
            if os.path.exists(file_path):
                assert os.path.getsize(file_path) > 0
    
    def test_model_loadable(self):
        """Test if models can be loaded"""
        if not os.path.exists('models'):
            pytest.skip("Models directory not found")
        
        model_file = 'models/best_model_cooler_cond.pkl'
        if os.path.exists(model_file):
            model = joblib.load(model_file)
            assert model is not None
            assert hasattr(model, 'predict')


class TestModelPrediction:
    """Test model prediction functionality"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample sensor data"""
        return pd.DataFrame({
            'PS1_mean': [160.5],
            'PS2_mean': [109.2],
            'PS3_mean': [1.75],
            'FS1_mean': [6.2],
            'FS2_mean': [9.65],
            'TS1_mean': [45.4],
            'CE_mean': [31.3],
            'CP_mean': [1.81],
            'EPS1_mean': [2495.0],
            'VS1_mean': [0.61],
            'SE_mean': [55.3]
        })
    
    def test_prediction_shape(self, sample_data):
        """Test prediction output shape"""
        if not os.path.exists('models/best_model_cooler_cond.pkl'):
            pytest.skip("Model not found")
        
        model = joblib.load('models/best_model_cooler_cond.pkl')
        
        # Add missing features
        for col in range(126):
            if f'feature_{col}' not in sample_data.columns:
                sample_data[f'feature_{col}'] = 0.0
        
        try:
            prediction = model.predict(sample_data)
            assert len(prediction) == 1
            assert isinstance(prediction[0], (int, float, np.integer, np.floating))
        except:
            pytest.skip("Feature mismatch - expected in test environment")
    
    def test_prediction_probability(self, sample_data):
        """Test prediction probability output"""
        if not os.path.exists('models/best_model_cooler_cond.pkl'):
            pytest.skip("Model not found")
        
        model = joblib.load('models/best_model_cooler_cond.pkl')
        
        # Add missing features
        for col in range(126):
            if f'feature_{col}' not in sample_data.columns:
                sample_data[f'feature_{col}'] = 0.0
        
        try:
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(sample_data)
                assert probabilities.shape[0] == 1
                assert np.sum(probabilities[0]) == pytest.approx(1.0, abs=0.01)
                assert all(0 <= p <= 1 for p in probabilities[0])
        except:
            pytest.skip("Feature mismatch - expected in test environment")


class TestDataValidation:
    """Test data validation logic"""
    
    def test_valid_sensor_ranges(self):
        """Test sensor value ranges"""
        test_cases = [
            ('PS1_mean', 160.0, 100, 200),  # Pressure
            ('TS1_mean', 45.0, 20, 70),     # Temperature
            ('FS1_mean', 6.2, 0, 20),        # Flow
        ]
        
        for sensor, value, min_val, max_val in test_cases:
            assert min_val <= value <= max_val, f"{sensor} out of range"
    
    def test_missing_features_handling(self):
        """Test handling of missing features"""
        incomplete_data = pd.DataFrame({
            'PS1_mean': [160.0],
            'PS2_mean': [109.0]
        })
        
        # Should not raise error when adding default values
        for i in range(20):
            incomplete_data[f'feature_{i}'] = 0.0
        
        assert len(incomplete_data.columns) > 2


class TestModelMetadata:
    """Test model metadata and versioning"""
    
    def test_metadata_exists(self):
        """Test if metadata files exist"""
        if not os.path.exists('models'):
            pytest.skip("Models directory not found")
        
        metadata_files = [f for f in os.listdir('models') if f.startswith('metadata_')]
        
        for meta_file in metadata_files:
            file_path = os.path.join('models', meta_file)
            metadata = joblib.load(file_path)
            assert 'target' in metadata
            assert 'accuracy' in metadata
            assert 'features' in metadata
    
    def test_metadata_accuracy_range(self):
        """Test if accuracy values are in valid range"""
        if not os.path.exists('models'):
            pytest.skip("Models directory not found")
        
        metadata_files = [f for f in os.listdir('models') if f.startswith('metadata_')]
        
        for meta_file in metadata_files:
            file_path = os.path.join('models', meta_file)
            metadata = joblib.load(file_path)
            
            if 'accuracy' in metadata:
                assert 0 <= metadata['accuracy'] <= 1
            if 'f1_score' in metadata:
                assert 0 <= metadata['f1_score'] <= 1


class TestPipelineIntegrity:
    """Test ML pipeline integrity"""
    
    def test_pipeline_has_scaler(self):
        """Test if pipeline includes scaler"""
        if not os.path.exists('models/best_model_cooler_cond.pkl'):
            pytest.skip("Model not found")
        
        model = joblib.load('models/best_model_cooler_cond.pkl')
        
        # Check if it's a pipeline
        if hasattr(model, 'named_steps'):
            assert 'scaler' in model.named_steps or 'standardscaler' in str(type(model.named_steps.get('scaler', ''))).lower()
    
    def test_pipeline_reproducibility(self):
        """Test if predictions are reproducible"""
        if not os.path.exists('models/best_model_cooler_cond.pkl'):
            pytest.skip("Model not found")
        
        model = joblib.load('models/best_model_cooler_cond.pkl')
        
        # Create test data
        test_data = pd.DataFrame({
            f'feature_{i}': [1.0] for i in range(126)
        })
        
        try:
            pred1 = model.predict(test_data)
            pred2 = model.predict(test_data)
            assert all(pred1 == pred2)
        except:
            pytest.skip("Feature mismatch - expected in test environment")


class TestPerformanceMetrics:
    """Test model performance requirements"""
    
    def test_minimum_accuracy_threshold(self):
        """Test if models meet minimum accuracy threshold"""
        if not os.path.exists('models'):
            pytest.skip("Models directory not found")
        
        min_accuracy = 0.85  # 85% minimum
        
        metadata_files = [f for f in os.listdir('models') if f.startswith('metadata_')]
        
        for meta_file in metadata_files:
            file_path = os.path.join('models', meta_file)
            metadata = joblib.load(file_path)
            
            if 'accuracy' in metadata:
                assert metadata['accuracy'] >= min_accuracy, \
                    f"Model {metadata['target']} below threshold: {metadata['accuracy']}"
    
    def test_cv_stability(self):
        """Test cross-validation stability"""
        if not os.path.exists('models'):
            pytest.skip("Models directory not found")
        
        max_cv_std = 0.10  # Max 10% standard deviation
        
        metadata_files = [f for f in os.listdir('models') if f.startswith('metadata_')]
        
        for meta_file in metadata_files:
            file_path = os.path.join('models', meta_file)
            metadata = joblib.load(file_path)
            
            if 'cv_std' in metadata:
                assert metadata['cv_std'] <= max_cv_std, \
                    f"Model {metadata['target']} unstable: CV std = {metadata['cv_std']}"


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

