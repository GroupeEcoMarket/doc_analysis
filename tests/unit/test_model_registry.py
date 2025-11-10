"""
Unit tests for ModelRegistry
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.models.registry import ModelRegistry, ModelType, ModelConfig
from src.utils.exceptions import ModelLoadingError


class TestModelRegistry:
    """Unit tests for ModelRegistry"""
    
    def test_registry_initialization(self):
        """Test that ModelRegistry initializes correctly"""
        registry = ModelRegistry()
        assert registry is not None
        assert len(registry._model_configs) == 2  # ORIENTATION and DETECTION
        assert ModelType.ORIENTATION in registry._model_configs
        assert ModelType.DETECTION in registry._model_configs
    
    def test_list_models(self):
        """Test listing all registered models"""
        registry = ModelRegistry()
        models = registry.list_models()
        
        assert isinstance(models, dict)
        assert 'orientation' in models
        assert 'detection' in models
        
        # Check model info structure
        orientation_info = models['orientation']
        assert 'type' in orientation_info
        assert 'name' in orientation_info
        assert 'source' in orientation_info
        assert 'identifier' in orientation_info
        assert 'loaded' in orientation_info
    
    def test_get_model_info(self):
        """Test getting information about a specific model"""
        registry = ModelRegistry()
        
        info = registry.get_model_info(ModelType.ORIENTATION)
        assert info['type'] == 'orientation'
        assert info['name'] == 'page-orientation'
        assert info['source'] == 'huggingface'
        assert info['identifier'] == 'Felix92/onnxtr-mobilenet-v3-small-page-orientation'
        assert info['loaded'] is False  # Not loaded yet
    
    def test_register_model(self):
        """Test registering a custom model configuration"""
        registry = ModelRegistry()
        
        custom_config = ModelConfig(
            model_type=ModelType.ORIENTATION,
            name="custom-orientation",
            source="huggingface",
            identifier="custom/repo-name",
            version="1.0.0"
        )
        
        registry.register_model(ModelType.ORIENTATION, custom_config)
        
        info = registry.get_model_info(ModelType.ORIENTATION)
        assert info['name'] == 'custom-orientation'
        assert info['identifier'] == 'custom/repo-name'
        assert info['version'] == '1.0.0'
    
    @patch('src.models.registry.from_hub')
    @patch('src.models.registry.page_orientation_predictor')
    @patch('src.models.registry.OrientationModelAdapter')
    def test_get_orientation_adapter_lazy_loading(self, mock_adapter_class, mock_predictor, mock_from_hub):
        """Test lazy loading of orientation adapter"""
        # Mock the model loading chain
        mock_raw_model = Mock()
        mock_predictor_instance = Mock()
        mock_adapter_instance = Mock()
        
        mock_from_hub.return_value = mock_raw_model
        mock_predictor.return_value = mock_predictor_instance
        mock_adapter_class.return_value = mock_adapter_instance
        
        registry = ModelRegistry()
        
        # First call should load the model
        adapter1 = registry.get_orientation_adapter()
        assert adapter1 == mock_adapter_instance
        mock_from_hub.assert_called_once()
        mock_predictor.assert_called_once_with(arch=mock_raw_model)
        mock_adapter_class.assert_called_once_with(mock_predictor_instance)
        
        # Second call should use cached model
        mock_from_hub.reset_mock()
        mock_predictor.reset_mock()
        mock_adapter_class.reset_mock()
        
        adapter2 = registry.get_orientation_adapter()
        assert adapter2 == mock_adapter_instance
        # Should not reload
        mock_from_hub.assert_not_called()
        mock_predictor.assert_not_called()
        mock_adapter_class.assert_not_called()
    
    @patch('src.models.registry.detection')
    def test_get_detection_model_lazy_loading(self, mock_detection):
        """Test lazy loading of detection model"""
        mock_model = Mock()
        mock_detection.detection_predictor.return_value = mock_model
        
        registry = ModelRegistry()
        
        # First call should load the model
        model1 = registry.get_detection_model()
        assert model1 == mock_model
        mock_detection.detection_predictor.assert_called_once_with(
            arch="db_resnet50",
            pretrained=True
        )
        
        # Second call should use cached model
        mock_detection.detection_predictor.reset_mock()
        model2 = registry.get_detection_model()
        assert model2 == mock_model
        # Should not reload
        mock_detection.detection_predictor.assert_not_called()
    
    @patch('src.models.registry.from_hub')
    def test_get_orientation_adapter_error_handling(self, mock_from_hub):
        """Test error handling when loading orientation model fails"""
        mock_from_hub.side_effect = Exception("Network error")
        
        registry = ModelRegistry()
        
        with pytest.raises(ModelLoadingError) as exc_info:
            registry.get_orientation_adapter()
        
        assert "Impossible de charger le modèle d'orientation" in str(exc_info.value)
    
    def test_unload_model(self):
        """Test unloading a model from memory"""
        registry = ModelRegistry()
        
        # Load a model first (mock)
        with patch.object(registry, '_models', {'detection': Mock()}):
            registry.unload_model(ModelType.DETECTION)
            assert 'detection' not in registry._models
    
    def test_unload_all(self):
        """Test unloading all models from memory"""
        registry = ModelRegistry()
        
        # Load some models first (mock)
        registry._models = {'orientation': Mock(), 'detection': Mock()}
        registry._adapters = {'orientation': Mock()}
        
        registry.unload_all()
        
        assert len(registry._models) == 0
        assert len(registry._adapters) == 0
    
    def test_get_orientation_adapter_force_reload(self):
        """Test forcing reload of orientation adapter"""
        registry = ModelRegistry()
        
        with patch('src.models.registry.from_hub') as mock_from_hub, \
             patch('src.models.registry.page_orientation_predictor') as mock_predictor, \
             patch('src.models.registry.OrientationModelAdapter') as mock_adapter:
            
            mock_from_hub.return_value = Mock()
            mock_predictor.return_value = Mock()
            mock_adapter.return_value = Mock()
            
            # First load
            registry.get_orientation_adapter()
            assert mock_from_hub.call_count == 1
            
            # Force reload
            mock_from_hub.reset_mock()
            registry.get_orientation_adapter(force_reload=True)
            assert mock_from_hub.call_count == 1

