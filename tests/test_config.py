import pytest
import yaml
import logging
from pathlib import Path
from unittest.mock import patch, mock_open
from src.utils.config import Config


class TestConfig:
    """Test suite for the Config class."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            'embeddings': {
                'model_name': 'all-MiniLM-L6-v2',
                'device': 'cpu',
                'cache_dir': './test_cache'
            },
            'generation': {
                'model': 'qwen2.5:7b',
                'temperature': 0.7
            },
            'books': {
                'data_dir': './test_data',
                'processed_dir': './test_processed',
                'index_dir': './test_index'
            },
            'vector_store': {
                'collection_name': 'test_books'
            },
            'search': {
                'top_k': 5
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(levelname)s - %(message)s'
            }
        }
    
    @pytest.fixture
    def config_file(self, tmp_path, sample_config):
        """Create a temporary config file."""
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(sample_config, f)
        return config_path
    
    def test_load_config_success(self, config_file, sample_config):
        """Test successful configuration loading."""
        with patch.object(Path, 'mkdir'):
            config = Config(str(config_file))
            assert config._config == sample_config
    
    def test_load_config_file_not_found(self):
        """Test configuration loading with non-existent file."""
        with pytest.raises(FileNotFoundError):
            Config("non_existent.yaml")
    
    def test_create_directories(self, config_file, tmp_path):
        """Test directory creation."""
        with patch.object(Path, 'mkdir') as mock_mkdir:
            config = Config(str(config_file))
            # Should create 4 directories (data_dir, processed_dir, index_dir, cache_dir)
            assert mock_mkdir.call_count == 4
            mock_mkdir.assert_called_with(parents=True, exist_ok=True)
    
    def test_property_accessors(self, config_file, sample_config):
        """Test property accessors."""
        with patch.object(Path, 'mkdir'):
            config = Config(str(config_file))
            
            assert config.embeddings == sample_config['embeddings']
            assert config.generation == sample_config['generation']
            assert config.books == sample_config['books']
            assert config.vector_store == sample_config['vector_store']
            assert config.search == sample_config['search']
    
    def test_get_method_simple_key(self, config_file):
        """Test get method with simple keys."""
        with patch.object(Path, 'mkdir'):
            config = Config(str(config_file))
            
            assert config.get('embeddings') is not None
            assert config.get('non_existent') is None
            assert config.get('non_existent', 'default') == 'default'
    
    def test_get_method_nested_key(self, config_file):
        """Test get method with nested keys (dot notation)."""
        with patch.object(Path, 'mkdir'):
            config = Config(str(config_file))
            
            assert config.get('embeddings.model_name') == 'all-MiniLM-L6-v2'
            assert config.get('generation.temperature') == 0.7
            assert config.get('search.top_k') == 5
            assert config.get('embeddings.non_existent') is None
            assert config.get('embeddings.non_existent', 42) == 42
    
    def test_get_method_deep_nesting(self, config_file):
        """Test get method with deep nesting."""
        with patch.object(Path, 'mkdir'):
            config = Config(str(config_file))
            
            # Test non-existent deep path
            assert config.get('a.b.c.d') is None
            assert config.get('embeddings.model.type') is None
    
    @patch('logging.basicConfig')
    def test_setup_logging(self, mock_logging, config_file):
        """Test logging setup."""
        with patch.object(Path, 'mkdir'):
            config = Config(str(config_file))
            
            mock_logging.assert_called_once()
            call_kwargs = mock_logging.call_args[1]
            assert call_kwargs['level'] == logging.INFO
            assert 'format' in call_kwargs
    
    def test_config_with_missing_sections(self, tmp_path):
        """Test configuration with missing sections."""
        # Create config with missing sections
        incomplete_config = {
            'embeddings': {'model_name': 'test'},
            'logging': {'level': 'INFO', 'format': '%(message)s'}
        }
        
        config_path = tmp_path / "incomplete_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(incomplete_config, f)
        
        with patch.object(Path, 'mkdir'):
            # This should raise KeyError when trying to access missing sections
            with pytest.raises(KeyError):
                config = Config(str(config_path))