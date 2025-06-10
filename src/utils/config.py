"""Configuration management for AI Book Agent."""

import yaml
import os
from pathlib import Path
from typing import Dict, Any
import logging

class Config:
    """Configuration manager for the AI Book Agent."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._config = self._load_config()
        self._setup_logging()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Ensure directories exist
        self._create_directories(config)
        return config
    
    def _create_directories(self, config: Dict[str, Any]) -> None:
        """Create necessary directories."""
        directories = [
            config['books']['data_dir'],
            config['books']['processed_dir'],
            config['books']['index_dir'],
            config['embeddings']['cache_dir'],
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self._config['logging']['level']),
            format=self._config['logging']['format']
        )
    
    @property
    def embeddings(self) -> Dict[str, Any]:
        """Embedding configuration."""
        return self._config['embeddings']
    
    @property
    def generation(self) -> Dict[str, Any]:
        """Generation model configuration."""
        return self._config['generation']
    
    @property
    def books(self) -> Dict[str, Any]:
        """Books directory configuration."""
        return self._config['books']
    
    @property
    def vector_store(self) -> Dict[str, Any]:
        """Vector store configuration."""
        return self._config['vector_store']
    
    @property
    def search(self) -> Dict[str, Any]:
        """Search configuration."""
        return self._config['search']
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value

# Global configuration instance
config = Config()