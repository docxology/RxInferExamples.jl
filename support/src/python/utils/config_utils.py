"""
config_utils - Configuration file utilities.

Provides utilities for loading and managing configuration files.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Try to import tomllib (Python 3.11+) or toml
try:
    import tomllib
    def _load_toml(path: str) -> Dict[str, Any]:
        with open(path, 'rb') as f:
            return tomllib.load(f)
except ImportError:
    try:
        import toml
        def _load_toml(path: str) -> Dict[str, Any]:
            return toml.load(path)
    except ImportError:
        def _load_toml(path: str) -> Dict[str, Any]:
            raise ImportError("Neither tomllib (Python 3.11+) nor toml package is available")


def load_toml(path: str, required: bool = True) -> Dict[str, Any]:
    """
    Load a TOML configuration file.
    
    Args:
        path: Path to the TOML file
        required: If True, raise error if file doesn't exist
        
    Returns:
        Dictionary of configuration values
    """
    if not Path(path).exists():
        if required:
            raise FileNotFoundError(f"Configuration file not found: {path}")
        return {}
    
    return _load_toml(path)


def load_json(path: str, required: bool = True) -> Dict[str, Any]:
    """
    Load a JSON configuration file.
    
    Args:
        path: Path to the JSON file
        required: If True, raise error if file doesn't exist
        
    Returns:
        Dictionary of configuration values
    """
    if not Path(path).exists():
        if required:
            raise FileNotFoundError(f"Configuration file not found: {path}")
        return {}
    
    with open(path, 'r') as f:
        return json.load(f)


def get_config_value(config: Dict[str, Any], *keys, default: Any = None) -> Any:
    """
    Safely get a nested configuration value.
    
    Args:
        config: Configuration dictionary
        *keys: Keys to traverse
        default: Default value if key not found
        
    Returns:
        The configuration value or default
        
    Example:
        >>> config = {'section': {'key': 'value'}}
        >>> get_config_value(config, 'section', 'key', default='default')
        'value'
    """
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    
    Later configs override earlier ones. Performs deep merge for nested dicts.
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration dictionary
    """
    result = {}
    
    for config in configs:
        for key, value in config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_configs(result[key], value)
            else:
                result[key] = value
    
    return result
