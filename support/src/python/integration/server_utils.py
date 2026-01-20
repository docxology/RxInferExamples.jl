"""
server_utils - Utilities for RxInfer server integration.

This module provides utilities for connecting to and interacting with
RxInfer servers.

Functions:
    create_client: Create an RxInfer client instance
    ping_server: Check server availability
    create_model: Create a model instance
    delete_model: Delete a model instance
"""

import logging
from typing import Optional, Any, Dict

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_SERVER_URL = "http://localhost:8000/v1"


def create_client(
    server_url: Optional[str] = None,
    api_key: Optional[str] = None
):
    """
    Create an RxInfer client instance.
    
    Args:
        server_url: Server URL (default: localhost:8000/v1)
        api_key: API key for authentication (auto-generated if None)
        
    Returns:
        RxInferClient instance
        
    Raises:
        ImportError: If rxinferclient package is not installed
    """
    try:
        from rxinferclient import RxInferClient
    except ImportError:
        raise ImportError(
            "rxinferclient package not installed. "
            "Install with: pip install rxinferclient"
        )
    
    if server_url and api_key:
        return RxInferClient(server_url=server_url, api_key=api_key)
    elif server_url:
        return RxInferClient(server_url=server_url)
    elif api_key:
        return RxInferClient(api_key=api_key)
    else:
        return RxInferClient()


def ping_server(client) -> bool:
    """
    Check server availability.
    
    Args:
        client: RxInferClient instance
        
    Returns:
        True if server responds with 'ok'
    """
    try:
        response = client.server.ping_server()
        return response.status == 'ok'
    except Exception as e:
        logger.error(f"Server ping failed: {e}")
        return False


def create_model(client, model_name: str, config: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Create a model instance on the server.
    
    Args:
        client: RxInferClient instance
        model_name: Name of the model to create
        config: Optional model configuration
        
    Returns:
        Instance ID if successful, None otherwise
    """
    try:
        request = {"model_name": model_name}
        if config:
            request.update(config)
        response = client.models.create_model_instance(request)
        logger.info(f"Created model instance: {response.instance_id}")
        return response.instance_id
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        return None


def delete_model(client, instance_id: str) -> bool:
    """
    Delete a model instance from the server.
    
    Args:
        client: RxInferClient instance
        instance_id: Instance ID to delete
        
    Returns:
        True if successful
    """
    try:
        client.models.delete_model_instance(instance_id=instance_id)
        logger.info(f"Deleted model instance: {instance_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete model: {e}")
        return False
