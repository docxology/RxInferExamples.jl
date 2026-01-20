"""
Integration - External integration utilities.

Provides utilities for integrating with external services and repositories.
"""

from .gnn_utils import (
    check_git_available,
    clone_repository,
    verify_clone,
    setup_integration,
    GNN_REPO_URL,
    DEFAULT_BRANCH
)

from .server_utils import (
    create_client,
    ping_server,
    create_model,
    delete_model,
    DEFAULT_SERVER_URL
)

__all__ = [
    # GNN utils
    'check_git_available',
    'clone_repository', 
    'verify_clone',
    'setup_integration',
    'GNN_REPO_URL',
    'DEFAULT_BRANCH',
    # Server utils
    'create_client',
    'ping_server',
    'create_model',
    'delete_model',
    'DEFAULT_SERVER_URL'
]
