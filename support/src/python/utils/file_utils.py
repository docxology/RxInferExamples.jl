"""
file_utils - File system utilities.

Provides common file operations.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import List, Pattern
import re

logger = logging.getLogger(__name__)


def ensure_directory(path: str) -> str:
    """
    Create directory and all parent directories if they don't exist.
    
    Args:
        path: Directory path to create
        
    Returns:
        The path that was created
    """
    os.makedirs(path, exist_ok=True)
    return path


def copy_if_newer(source: str, target: str, quiet: bool = False) -> bool:
    """
    Copy source to target only if source is newer or target doesn't exist.
    
    Args:
        source: Source file path
        target: Target file path
        quiet: If True, suppress output
        
    Returns:
        True if copy was performed
    """
    source_path = Path(source)
    target_path = Path(target)
    
    if not source_path.exists():
        if not quiet:
            logger.warning(f"Source file does not exist: {source}")
        return False
    
    if not target_path.exists() or source_path.stat().st_mtime > target_path.stat().st_mtime:
        ensure_directory(str(target_path.parent))
        shutil.copy2(source, target)
        if not quiet:
            logger.info(f"Copied: {source} → {target}")
        return True
    
    return False


def find_files(directory: str, pattern: str, recursive: bool = True) -> List[str]:
    """
    Find all files in directory matching the pattern.
    
    Args:
        directory: Directory to search
        pattern: Regex pattern to match filenames
        recursive: If True, search recursively
        
    Returns:
        List of matching file paths
    """
    matches = []
    regex = re.compile(pattern)
    
    if recursive:
        for root, dirs, files in os.walk(directory):
            for f in files:
                if regex.search(f):
                    matches.append(os.path.join(root, f))
    else:
        for f in os.listdir(directory):
            full_path = os.path.join(directory, f)
            if os.path.isfile(full_path) and regex.search(f):
                matches.append(full_path)
    
    return matches


def safe_write(path: str, content: str) -> None:
    """
    Write content to file atomically (write to temp, then rename).
    
    Args:
        path: Target file path
        content: Content to write
    """
    temp_path = path + ".tmp"
    ensure_directory(os.path.dirname(path))
    
    with open(temp_path, 'w') as f:
        f.write(content)
    
    shutil.move(temp_path, path)
