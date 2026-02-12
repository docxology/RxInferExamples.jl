"""
file_utils - File system utilities.

Provides common file operations with absolute path handling.
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
    Returns the absolute path.
    """
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path


def ensure_parent_directory(path: str) -> str:
    """
    Ensure the parent directory of a file exists.
    Returns the absolute path to the parent.
    """
    abs_path = os.path.abspath(path)
    parent = os.path.dirname(abs_path)
    os.makedirs(parent, exist_ok=True)
    return parent


def copy_if_newer(source: str, target: str, quiet: bool = False) -> bool:
    """
    Copy source to target only if source is newer or target doesn't exist.
    """
    source_path = Path(os.path.abspath(source))
    target_path = Path(os.path.abspath(target))
    
    if not source_path.exists():
        if not quiet:
            logger.warning(f"⚠️  Source file does not exist: {source}")
        return False
    
    if not target_path.exists() or source_path.stat().st_mtime > target_path.stat().st_mtime:
        ensure_parent_directory(str(target_path))
        shutil.copy2(str(source_path), str(target_path))
        if not quiet:
            logger.info(f"ℹ️  Copied: {source} → {target}")
        return True
    
    return False


def find_files(directory: str, pattern: str, recursive: bool = True) -> List[str]:
    """
    Find all files in directory matching the pattern. Returns absolute paths.
    """
    matches = []
    regex = re.compile(pattern)
    abs_dir = os.path.abspath(directory)
    
    if recursive:
        for root, dirs, files in os.walk(abs_dir):
            for f in files:
                if regex.search(f):
                    matches.append(os.path.join(root, f))
    else:
        for f in os.listdir(abs_dir):
            full_path = os.path.join(abs_dir, f)
            if os.path.isfile(full_path) and regex.search(f):
                matches.append(full_path)
    
    return matches


def safe_write(path: str, content: str) -> None:
    """
    Write content to file atomically.
    """
    abs_path = os.path.abspath(path)
    temp_path = abs_path + ".tmp"
    ensure_parent_directory(abs_path)
    
    with open(temp_path, 'w') as f:
        f.write(content)
    
    shutil.move(temp_path, abs_path)
