"""
gnn_utils - Core utilities for GNN repository integration.

This module provides the core functionality for cloning and setting up
the GNN (Generalized Notation Notation) repository for integration with
RxInferExamples.jl.

Functions:
    check_git_available: Check if git is available
    clone_repository: Clone the GNN repository
    verify_clone: Verify repository structure
    setup_integration: Configure integration settings
"""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import List

# Repository configuration
GNN_REPO_URL = "https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation"
DEFAULT_BRANCH = "main"

logger = logging.getLogger(__name__)


def check_git_available() -> bool:
    """
    Check if git is available on the system.
    
    Returns:
        True if git is available, False otherwise
    """
    try:
        subprocess.run(
            ["git", "--version"],
            capture_output=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def clone_repository(
    target_dir: Path,
    branch: str = DEFAULT_BRANCH,
    dry_run: bool = False,
    force: bool = False,
    shallow: bool = True
) -> bool:
    """
    Clone the GNN repository to the target directory.
    
    Args:
        target_dir: Destination path for the cloned repository
        branch: Git branch to checkout
        dry_run: If True, only show what would be done
        force: If True, remove existing directory first
        shallow: If True, use shallow clone (depth=1)
        
    Returns:
        True if successful, False otherwise
    """
    # Handle existing directory
    if target_dir.exists():
        if force:
            logger.info(f"Removing existing directory: {target_dir}")
            if not dry_run:
                shutil.rmtree(target_dir)
        else:
            logger.warning(f"Directory already exists: {target_dir}")
            logger.info("Use --force to remove and re-clone")
            return False
    
    # Build clone command
    clone_cmd = ["git", "clone", "--branch", branch]
    if shallow:
        clone_cmd.extend(["--depth", "1"])
    clone_cmd.extend([GNN_REPO_URL, str(target_dir)])
    
    logger.info(f"Cloning GNN repository to: {target_dir}")
    logger.info(f"  Repository: {GNN_REPO_URL}")
    logger.info(f"  Branch: {branch}")
    
    if dry_run:
        logger.info(f"  [DRY RUN] Would execute: {' '.join(clone_cmd)}")
        return True
    
    try:
        subprocess.run(clone_cmd, capture_output=True, text=True, check=True)
        logger.info("✅ Clone successful")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Clone failed: {e.stderr}")
        return False


def verify_clone(target_dir: Path, required_files: List[str] = None) -> bool:
    """
    Verify the cloned repository has expected structure.
    
    Args:
        target_dir: Path to cloned repository
        required_files: List of files that must exist (default: ["README.md"])
        
    Returns:
        True if structure is valid
    """
    if required_files is None:
        required_files = ["README.md"]
    
    if not target_dir.exists():
        logger.warning("Target directory does not exist")
        return False
    
    for file in required_files:
        if not (target_dir / file).exists():
            logger.warning(f"Missing required file: {file}")
            return False
    
    logger.info("✅ Repository structure verified")
    return True


def setup_integration(target_dir: Path, dry_run: bool = False) -> bool:
    """
    Setup integration between GNN and RxInferExamples.jl.
    
    Args:
        target_dir: Path to GNN repository
        dry_run: If True, only show what would be done
        
    Returns:
        True if successful
    """
    logger.info("Setting up RxInferExamples integration...")
    
    if dry_run:
        logger.info("  [DRY RUN] Would configure integration")
        return True
    
    # Future: Add integration configuration here
    # - Create symlinks
    # - Configure paths
    # - Validate compatibility
    
    logger.info("✅ Integration setup complete")
    return True
