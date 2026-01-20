#!/usr/bin/env python3
"""
Orchestration script for GNN repository setup.

This thin orchestrator uses the gnn_utils module from support/src/
to clone and configure the GNN repository.

Usage:
    python support/scripts/gnn/clone_and_setup_gnn.py [options]

Options:
    --target-dir DIR    Target directory for cloning
    --branch BRANCH     Git branch to checkout (default: main)
    --dry-run           Show what would be done without executing
    --force             Remove existing directory and re-clone
    --quiet             Reduce output verbosity
"""

import argparse
import logging
import sys
from pathlib import Path

# Setup path for importing from src/
script_dir = Path(__file__).parent
support_dir = script_dir.parent.parent
src_dir = support_dir / "src"
sys.path.insert(0, str(src_dir))

from gnn_utils import (
    check_git_available,
    clone_repository,
    verify_clone,
    setup_integration,
    DEFAULT_BRANCH
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Clone and configure GNN repository for RxInferExamples.jl integration",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=script_dir / "GeneralizedNotationNotation",
        help="Target directory for cloning"
    )
    parser.add_argument(
        "--branch",
        default=DEFAULT_BRANCH,
        help=f"Git branch to checkout (default: {DEFAULT_BRANCH})"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove existing directory and re-clone"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    return parser.parse_args()


def main() -> int:
    """Main orchestration entry point."""
    args = parse_args()
    
    if args.quiet:
        logger.setLevel(logging.WARNING)
    
    logger.info("GNN Clone and Setup Script")
    logger.info("=" * 40)
    
    # Check prerequisites
    if not check_git_available():
        logger.error("Git is not available. Please install git and try again.")
        return 1
    
    # Clone repository
    if not clone_repository(
        target_dir=args.target_dir,
        branch=args.branch,
        dry_run=args.dry_run,
        force=args.force
    ):
        return 1
    
    # Verify clone (skip in dry run)
    if not args.dry_run:
        if not verify_clone(args.target_dir):
            logger.warning("Clone verification found issues")
    
    # Setup integration
    if not setup_integration(args.target_dir, dry_run=args.dry_run):
        return 1
    
    logger.info("=" * 40)
    logger.info("✅ GNN setup complete!")
    logger.info(f"Repository location: {args.target_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())