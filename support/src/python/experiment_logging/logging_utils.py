"""
logging_utils - Logging infrastructure utilities.

Provides standardized logging setup and helpers.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional


def setup_logger(
    output_dir: str,
    config_path: str = '',
    seed: int = 0,
    log_filename: str = 'execution.log'
) -> logging.Logger:
    """
    Initialize the execution log file with system context.
    
    Args:
        output_dir: Directory for the log file
        config_path: Path to configuration file (for logging)
        seed: Random seed (for logging)
        log_filename: Name of the log file
        
    Returns:
        Configured logger instance
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, log_filename)
    
    # Create logger
    logger = logging.getLogger('experiment')
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s',
                                    datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(file_format)
    logger.addHandler(console_handler)
    
    # Write header
    logger.info("Execution Log")
    logger.info("=" * 40)
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Configuration: {config_path}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Seed: {seed}")
    logger.info("-" * 40)
    
    return logger


def log_section(logger: logging.Logger, name: str):
    """
    Log a visual section header.
    
    Args:
        logger: Logger instance
        name: Section name
    """
    logger.info("")
    logger.info("=" * 40)
    logger.info(f"SECTION: {name}")
    logger.info("=" * 40)


def log_validation(logger: logging.Logger, result: Any):
    """
    Log the validation result with visual emphasis.
    
    Args:
        logger: Logger instance
        result: ValidationResult object with .passed and .messages attributes
    """
    status = "[PASS]" if result.passed else "[FAIL]"
    logger.info("")
    logger.info("-" * 40)
    logger.info(f"VALIDATION {status}")
    
    for msg in result.messages:
        level = logging.WARNING if 'FAIL' in msg else logging.INFO
        logger.log(level, msg)
    
    logger.info("-" * 40)


def log_config(logger: logging.Logger, config: Dict[str, Any]):
    """
    Log the active configuration dictionary.
    
    Args:
        logger: Logger instance
        config: Configuration dictionary
    """
    logger.info("Active Configuration:")
    for key, val in config.items():
        logger.info(f"  {key}: {val}")
