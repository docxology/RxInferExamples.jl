"""
logging_utils - Logging infrastructure utilities.

Provides standardized logging setup and helpers with emoji support.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional


class EmojiFormatter(logging.Formatter):
    """Custom formatter for adding emojis based on log level."""
    
    LEVEL_EMOJIS = {
        logging.DEBUG: "🐛",
        logging.INFO: "ℹ️ ",
        logging.WARNING: "⚠️ ",
        logging.ERROR: "❌",
        logging.CRITICAL: "🚨"
    }

    def format(self, record):
        emoji = self.LEVEL_EMOJIS.get(record.levelno, "🔹")
        record.emoji = emoji
        return super().format(record)


def setup_logger(
    output_dir: str,
    config_path: str = '',
    seed: int = 0,
    log_filename: str = 'execution.log'
) -> logging.Logger:
    """
    Initialize the execution log file with system context and emoji support.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, log_filename)
    
    # Write header first with 'w' mode to clear the file and add the header
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("╔" + "═" * 58 + "╗\n")
        f.write("║ EXECUTION LOG                                            ║\n")
        f.write("╚" + "═" * 58 + "╝\n")
    
    # Create logger
    logger = logging.getLogger('experiment')
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Format strings
    file_format_str = '[%(asctime)s] %(emoji)s [%(levelname)s] %(message)s'
    date_format_str = '%Y-%m-%d %H:%M:%S'
    
    # File handler (use 'a' mode because we already wrote the header)
    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = EmojiFormatter(file_format_str, datefmt=date_format_str)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = EmojiFormatter('[%(asctime)s] %(emoji)s %(message)s', 
                                    datefmt='%H:%M:%S')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Date:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Config:    {config_path}")
    logger.info(f"Output:    {output_dir}")
    logger.info(f"Seed:      {seed}")
    logger.info("─" * 60)
    
    return logger


def log_section(logger: logging.Logger, name: str):
    """
    Log a visual section header.
    """
    logger.info("")
    logger.info("═" * 40)
    logger.info(f"📌 SECTION: {name}")
    logger.info("═" * 40)


def log_validation(logger: logging.Logger, result: Any):
    """
    Log the validation result with visual emphasis.
    """
    status = "SUCCESS" if result.passed else "FAILURE"
    emoji = "✅" if result.passed else "🚨"
    
    logger.info("")
    logger.info("─" * 40)
    level = logging.INFO if result.passed else logging.WARNING
    logger.log(level, f"{emoji} VALIDATION {status}")
    
    if hasattr(result, 'messages'):
        for msg in result.messages:
            m_emoji = "❌" if 'FAIL' in msg else "🔹"
            logger.info(f"{m_emoji} {msg}")
    
    logger.info("─" * 40)


def log_config(logger: logging.Logger, config: Dict[str, Any]):
    """
    Log the active configuration dictionary.
    """
    logger.info("⚙️  Active Configuration:")
    for key, val in config.items():
        logger.info(f"   {key}: {val}")
