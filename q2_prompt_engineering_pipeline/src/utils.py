"""Utility functions for the prompt engineering pipeline."""
import json
import hashlib
import random
import string
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar, Type
from datetime import datetime
import logging

from loguru import logger

T = TypeVar('T')

def generate_id(prefix: str = "", length: int = 8) -> str:
    """Generate a random ID with an optional prefix.
    
    Args:
        prefix: Optional prefix for the ID
        length: Length of the random part of the ID
        
    Returns:
        A string ID in the format "prefix_randomchars"
    """
    chars = string.ascii_lowercase + string.digits
    random_part = ''.join(random.choices(chars, k=length))
    return f"{prefix}_{random_part}" if prefix else random_part

def calculate_md5(data: str) -> str:
    """Calculate the MD5 hash of a string."""
    return hashlib.md5(data.encode('utf-8')).hexdigest()

def load_json_file(file_path: Path) -> Any:
    """Load JSON data from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        raise

def save_json_file(data: Any, file_path: Path, indent: int = 2) -> None:
    """Save data to a JSON file."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
    except (TypeError, IOError) as e:
        logger.error(f"Error saving JSON to {file_path}: {e}")
        raise

def parse_datetime(dt_str: str) -> datetime:
    """Parse a datetime string in ISO format."""
    try:
        if dt_str.endswith('Z'):
            dt_str = dt_str[:-1] + '+00:00'
        return datetime.fromisoformat(dt_str)
    except ValueError as e:
        logger.warning(f"Failed to parse datetime '{dt_str}': {e}")
        return datetime.utcnow()

def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """Configure logging with loguru.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to
    """
    # Remove default handler
    logger.remove()
    
    # Add stderr handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=log_level
    )
    
    # Add file handler if specified
    if log_file:
        logger.add(
            log_file,
            rotation="10 MB",
            retention="1 month",
            level=log_level,
            encoding="utf-8"
        )

def validate_config(config: Dict[str, Any], required_fields: List[str]) -> bool:
    """Validate that a configuration dictionary contains all required fields."""
    missing = [field for field in required_fields if field not in config]
    if missing:
        logger.error(f"Missing required configuration fields: {', '.join(missing)}")
        return False
    return True

def get_logger(name: str = None) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logger.bind(module=name) if name else logger

def retry_on_exception(
    max_retries: int = 3,
    exceptions: tuple = (Exception,),
    backoff_factor: float = 1.0
):
    """Decorator to retry a function on specified exceptions.
    
    Args:
        max_retries: Maximum number of retry attempts
        exceptions: Tuple of exceptions to catch and retry on
        backoff_factor: Multiplier for exponential backoff
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    retries += 1
                    if retries >= max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__}")
                        raise
                    
                    wait_time = backoff_factor * (2 ** (retries - 1))
                    logger.warning(
                        f"Retry {retries}/{max_retries} for {func.__name__} after error: {e}. "
                        f"Waiting {wait_time:.1f}s before retry..."
                    )
                    time.sleep(wait_time)
        return wrapper
    return decorator
