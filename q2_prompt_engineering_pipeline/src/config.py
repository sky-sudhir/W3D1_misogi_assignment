"""Configuration settings for the prompt engineering pipeline."""
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, Optional

# Load environment variables from .env file if it exists
load_dotenv()

class Config:
    """Application configuration."""
    
    # Base directories
    BASE_DIR = Path(__file__).parent.parent
    LOGS_DIR = BASE_DIR / "logs"
    PROMPTS_DIR = BASE_DIR / "prompts"
    TASKS_DIR = BASE_DIR / "tasks"
    EVALUATION_DIR = BASE_DIR / "evaluation"
    
    # Ensure directories exist
    for directory in [LOGS_DIR, PROMPTS_DIR, TASKS_DIR, EVALUATION_DIR]:
        directory.mkdir(exist_ok=True)
    
    # Model configuration
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini-2.0-flash")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
    
    # Reasoning configuration
    DEFAULT_NUM_PATHS = int(os.getenv("DEFAULT_NUM_PATHS", "3"))
    MIN_CONFIDENCE_THRESHOLD = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.7"))
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = LOGS_DIR / "pipeline.log"
    
    # Evaluation
    EVALUATION_METRICS = [
        "accuracy",
        "confidence",
        "response_time",
        "path_length"
    ]
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith('_') and not callable(value)
            and not isinstance(value, classmethod)
            and not isinstance(value, staticmethod)
        }

# Create a singleton instance
config = Config()

def get_config() -> Config:
    """Get the configuration instance."""
    return config
