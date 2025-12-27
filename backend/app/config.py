"""
Configuration module for Quantum Computing LLM API.
Loads environment variables from .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# API Keys
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model paths
MODEL_PATH = PROJECT_ROOT / "training" / "model" / "final_model.pt"
TOKENIZER_PATH = PROJECT_ROOT / "training" / "tokenizer" / "tokenizer.json"

# Custom model settings
MODEL_TEMPERATURE = 0.2
MODEL_TOP_K = 30
MODEL_MAX_NEW_TOKENS = 150

# Groq settings
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
GROQ_TEMPERATURE = 0.2
GROQ_MAX_TOKENS = 300

# Lazy loading settings (custom model only)
IDLE_TIMEOUT_SECONDS = 300  # 5 minutes


def validate_config():
    """Check that required environment variables are set."""
    missing = []
    if not VOYAGE_API_KEY:
        missing.append("VOYAGE_API_KEY")
    if not DATABASE_URL:
        missing.append("DATABASE_URL")
    
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_PATH}")
    
    return True


def validate_groq_config():
    """Check that Groq API key is set. Called only when using Groq."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found. Set it in .env to use Groq mode.")
    return True
