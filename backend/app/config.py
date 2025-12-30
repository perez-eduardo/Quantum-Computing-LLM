"""Configuration for Quantum Computing LLM API."""

import os
from dotenv import load_dotenv

load_dotenv()

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
GROQ_TEMPERATURE = 0.2
GROQ_MAX_TOKENS = 300


def validate_config():
    """Check required environment variables."""
    missing = []
    if not VOYAGE_API_KEY:
        missing.append("VOYAGE_API_KEY")
    if not DATABASE_URL:
        missing.append("DATABASE_URL")
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    
    if missing:
        raise ValueError(f"Missing: {', '.join(missing)}")
    return True
