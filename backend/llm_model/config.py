"""Configuration module for the LLM agent backend."""
import os
from typing import List


class Config:
    """Configuration settings for the backend."""

    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', '')
    DEFAULT_MODEL: str = 'openai/gpt-4o-mini'

    # Agent Configuration
    MAX_AGENT_ITERATIONS: int = 5

    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000"]

    # Streaming Configuration
    STREAM_CHUNK_SIZE: int = 3
    QUEUE_TIMEOUT: float = 0.02
    ASYNC_SLEEP: float = 0.001

    @classmethod
    def validate_api_key(cls) -> None:
        """Validate that the OpenAI API key is set and appears valid."""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        if 'your_openai_api_key_here' in cls.OPENAI_API_KEY.lower() or 'sk-' not in cls.OPENAI_API_KEY:
            raise ValueError(
                f"OPENAI_API_KEY appears to be invalid. "
                f"Please set a real OpenAI API key in backend/.env"
            )
