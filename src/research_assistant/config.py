import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    ollama_host: str
    ollama_model: str
    max_search_query_chars: int = 300
    max_user_message_chars: int = 2000
    max_search_results: int = 5
    max_tool_rounds: int = 6
    scrape_timeout_seconds: int = 10
    scrape_max_bytes: int = 1_000_000
    scrape_max_chars: int = 12_000


def load_settings() -> Settings:
    return Settings(
        ollama_host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
    )
