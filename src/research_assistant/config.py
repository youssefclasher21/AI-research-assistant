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
    max_tool_rounds: int = 12
    max_tool_result_chars: int = 8_000
    scrape_timeout_seconds: int = 10
    scrape_max_bytes: int = 1_000_000
    scrape_max_chars: int = 12_000
    summarize_max_chars: int = 12_000
    compare_max_chars: int = 8_000
    report_max_chars: int = 16_000
    report_min_sources: int = 2
    report_min_source_chars: int = 40


def load_settings() -> Settings:
    return Settings(
        ollama_host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
    )
