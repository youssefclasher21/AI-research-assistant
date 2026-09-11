from research_assistant.config import Settings
from research_assistant.tools.search_web import search_web

SETTINGS = Settings(
    ollama_host="http://127.0.0.1:11434",
    ollama_model="qwen3:8b",
)


def test_empty_query_is_rejected():
    assert search_web("", SETTINGS)["ok"] is False
    assert search_web("   ", SETTINGS)["ok"] is False
    assert search_web(None, SETTINGS)["ok"] is False  # type: ignore[arg-type]


def test_query_too_long_is_rejected():
    result = search_web("q" * (SETTINGS.max_search_query_chars + 1), SETTINGS)
    assert result["ok"] is False
    assert "exceeds" in result["error"]
