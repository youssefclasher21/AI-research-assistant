from research_assistant.tools.registry import ToolRegistry, build_default_registry
from research_assistant.tools.search_web import SEARCH_WEB_SPEC


def test_default_registry_includes_search_web_and_scrape_page():
    registry = build_default_registry()
    assert registry.names() == ["scrape_page", "search_web"]
    names = {schema["function"]["name"] for schema in registry.ollama_tools()}
    assert names == {"search_web", "scrape_page"}


def test_unknown_tool_returns_error():
    registry = ToolRegistry()
    registry.register(SEARCH_WEB_SPEC)
    result = registry.execute("not_a_tool", {"query": "x"})
    assert result["ok"] is False
    assert "Unknown tool" in result["error"]
