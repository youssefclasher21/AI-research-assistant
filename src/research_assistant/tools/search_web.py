from __future__ import annotations

from typing import Any

from research_assistant.config import Settings, load_settings
from research_assistant.tools.registry import ToolSpec


def search_web(query: str, settings: Settings | None = None) -> dict[str, Any]:
    """Search the public web. Returns structured results or an error object."""
    cfg = settings or load_settings()
    if query is None:
        return {"ok": False, "data": None, "error": "Query must not be empty."}

    cleaned = str(query).strip()
    if not cleaned:
        return {"ok": False, "data": None, "error": "Query must not be empty."}
    if len(cleaned) > cfg.max_search_query_chars:
        return {
            "ok": False,
            "data": None,
            "error": f"Query exceeds {cfg.max_search_query_chars} characters.",
        }

    try:
        from ddgs import DDGS

        raw_results = DDGS().text(cleaned, max_results=cfg.max_search_results)
        items = []
        for row in raw_results or []:
            items.append(
                {
                    "title": row.get("title") or "",
                    "url": row.get("href") or row.get("url") or "",
                    "snippet": row.get("body") or row.get("snippet") or "",
                }
            )
    except Exception as exc:
        return {"ok": False, "data": None, "error": f"Search failed: {exc}"}

    if not items:
        return {"ok": False, "data": [], "error": "No search results found."}

    return {"ok": True, "data": items, "error": None}


def handle_search_web(**kwargs: Any) -> dict[str, Any]:
    return search_web(query=kwargs.get("query", ""))


SEARCH_WEB_SPEC = ToolSpec(
    name="search_web",
    description=(
        "Search the public web for a query. "
        "Use this when you need current information, sources, or facts you should look up. "
        "Returns a list of titles, URLs, and snippets."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The web search query.",
            }
        },
        "required": ["query"],
    },
    handler=handle_search_web,
)
