from __future__ import annotations

import json
import re
from typing import Any

from research_assistant.config import Settings, load_settings
from research_assistant.llm.client import OllamaClient
from research_assistant.llm.types import LLMClient
from research_assistant.tools.registry import ToolSpec

_SUMMARIZE_SYSTEM = """You summarize source text for a research assistant.
Return JSON only with this exact shape:
{"summary": "2-6 sentence summary", "key_points": ["point 1", "point 2"]}
Use only information present in the source. Do not invent facts.
If the source is too thin to summarize well, say that in the summary and return few key points.
"""


def _fail(error: str, truncated: bool = False) -> dict[str, Any]:
    payload = {
        "summary": None,
        "key_points": [],
        "truncated": truncated,
    }
    return {
        "ok": False,
        "summary": None,
        "key_points": [],
        "error": error,
        "data": payload,
    }


def _extract_json(text: str) -> dict[str, Any] | None:
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None


def _normalize_points(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        return [text] if text else []
    if not isinstance(raw, list):
        return []
    points: list[str] = []
    for item in raw:
        text = str(item).strip()
        if text:
            points.append(text)
        if len(points) >= 8:
            break
    return points


def summarize_source(
    content: str,
    settings: Settings | None = None,
    llm: LLMClient | None = None,
) -> dict[str, Any]:
    """Summarize source text with Ollama. Returns structured summary or an error."""
    cfg = settings or load_settings()
    if content is None:
        return _fail("Content must not be empty.")

    text = str(content).strip()
    if not text:
        return _fail("Content must not be empty.")

    truncated = len(text) > cfg.summarize_max_chars
    if truncated:
        text = text[: cfg.summarize_max_chars].rstrip()

    client = llm or OllamaClient(cfg)
    messages = [
        {"role": "system", "content": _SUMMARIZE_SYSTEM},
        {"role": "user", "content": f"Source text:\n\n{text}"},
    ]

    try:
        result = client.chat(messages, tools=[], format="json")
    except Exception as exc:
        return _fail(f"Summarization failed: {exc}", truncated=truncated)

    parsed = _extract_json(result.content or "")
    if not parsed:
        return _fail("Model did not return valid JSON summary.", truncated=truncated)

    summary = str(parsed.get("summary") or "").strip()
    key_points = _normalize_points(parsed.get("key_points"))
    if not summary:
        return _fail("Model returned an empty summary.", truncated=truncated)

    data = {
        "summary": summary,
        "key_points": key_points,
        "truncated": truncated,
    }
    return {
        "ok": True,
        "summary": summary,
        "key_points": key_points,
        "error": None,
        "data": data,
    }


def handle_summarize_source(**kwargs: Any) -> dict[str, Any]:
    return summarize_source(content=kwargs.get("content", ""))


SUMMARIZE_SOURCE_SPEC = ToolSpec(
    name="summarize_source",
    description=(
        "Summarize cleaned source or page text. "
        "Use this after scrape_page (or when the user pastes substantial source text) "
        "to get a concise summary and key points. "
        "Pass the source text in the content argument. "
        "Do not pass a URL here; scrape the page first."
    ),
    parameters={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The source or page text to summarize.",
            }
        },
        "required": ["content"],
    },
    handler=handle_summarize_source,
)
