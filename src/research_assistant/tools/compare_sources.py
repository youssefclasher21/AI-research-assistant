from __future__ import annotations

import json
import re
from typing import Any

from research_assistant.config import Settings, load_settings
from research_assistant.llm.client import OllamaClient
from research_assistant.llm.types import LLMClient
from research_assistant.tools.registry import ToolSpec

_COMPARE_SYSTEM = """You compare two research sources for a research assistant.
Return JSON only with this exact shape:
{
  "similarities": ["shared point 1"],
  "differences": ["how the sources disagree or differ"],
  "key_takeaways": ["what a reader should conclude"]
}
Each value must be an array of short strings.
Use only information present in the two sources. Do not invent facts.
If they mostly agree, say so in similarities and still note remaining differences if any.
"""

_REQUIRED_KEYS = ("similarities", "differences", "key_takeaways")


def _empty_payload(truncated: bool = False) -> dict[str, Any]:
    return {
        "similarities": [],
        "differences": [],
        "key_takeaways": [],
        "truncated": truncated,
    }


def _fail(error: str, truncated: bool = False) -> dict[str, Any]:
    payload = _empty_payload(truncated)
    return {
        "ok": False,
        "similarities": [],
        "differences": [],
        "key_takeaways": [],
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


def _clean_source(value: Any, label: str) -> tuple[str | None, str | None]:
    if value is None:
        return None, f"{label} must not be empty."
    text = str(value).strip()
    if not text:
        return None, f"{label} must not be empty."
    return text, None


def compare_sources(
    source1: str,
    source2: str,
    settings: Settings | None = None,
    llm: LLMClient | None = None,
) -> dict[str, Any]:
    """Compare two source texts or summaries with Ollama."""
    cfg = settings or load_settings()
    left, err1 = _clean_source(source1, "source1")
    if err1:
        return _fail(err1)
    right, err2 = _clean_source(source2, "source2")
    if err2:
        return _fail(err2)

    if left == right:
        return _fail("source1 and source2 must be different texts.")

    truncated = False
    if len(left) > cfg.compare_max_chars:
        left = left[: cfg.compare_max_chars].rstrip()
        truncated = True
    if len(right) > cfg.compare_max_chars:
        right = right[: cfg.compare_max_chars].rstrip()
        truncated = True

    client = llm or OllamaClient(cfg)
    messages = [
        {"role": "system", "content": _COMPARE_SYSTEM},
        {
            "role": "user",
            "content": f"Source 1:\n\n{left}\n\n---\n\nSource 2:\n\n{right}",
        },
    ]

    try:
        result = client.chat(messages, tools=[], format="json")
    except Exception as exc:
        return _fail(f"Comparison failed: {exc}", truncated=truncated)

    parsed = _extract_json(result.content or "")
    if not parsed:
        return _fail("Model did not return valid JSON comparison.", truncated=truncated)

    missing = [key for key in _REQUIRED_KEYS if key not in parsed]
    if missing:
        return _fail(
            "Model JSON is missing required keys: " + ", ".join(missing) + ".",
            truncated=truncated,
        )

    similarities = _normalize_points(parsed.get("similarities"))
    differences = _normalize_points(parsed.get("differences"))
    key_takeaways = _normalize_points(parsed.get("key_takeaways"))

    if not similarities and not differences and not key_takeaways:
        return _fail("Model returned an empty comparison.", truncated=truncated)

    data = {
        "similarities": similarities,
        "differences": differences,
        "key_takeaways": key_takeaways,
        "truncated": truncated,
    }
    return {
        "ok": True,
        "similarities": similarities,
        "differences": differences,
        "key_takeaways": key_takeaways,
        "error": None,
        "data": data,
    }


def handle_compare_sources(**kwargs: Any) -> dict[str, Any]:
    return compare_sources(
        source1=kwargs.get("source1", ""),
        source2=kwargs.get("source2", ""),
    )


COMPARE_SOURCES_SPEC = ToolSpec(
    name="compare_sources",
    description=(
        "Compare two source texts or source summaries. "
        "Use this when the user asks to compare, contrast, or cross-check two sources, "
        "or when you already have two distinct summaries/texts. "
        "Pass the first text as source1 and the second as source2. "
        "Do not call this with only one source."
    ),
    parameters={
        "type": "object",
        "properties": {
            "source1": {
                "type": "string",
                "description": "The first source text or summary.",
            },
            "source2": {
                "type": "string",
                "description": "The second source text or summary.",
            },
        },
        "required": ["source1", "source2"],
    },
    handler=handle_compare_sources,
)
