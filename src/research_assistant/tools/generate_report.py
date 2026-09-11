from __future__ import annotations

import json
import re
from typing import Any

from research_assistant.config import Settings, load_settings
from research_assistant.llm.client import OllamaClient
from research_assistant.llm.types import LLMClient
from research_assistant.tools.registry import ToolSpec

_REPORT_SYSTEM = """You write a structured research report for a research assistant.
Return JSON only with this exact shape:
{
  "title": "short report title",
  "introduction": "1-2 paragraph introduction",
  "key_findings": ["finding 1", "finding 2"],
  "conclusion": "closing synthesis",
  "sources": [{"title": "source title", "url": "https://..."}]
}
Use only the supplied research material. Do not invent studies, quotes, URLs, or findings.
If a supplied source has a title or URL, keep it in the sources array.
key_findings must be an array of strings.
"""

_REQUIRED_KEYS = ("title", "introduction", "key_findings", "conclusion", "sources")
_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)


def _empty_payload(truncated: bool = False) -> dict[str, Any]:
    return {
        "title": None,
        "introduction": None,
        "key_findings": [],
        "conclusion": None,
        "sources": [],
        "truncated": truncated,
    }


def _fail(error: str, truncated: bool = False) -> dict[str, Any]:
    payload = _empty_payload(truncated)
    return {
        "ok": False,
        "title": None,
        "introduction": None,
        "key_findings": [],
        "conclusion": None,
        "sources": [],
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


def _normalize_findings(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        return [text] if text else []
    if not isinstance(raw, list):
        return []
    findings: list[str] = []
    for item in raw:
        text = str(item).strip()
        if text:
            findings.append(text)
        if len(findings) >= 12:
            break
    return findings


def _coerce_sources_argument(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, (list, dict)):
        return raw
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    return parsed


def _entry_from_mapping(item: dict[str, Any], index: int) -> dict[str, Any] | None:
    text = (
        item.get("text")
        or item.get("content")
        or item.get("summary")
        or item.get("snippet")
        or item.get("body")
        or ""
    )
    title = item.get("title") or item.get("name") or ""
    url = item.get("url") or item.get("href") or ""
    blob = " ".join(str(part) for part in (title, url, text) if part).strip()
    if not blob:
        return None
    if not url:
        match = _URL_RE.search(blob)
        url = match.group(0) if match else ""
    return {
        "title": str(title).strip() or f"Source {index}",
        "url": str(url).strip() or None,
        "text": str(text).strip() or blob,
    }


def _parse_source_entries(raw: Any) -> tuple[list[dict[str, Any]], str | None]:
    if raw is None:
        return [], "Sources must not be empty."
    if isinstance(raw, str) and not raw.strip():
        return [], "Sources must not be empty."

    values: list[Any]
    if isinstance(raw, dict):
        values = [raw]
    elif isinstance(raw, list):
        values = raw
    elif isinstance(raw, str):
        values = [raw]
    else:
        return [], "Sources must be text, a JSON list, or source objects."

    entries: list[dict[str, Any]] = []
    index = 1
    for item in values:
        if item is None or item == "":
            continue
        if isinstance(item, dict):
            entry = _entry_from_mapping(item, index)
            if entry is None:
                continue
            entries.append(entry)
            index += 1
            continue
        text = str(item).strip()
        if not text:
            continue
        match = _URL_RE.search(text)
        entries.append(
            {
                "title": f"Source {index}",
                "url": match.group(0) if match else None,
                "text": text,
            }
        )
        index += 1
    return entries, None


def _citations_from_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for i, entry in enumerate(entries, start=1):
        citations.append(
            {
                "title": entry.get("title") or f"Source {i}",
                "url": entry.get("url"),
            }
        )
    return citations


def _format_material(entries: list[dict[str, Any]], max_chars: int) -> tuple[str, bool]:
    chunks: list[str] = []
    used = 0
    truncated = False
    for i, entry in enumerate(entries, start=1):
        header = f"Source {i}"
        if entry.get("title"):
            header += f" — {entry['title']}"
        if entry.get("url"):
            header += f" ({entry['url']})"
        body = entry.get("text") or ""
        block = f"{header}\n{body}"
        remaining = max_chars - used
        if remaining <= 0:
            truncated = True
            break
        if len(block) > remaining:
            block = block[:remaining].rstrip()
            truncated = True
        chunks.append(block)
        used += len(block) + 2
    return "\n\n".join(chunks), truncated


def generate_report(
    sources: Any,
    settings: Settings | None = None,
    llm: LLMClient | None = None,
) -> dict[str, Any]:
    """Generate a structured research report from collected sources."""
    cfg = settings or load_settings()
    raw = _coerce_sources_argument(sources)
    entries, parse_error = _parse_source_entries(raw)
    if parse_error:
        return _fail(parse_error)

    usable = [e for e in entries if len((e.get("text") or "").strip()) >= cfg.report_min_source_chars]
    if len(usable) < cfg.report_min_sources:
        return _fail(
            f"Need at least {cfg.report_min_sources} research sources with meaningful text "
            "before generating a report."
        )

    material, truncated = _format_material(usable, cfg.report_max_chars)
    citations = _citations_from_entries(usable)
    client = llm or OllamaClient(cfg)
    messages = [
        {"role": "system", "content": _REPORT_SYSTEM},
        {
            "role": "user",
            "content": (
                "Write a structured research report using only the following collected sources.\n\n"
                f"{material}\n\n"
                "Preserve these citations in the sources array:\n"
                f"{json.dumps(citations, ensure_ascii=False)}"
            ),
        },
    ]

    try:
        result = client.chat(messages, tools=[], format="json")
    except Exception as exc:
        return _fail(f"Report generation failed: {exc}", truncated=truncated)

    parsed = _extract_json(result.content or "")
    if not parsed:
        return _fail("Model did not return valid JSON report.", truncated=truncated)

    missing = [key for key in _REQUIRED_KEYS if key not in parsed]
    if missing:
        return _fail(
            "Model JSON is missing required keys: " + ", ".join(missing) + ".",
            truncated=truncated,
        )

    title = str(parsed.get("title") or "").strip()
    introduction = str(parsed.get("introduction") or "").strip()
    conclusion = str(parsed.get("conclusion") or "").strip()
    key_findings = _normalize_findings(parsed.get("key_findings"))

    if not title:
        return _fail("Model returned an empty title.", truncated=truncated)
    if not introduction:
        return _fail("Model returned an empty introduction.", truncated=truncated)
    if not key_findings:
        return _fail("Model returned empty key findings.", truncated=truncated)
    if not conclusion:
        return _fail("Model returned an empty conclusion.", truncated=truncated)

    data = {
        "title": title,
        "introduction": introduction,
        "key_findings": key_findings,
        "conclusion": conclusion,
        "sources": citations,
        "truncated": truncated,
    }
    return {
        "ok": True,
        "title": title,
        "introduction": introduction,
        "key_findings": key_findings,
        "conclusion": conclusion,
        "sources": citations,
        "error": None,
        "data": data,
    }


def handle_generate_report(**kwargs: Any) -> dict[str, Any]:
    return generate_report(sources=kwargs.get("sources", ""))


GENERATE_REPORT_SPEC = ToolSpec(
    name="generate_report",
    description=(
        "Generate a structured research report from collected sources. "
        "Use this when the user wants a complete report with introduction, key findings, "
        "conclusion, and sources. Pass the collected source texts/summaries as sources "
        "(a JSON list of objects with title, url, and text/summary is best). "
        "Do not call this until you have at least two meaningful sources."
    ),
    parameters={
        "type": "object",
        "properties": {
            "sources": {
                "description": (
                    "Collected research material: a JSON list of source objects "
                    "(title, url, text or summary) or concatenated source text."
                )
            }
        },
        "required": ["sources"],
    },
    handler=handle_generate_report,
)
