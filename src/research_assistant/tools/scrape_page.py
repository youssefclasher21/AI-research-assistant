from __future__ import annotations

import re
import socket
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from research_assistant.config import Settings, load_settings
from research_assistant.tools.registry import ToolSpec

_NOISE_TAGS = (
    "script",
    "style",
    "noscript",
    "iframe",
    "svg",
    "canvas",
    "form",
    "nav",
    "footer",
    "header",
    "aside",
)
_USER_AGENT = "AI-Research-Assistant/0.1 (+local student project)"
_ALLOWED_SCHEMES = {"http", "https"}


def _empty_data(url: str = "", status: int | None = None) -> dict[str, Any]:
    return {
        "status": status,
        "url": url,
        "title": None,
        "text": None,
        "char_count": 0,
        "truncated": False,
    }


def _normalize_url(url: Any) -> tuple[str | None, str | None]:
    if url is None:
        return None, "URL must not be empty."
    cleaned = str(url).strip()
    if not cleaned:
        return None, "URL must not be empty."
    parsed = urlparse(cleaned)
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        return None, "URL must start with http:// or https://."
    if not parsed.netloc:
        return None, "URL is missing a host."
    return cleaned, None


def _decode_body(raw: bytes, content_type: str) -> str:
    charset = "utf-8"
    match = re.search(r"charset=([\w-]+)", content_type or "", flags=re.IGNORECASE)
    if match:
        charset = match.group(1).strip()
    try:
        return raw.decode(charset, errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


def _clean_html(html: str) -> tuple[str, str]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_NOISE_TAGS):
        tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.get_text(" ", strip=True)

    root = soup.find("article") or soup.find("main") or soup.body or soup
    text = root.get_text("\n", strip=True)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return title, text


def scrape_page(url: str, settings: Settings | None = None) -> dict[str, Any]:
    """Fetch a page and return cleaned readable text, or a structured error."""
    cfg = settings or load_settings()
    cleaned_url, error = _normalize_url(url)
    if error:
        return {"ok": False, "data": _empty_data(str(url).strip() if url else ""), "error": error}

    request = Request(
        cleaned_url,
        headers={
            "User-Agent": _USER_AGENT,
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
        },
        method="GET",
    )

    try:
        with urlopen(request, timeout=cfg.scrape_timeout_seconds) as response:
            status = int(getattr(response, "status", 200) or 200)
            content_type = (response.headers.get("Content-Type") or "").lower()
            raw = response.read(cfg.scrape_max_bytes + 1)
    except HTTPError as exc:
        code = int(getattr(exc, "code", 0) or 0)
        return {
            "ok": False,
            "data": _empty_data(cleaned_url, status=code or None),
            "error": f"HTTP error: {code or str(exc)}",
        }
    except (TimeoutError, socket.timeout):
        return {
            "ok": False,
            "data": _empty_data(cleaned_url),
            "error": "Request timed out.",
        }
    except URLError as exc:
        return {
            "ok": False,
            "data": _empty_data(cleaned_url),
            "error": f"Network error: {exc.reason if getattr(exc, 'reason', None) else exc}",
        }
    except Exception as exc:
        return {
            "ok": False,
            "data": _empty_data(cleaned_url),
            "error": f"Scrape failed: {exc}",
        }

    if not any(kind in content_type for kind in ("html", "xml", "text/plain", "text/")) and content_type:
        if "json" in content_type or "image/" in content_type or "octet-stream" in content_type:
            return {
                "ok": False,
                "data": _empty_data(cleaned_url, status=status),
                "error": f"Unsupported content type: {content_type}",
            }

    if len(raw) > cfg.scrape_max_bytes:
        raw = raw[: cfg.scrape_max_bytes]

    html = _decode_body(raw, content_type)
    try:
        title, text = _clean_html(html)
    except Exception as exc:
        return {
            "ok": False,
            "data": _empty_data(cleaned_url, status=status),
            "error": f"Failed to parse HTML: {exc}",
        }

    if not text:
        return {
            "ok": False,
            "data": {
                "status": status,
                "url": cleaned_url,
                "title": title or None,
                "text": None,
                "char_count": 0,
                "truncated": False,
            },
            "error": "No readable text found on the page.",
        }

    truncated = len(text) > cfg.scrape_max_chars
    if truncated:
        text = text[: cfg.scrape_max_chars].rstrip()

    return {
        "ok": True,
        "data": {
            "status": status,
            "url": cleaned_url,
            "title": title or None,
            "text": text,
            "char_count": len(text),
            "truncated": truncated,
        },
        "error": None,
    }


def handle_scrape_page(**kwargs: Any) -> dict[str, Any]:
    return scrape_page(url=kwargs.get("url", ""))


SCRAPE_PAGE_SPEC = ToolSpec(
    name="scrape_page",
    description=(
        "Fetch a web page by URL and extract cleaned readable text. "
        "Use this when you already have a specific http or https URL "
        "(from the user or from search_web) and need the page contents. "
        "Returns status, url, title, and cleaned text."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The http(s) URL of the page to scrape.",
            }
        },
        "required": ["url"],
    },
    handler=handle_scrape_page,
)
