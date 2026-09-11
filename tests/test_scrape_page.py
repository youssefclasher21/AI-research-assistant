from email.message import EmailMessage
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from research_assistant.config import Settings
from research_assistant.tools.scrape_page import scrape_page

HTML = """
<html>
  <head><title>Example Article</title></head>
  <body>
    <nav>Home About Contact</nav>
    <script>window.track = true;</script>
    <style>body { color: red; }</style>
    <article>
      <h1>Example Article</h1>
      <p>This is the main readable content for the research assistant.</p>
    </article>
    <footer>Copyright 2026</footer>
  </body>
</html>
"""

SETTINGS = Settings(
    ollama_host="http://127.0.0.1:11434",
    ollama_model="qwen3:8b",
    scrape_timeout_seconds=5,
    scrape_max_bytes=50_000,
    scrape_max_chars=10_000,
)


class FakeResponse:
    def __init__(self, body: bytes, status: int = 200, content_type: str = "text/html; charset=utf-8"):
        self.status = status
        self.headers = {"Content-Type": content_type}
        self._body = body

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            return self._body
        return self._body[:n]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_invalid_urls_are_rejected():
    for raw in ("", "   ", "not-a-url", "ftp://example.com/file", "javascript:alert(1)"):
        result = scrape_page(raw, SETTINGS)
        assert result["ok"] is False
        assert result["error"]
        assert result["data"]["text"] is None


def test_extracts_readable_text_from_mocked_html():
    fake = FakeResponse(HTML.encode("utf-8"))
    with patch("research_assistant.tools.scrape_page.urlopen", return_value=fake):
        result = scrape_page("https://example.com/article", SETTINGS)

    assert result["ok"] is True
    data = result["data"]
    assert data["status"] == 200
    assert data["url"] == "https://example.com/article"
    assert data["title"] == "Example Article"
    assert "main readable content" in data["text"]
    assert "window.track" not in data["text"]
    assert "color: red" not in data["text"]
    assert "Home About Contact" not in data["text"]
    assert "Copyright 2026" not in data["text"]
    assert data["truncated"] is False


def test_http_error_is_returned_gracefully():
    headers = EmailMessage()
    error = HTTPError(
        "https://example.com/missing",
        404,
        "Not Found",
        headers,
        BytesIO(b""),
    )
    with patch("research_assistant.tools.scrape_page.urlopen", side_effect=error):
        result = scrape_page("https://example.com/missing", SETTINGS)

    assert result["ok"] is False
    assert "404" in result["error"]
    assert result["data"]["status"] == 404
    assert result["data"]["text"] is None


def test_network_failure_is_returned_gracefully():
    with patch(
        "research_assistant.tools.scrape_page.urlopen",
        side_effect=URLError("connection refused"),
    ):
        result = scrape_page("https://example.com/down", SETTINGS)

    assert result["ok"] is False
    assert "Network error" in result["error"]
    assert result["data"]["text"] is None


def test_long_content_is_truncated():
    long_html = (
        "<html><head><title>Long</title></head><body><article>"
        + ("word " * 200)
        + "</article></body></html>"
    )
    tight = Settings(
        ollama_host="http://127.0.0.1:11434",
        ollama_model="qwen3:8b",
        scrape_max_chars=40,
    )
    fake = FakeResponse(long_html.encode("utf-8"))
    with patch("research_assistant.tools.scrape_page.urlopen", return_value=fake):
        result = scrape_page("https://example.com/long", tight)

    assert result["ok"] is True
    assert result["data"]["truncated"] is True
    assert result["data"]["char_count"] <= 40
    assert len(result["data"]["text"]) <= 40
