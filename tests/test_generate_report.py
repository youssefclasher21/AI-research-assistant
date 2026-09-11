from typing import Any

from research_assistant.config import Settings
from research_assistant.llm.types import ChatResult, LLMClient
from research_assistant.tools.generate_report import generate_report

SOURCE_A = {
    "title": "Ollama docs",
    "url": "https://ollama.com/blog/tool-support",
    "summary": "Ollama documents how models can call tools through the chat API.",
}
SOURCE_B = {
    "title": "Hardware notes",
    "url": "https://example.com/ram-limits",
    "text": "Running an 8B model on a 16 GB laptop is possible but slow on CPU.",
}

SETTINGS = Settings(
    ollama_host="http://127.0.0.1:11434",
    ollama_model="qwen3:8b",
    report_min_source_chars=20,
    report_min_sources=2,
    report_max_chars=4000,
)


class ScriptedLLM(LLMClient):
    def __init__(self, script: list[ChatResult]) -> None:
        self._script = list(script)
        self.calls = 0
        self.messages_seen: list[list[dict[str, Any]]] = []
        self.formats: list[str | None] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        format: str | None = None,
    ) -> ChatResult:
        self.calls += 1
        self.messages_seen.append(messages)
        self.formats.append(format)
        if not self._script:
            raise AssertionError("LLM was called more times than scripted")
        return self._script.pop(0)


def _json_result(payload: dict[str, Any]) -> ChatResult:
    import json

    content = json.dumps(payload)
    return ChatResult(
        content=content,
        tool_calls=[],
        message={"role": "assistant", "content": content},
    )


VALID_REPORT = {
    "title": "Local LLM research",
    "introduction": "This report reviews collected notes on local models.",
    "key_findings": ["Ollama supports tool calling", "8B models are RAM-heavy"],
    "conclusion": "Local research assistants are feasible on modest laptops.",
    "sources": [{"title": "Dropped", "url": "https://invented.example"}],
}


def test_empty_sources_are_rejected_without_llm():
    llm = ScriptedLLM([])
    for raw in (None, "", "   ", [], {}):
        result = generate_report(raw, SETTINGS, llm=llm)
        assert result["ok"] is False
        assert result["error"]
    assert llm.calls == 0


def test_single_source_is_rejected_without_llm():
    llm = ScriptedLLM([])
    result = generate_report([SOURCE_A], SETTINGS, llm=llm)
    assert result["ok"] is False
    assert "at least 2" in result["error"]
    assert llm.calls == 0


def test_malformed_entries_are_skipped_if_two_usable_remain():
    llm = ScriptedLLM([_json_result(VALID_REPORT)])
    mixed = [None, {}, {"foo": 1}, SOURCE_A, SOURCE_B]
    result = generate_report(mixed, SETTINGS, llm=llm)
    assert result["ok"] is True
    assert llm.calls == 1


def test_generate_report_parses_json_and_preserves_source_citations():
    llm = ScriptedLLM([_json_result(VALID_REPORT)])
    result = generate_report([SOURCE_A, SOURCE_B], SETTINGS, llm=llm)
    assert result["ok"] is True
    assert result["title"] == "Local LLM research"
    assert "tool calling" in result["key_findings"][0]
    assert result["sources"][0]["url"] == SOURCE_A["url"]
    assert result["sources"][1]["url"] == SOURCE_B["url"]
    assert result["sources"][0]["title"] == "Ollama docs"
    assert "invented.example" not in str(result["sources"])
    assert llm.formats == ["json"]
    user = llm.messages_seen[0][1]["content"]
    assert "https://ollama.com/blog/tool-support" in user
    assert "16 GB laptop" in user


def test_json_string_sources_are_accepted():
    import json

    llm = ScriptedLLM([_json_result(VALID_REPORT)])
    encoded = json.dumps([SOURCE_A, SOURCE_B])
    result = generate_report(encoded, SETTINGS, llm=llm)
    assert result["ok"] is True
    assert len(result["sources"]) == 2


def test_invalid_model_json_returns_structured_error():
    llm = ScriptedLLM(
        [
            ChatResult(
                content="not json",
                tool_calls=[],
                message={"role": "assistant", "content": "not json"},
            )
        ]
    )
    result = generate_report([SOURCE_A, SOURCE_B], SETTINGS, llm=llm)
    assert result["ok"] is False
    assert "JSON" in result["error"]


def test_missing_required_fields_return_structured_error():
    llm = ScriptedLLM([_json_result({"title": "Only a title"})])
    result = generate_report([SOURCE_A, SOURCE_B], SETTINGS, llm=llm)
    assert result["ok"] is False
    assert "missing required keys" in result["error"]


def test_empty_findings_or_conclusion_return_structured_error():
    payload = dict(VALID_REPORT)
    payload["key_findings"] = []
    llm = ScriptedLLM([_json_result(payload)])
    result = generate_report([SOURCE_A, SOURCE_B], SETTINGS, llm=llm)
    assert result["ok"] is False
    assert "key findings" in result["error"]


def test_long_material_is_truncated_before_llm():
    tight = Settings(
        ollama_host="http://127.0.0.1:11434",
        ollama_model="qwen3:8b",
        report_min_source_chars=20,
        report_min_sources=2,
        report_max_chars=80,
    )
    llm = ScriptedLLM([_json_result(VALID_REPORT)])
    long_a = {**SOURCE_A, "summary": "alpha " * 40}
    long_b = {**SOURCE_B, "text": "beta " * 40}
    result = generate_report([long_a, long_b], tight, llm=llm)
    assert result["ok"] is True
    assert result["data"]["truncated"] is True
    user = llm.messages_seen[0][1]["content"]
    assert len(user) < 400


def test_ollama_failure_returns_structured_error():
    class BoomLLM(LLMClient):
        def chat(self, messages, tools, format=None):
            raise ConnectionError("ollama is down")

    result = generate_report([SOURCE_A, SOURCE_B], SETTINGS, llm=BoomLLM())
    assert result["ok"] is False
    assert "Report generation failed" in result["error"]
