from typing import Any

from research_assistant.config import Settings
from research_assistant.llm.types import ChatResult, LLMClient
from research_assistant.tools.summarize_source import summarize_source


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


SETTINGS = Settings(
    ollama_host="http://127.0.0.1:11434",
    ollama_model="qwen3:8b",
    summarize_max_chars=200,
)


def _json_result(payload: dict[str, Any]) -> ChatResult:
    import json

    content = json.dumps(payload)
    return ChatResult(
        content=content,
        tool_calls=[],
        message={"role": "assistant", "content": content},
    )


def test_empty_content_is_rejected_without_llm():
    llm = ScriptedLLM([])
    for raw in (None, "", "   "):
        result = summarize_source(raw, SETTINGS, llm=llm)  # type: ignore[arg-type]
        assert result["ok"] is False
        assert result["summary"] is None
        assert result["key_points"] == []
        assert result["error"]
    assert llm.calls == 0


def test_summarize_source_parses_mocked_ollama_json():
    llm = ScriptedLLM(
        [
            _json_result(
                {
                    "summary": "The article explains local LLM tool calling.",
                    "key_points": ["Uses Ollama", "Tools are model-selected"],
                }
            )
        ]
    )
    result = summarize_source(
        "Ollama can call tools when the model returns tool_calls.",
        SETTINGS,
        llm=llm,
    )
    assert result["ok"] is True
    assert result["error"] is None
    assert "local LLM tool calling" in result["summary"]
    assert result["key_points"] == ["Uses Ollama", "Tools are model-selected"]
    assert result["data"]["summary"] == result["summary"]
    assert llm.calls == 1
    assert llm.formats == ["json"]
    assert llm.messages_seen[0][0]["role"] == "system"
    assert llm.messages_seen[0][1]["role"] == "user"


def test_long_content_is_truncated_before_llm():
    llm = ScriptedLLM(
        [
            _json_result(
                {
                    "summary": "Truncated source was summarized.",
                    "key_points": ["Kept within limit"],
                }
            )
        ]
    )
    result = summarize_source("x" * 500, SETTINGS, llm=llm)
    assert result["ok"] is True
    assert result["data"]["truncated"] is True
    user_text = llm.messages_seen[0][1]["content"]
    assert len(user_text) < 500 + 40
    assert "x" * 201 not in user_text


def test_invalid_model_json_returns_structured_error():
    llm = ScriptedLLM(
        [
            ChatResult(
                content="this is not json",
                tool_calls=[],
                message={"role": "assistant", "content": "this is not json"},
            )
        ]
    )
    result = summarize_source("Some source text about research.", SETTINGS, llm=llm)
    assert result["ok"] is False
    assert result["summary"] is None
    assert "JSON" in result["error"]


def test_ollama_failure_returns_structured_error():
    class BoomLLM(LLMClient):
        def chat(self, messages, tools, format=None):
            raise ConnectionError("ollama is down")

    result = summarize_source("Some source text about research.", SETTINGS, llm=BoomLLM())
    assert result["ok"] is False
    assert "Summarization failed" in result["error"]
    assert result["key_points"] == []
