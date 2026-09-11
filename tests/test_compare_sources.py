from typing import Any

from research_assistant.config import Settings
from research_assistant.llm.types import ChatResult, LLMClient
from research_assistant.tools.compare_sources import compare_sources


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
    compare_max_chars=80,
)


def _json_result(payload: dict[str, Any]) -> ChatResult:
    import json

    content = json.dumps(payload)
    return ChatResult(
        content=content,
        tool_calls=[],
        message={"role": "assistant", "content": content},
    )


VALID_PAYLOAD = {
    "similarities": ["Both discuss local LLMs"],
    "differences": ["Source 2 warns about RAM limits"],
    "key_takeaways": ["Local models work, but hardware matters"],
}


def test_empty_inputs_are_rejected_without_llm():
    llm = ScriptedLLM([])
    assert compare_sources("", "second source text", SETTINGS, llm=llm)["ok"] is False
    assert compare_sources("first source text", "  ", SETTINGS, llm=llm)["ok"] is False
    assert compare_sources(None, "second", SETTINGS, llm=llm)["ok"] is False  # type: ignore[arg-type]
    assert llm.calls == 0


def test_identical_sources_are_rejected_without_llm():
    llm = ScriptedLLM([])
    result = compare_sources("same text", "same text", SETTINGS, llm=llm)
    assert result["ok"] is False
    assert "different" in result["error"]
    assert llm.calls == 0


def test_compare_sources_parses_mocked_ollama_json():
    llm = ScriptedLLM([_json_result(VALID_PAYLOAD)])
    result = compare_sources(
        "Source A discusses running local LLMs with Ollama.",
        "Source B discusses local LLMs and 16 GB RAM limits.",
        SETTINGS,
        llm=llm,
    )
    assert result["ok"] is True
    assert result["error"] is None
    assert result["similarities"] == VALID_PAYLOAD["similarities"]
    assert result["differences"] == VALID_PAYLOAD["differences"]
    assert result["key_takeaways"] == VALID_PAYLOAD["key_takeaways"]
    assert result["data"]["similarities"] == result["similarities"]
    assert llm.calls == 1
    assert llm.formats == ["json"]
    user = llm.messages_seen[0][1]["content"]
    assert "Source 1:" in user
    assert "Source 2:" in user


def test_long_sources_are_truncated_before_llm():
    llm = ScriptedLLM([_json_result(VALID_PAYLOAD)])
    result = compare_sources("a" * 200, "b" * 200, SETTINGS, llm=llm)
    assert result["ok"] is True
    assert result["data"]["truncated"] is True
    user = llm.messages_seen[0][1]["content"]
    assert "a" * 81 not in user
    assert "b" * 81 not in user


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
    result = compare_sources("first source text here", "second source text here", SETTINGS, llm=llm)
    assert result["ok"] is False
    assert "JSON" in result["error"]
    assert result["similarities"] == []


def test_missing_required_keys_returns_structured_error():
    llm = ScriptedLLM([_json_result({"similarities": ["overlap"]})])
    result = compare_sources("first source text here", "second source text here", SETTINGS, llm=llm)
    assert result["ok"] is False
    assert "missing required keys" in result["error"]


def test_empty_comparison_lists_return_structured_error():
    llm = ScriptedLLM(
        [_json_result({"similarities": [], "differences": [], "key_takeaways": []})]
    )
    result = compare_sources("first source text here", "second source text here", SETTINGS, llm=llm)
    assert result["ok"] is False
    assert "empty comparison" in result["error"]


def test_ollama_failure_returns_structured_error():
    class BoomLLM(LLMClient):
        def chat(self, messages, tools, format=None):
            raise ConnectionError("ollama is down")

    result = compare_sources(
        "first source text here",
        "second source text here",
        SETTINGS,
        llm=BoomLLM(),
    )
    assert result["ok"] is False
    assert "Comparison failed" in result["error"]
