from __future__ import annotations

import json
from typing import Any

from research_assistant.agent.orchestrator import AgentOrchestrator
from research_assistant.config import Settings
from research_assistant.llm.client import _parse_arguments
from research_assistant.llm.types import ChatResult, LLMClient, ToolCall
from research_assistant.tools.registry import ToolRegistry, ToolSpec


class ScriptedLLM(LLMClient):
    """Records a snapshot of messages on each call so later mutations are not hidden."""

    def __init__(self, script: list[ChatResult]) -> None:
        self._script = list(script)
        self.calls = 0
        self.messages_seen: list[list[dict[str, Any]]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        format: str | None = None,
    ) -> ChatResult:
        self.calls += 1
        self.messages_seen.append([dict(m) for m in messages])
        if not self._script:
            raise AssertionError("LLM was called more times than scripted")
        return self._script.pop(0)


def _settings(**overrides: Any) -> Settings:
    values = {
        "ollama_host": "http://127.0.0.1:11434",
        "ollama_model": "qwen3:8b",
        "max_tool_rounds": 12,
        "max_tool_result_chars": 8_000,
    }
    values.update(overrides)
    return Settings(**values)


def _tool_turn(name: str, arguments: dict[str, Any]) -> ChatResult:
    return ChatResult(
        content="",
        tool_calls=[ToolCall(name=name, arguments=arguments)],
        message={
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": name, "arguments": arguments}}],
        },
    )


def _final(text: str) -> ChatResult:
    return ChatResult(
        content=text,
        tool_calls=[],
        message={"role": "assistant", "content": text},
    )


def _research_registry() -> ToolRegistry:
    pages = {
        "https://example.com/a": "Page A discusses Ollama tool calling in detail for local models.",
        "https://example.com/b": "Page B discusses 16 GB RAM limits when running qwen3:8b.",
    }

    def search_web(**kwargs: Any) -> dict[str, Any]:
        return {
            "ok": True,
            "data": [
                {"title": "Ollama tools", "url": "https://example.com/a", "snippet": "tool calling"},
                {"title": "RAM notes", "url": "https://example.com/b", "snippet": "16 GB laptop"},
            ],
            "error": None,
        }

    def scrape_page(**kwargs: Any) -> dict[str, Any]:
        url = kwargs.get("url", "")
        text = pages.get(url, f"Fallback text for {url}")
        return {
            "ok": True,
            "data": {
                "status": 200,
                "url": url,
                "title": url,
                "text": text,
                "char_count": len(text),
                "truncated": False,
            },
            "error": None,
        }

    def summarize_source(**kwargs: Any) -> dict[str, Any]:
        content = kwargs.get("content", "")
        summary = f"Summary of: {content[:60]}"
        return {
            "ok": True,
            "summary": summary,
            "key_points": ["Local tools", "Hardware limits"],
            "error": None,
            "data": {"summary": summary, "key_points": ["Local tools", "Hardware limits"], "truncated": False},
        }

    def compare_sources(**kwargs: Any) -> dict[str, Any]:
        return {
            "ok": True,
            "similarities": ["Both discuss local LLMs"],
            "differences": ["Only B emphasizes RAM"],
            "key_takeaways": ["Hardware matters"],
            "error": None,
            "data": {
                "similarities": ["Both discuss local LLMs"],
                "differences": ["Only B emphasizes RAM"],
                "key_takeaways": ["Hardware matters"],
                "truncated": False,
            },
        }

    def generate_report(**kwargs: Any) -> dict[str, Any]:
        return {
            "ok": True,
            "title": "Local research assistants",
            "introduction": "Collected sources describe local tool calling.",
            "key_findings": ["Ollama can call tools", "8B models need RAM"],
            "conclusion": "A laptop can run this workflow with limits.",
            "sources": [
                {"title": "Ollama tools", "url": "https://example.com/a"},
                {"title": "RAM notes", "url": "https://example.com/b"},
            ],
            "error": None,
            "data": {
                "title": "Local research assistants",
                "introduction": "Collected sources describe local tool calling.",
                "key_findings": ["Ollama can call tools", "8B models need RAM"],
                "conclusion": "A laptop can run this workflow with limits.",
                "sources": [
                    {"title": "Ollama tools", "url": "https://example.com/a"},
                    {"title": "RAM notes", "url": "https://example.com/b"},
                ],
                "truncated": False,
            },
        }

    registry = ToolRegistry()
    for name, handler, params, required in (
        ("search_web", search_web, {"query": {"type": "string"}}, ["query"]),
        ("scrape_page", scrape_page, {"url": {"type": "string"}}, ["url"]),
        ("summarize_source", summarize_source, {"content": {"type": "string"}}, ["content"]),
        (
            "compare_sources",
            compare_sources,
            {"source1": {"type": "string"}, "source2": {"type": "string"}},
            ["source1", "source2"],
        ),
        ("generate_report", generate_report, {"sources": {"type": "string"}}, ["sources"]),
    ):
        registry.register(
            ToolSpec(
                name=name,
                description=name,
                parameters={"type": "object", "properties": params, "required": required},
                handler=handler,
            )
        )
    return registry


def _tool_payloads(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payloads = []
    for message in messages:
        if message.get("role") != "tool":
            continue
        payloads.append(json.loads(message["content"]))
    return payloads


def test_malformed_tool_arguments_do_not_become_query():
    assert _parse_arguments("{not-json") == {}
    assert _parse_arguments('{"url": "https://example.com"}') == {"url": "https://example.com"}


def test_scenario_a_search_scrape_summarize_sees_prior_tool_output():
    llm = ScriptedLLM(
        [
            _tool_turn("search_web", {"query": "ollama tool calling"}),
            _tool_turn("scrape_page", {"url": "https://example.com/a"}),
            _tool_turn(
                "summarize_source",
                {"content": "Page A discusses Ollama tool calling in detail for local models."},
            ),
            _final("Ollama can call tools locally."),
        ]
    )
    agent = AgentOrchestrator(settings=_settings(), llm=llm, registry=_research_registry())
    result = agent.run("What is Ollama tool calling?")

    assert [t.name for t in result.traces] == ["search_web", "scrape_page", "summarize_source"]
    assert result.reply == "Ollama can call tools locally."
    assert llm.calls == 4

    scrape_step_messages = llm.messages_seen[1]
    assert any(m.get("role") == "tool" and m.get("tool_name") == "search_web" for m in scrape_step_messages)
    search_payload = _tool_payloads(scrape_step_messages)[0]
    assert search_payload["data"][0]["url"] == "https://example.com/a"

    summarize_step = llm.messages_seen[2]
    scrape_payload = [p for p in _tool_payloads(summarize_step) if "text" in (p.get("data") or {})][0]
    assert "Ollama tool calling" in scrape_payload["data"]["text"]

    final_step = llm.messages_seen[3]
    assert [m.get("role") for m in final_step].count("tool") == 3
    assert any(p.get("summary") for p in _tool_payloads(final_step))


def test_multiple_tool_calls_in_one_model_turn():
    llm = ScriptedLLM(
        [
            ChatResult(
                content="",
                tool_calls=[
                    ToolCall(name="scrape_page", arguments={"url": "https://example.com/a"}),
                    ToolCall(name="scrape_page", arguments={"url": "https://example.com/b"}),
                ],
                message={"role": "assistant", "content": "", "tool_calls": []},
            ),
            _final("Two pages were scraped."),
        ]
    )
    agent = AgentOrchestrator(settings=_settings(), llm=llm, registry=_research_registry())
    result = agent.run("Read both example pages.")
    assert [t.name for t in result.traces] == ["scrape_page", "scrape_page"]
    assert result.traces[0].arguments["url"].endswith("/a")
    assert result.traces[1].arguments["url"].endswith("/b")
    payloads = _tool_payloads(llm.messages_seen[1])
    assert len(payloads) == 2
    assert "tool calling" in payloads[0]["data"]["text"]
    assert "16 GB" in payloads[1]["data"]["text"]


def test_scenario_b_report_workflow_across_iterations():
    llm = ScriptedLLM(
        [
            _tool_turn("search_web", {"query": "local llm research assistant"}),
            ChatResult(
                content="",
                tool_calls=[
                    ToolCall(name="scrape_page", arguments={"url": "https://example.com/a"}),
                    ToolCall(name="scrape_page", arguments={"url": "https://example.com/b"}),
                ],
                message={"role": "assistant", "content": "", "tool_calls": []},
            ),
            _tool_turn(
                "compare_sources",
                {
                    "source1": "Page A discusses Ollama tool calling in detail for local models.",
                    "source2": "Page B discusses 16 GB RAM limits when running qwen3:8b.",
                },
            ),
            _tool_turn(
                "generate_report",
                {
                    "sources": json.dumps(
                        [
                            {"title": "Ollama tools", "url": "https://example.com/a", "text": "tools"},
                            {"title": "RAM notes", "url": "https://example.com/b", "text": "ram"},
                        ]
                    )
                },
            ),
            _final("Here is the structured report from the collected sources."),
        ]
    )
    agent = AgentOrchestrator(settings=_settings(), llm=llm, registry=_research_registry())
    result = agent.run("Write a structured research report on local LLM assistants.")

    assert [t.name for t in result.traces] == [
        "search_web",
        "scrape_page",
        "scrape_page",
        "compare_sources",
        "generate_report",
    ]
    assert result.traces[-1].result["title"] == "Local research assistants"
    report_step = llm.messages_seen[4]
    names = [m.get("tool_name") for m in report_step if m.get("role") == "tool"]
    assert names == ["search_web", "scrape_page", "scrape_page", "compare_sources", "generate_report"]
    compare_payload = _tool_payloads(report_step)[-2]
    assert compare_payload["key_takeaways"] == ["Hardware matters"]


def test_max_rounds_still_returns_generate_report_if_present():
    llm = ScriptedLLM(
        [
            _tool_turn("search_web", {"query": "topic"}),
            _tool_turn(
                "generate_report",
                {"sources": "two collected sources with enough detail for a report"},
            ),
        ]
    )
    agent = AgentOrchestrator(
        settings=_settings(max_tool_rounds=2),
        llm=llm,
        registry=_research_registry(),
    )
    result = agent.run("Write a report now.")
    assert "Local research assistants" in result.reply
    assert "## Key Findings" in result.reply
    assert result.traces[-1].name == "generate_report"
