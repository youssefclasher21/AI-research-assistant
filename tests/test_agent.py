from typing import Any

from research_assistant.agent.orchestrator import AgentOrchestrator
from research_assistant.config import Settings
from research_assistant.llm.types import ChatResult, LLMClient, ToolCall
from research_assistant.tools.registry import ToolRegistry, ToolSpec
from research_assistant.tools.search_web import SEARCH_WEB_SPEC


class ScriptedLLM(LLMClient):
    def __init__(self, script: list[ChatResult]) -> None:
        self._script = list(script)
        self.calls = 0
        self.messages_seen: list[list[dict[str, Any]]] = []

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ChatResult:
        self.calls += 1
        self.messages_seen.append(messages)
        if not self._script:
            raise AssertionError("LLM was called more times than scripted")
        return self._script.pop(0)


def _settings() -> Settings:
    return Settings(ollama_host="http://127.0.0.1:11434", ollama_model="qwen3:8b")


def _registry_with_stub_search() -> ToolRegistry:
    def stub_search(**kwargs: Any) -> dict[str, Any]:
        query = kwargs.get("query", "")
        return {
            "ok": True,
            "data": [
                {
                    "title": "Stub result",
                    "url": "https://example.com",
                    "snippet": f"Result for {query}",
                }
            ],
            "error": None,
        }

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name=SEARCH_WEB_SPEC.name,
            description=SEARCH_WEB_SPEC.description,
            parameters=SEARCH_WEB_SPEC.parameters,
            handler=stub_search,
        )
    )
    return registry


def test_empty_user_message_does_not_call_llm():
    llm = ScriptedLLM([])
    agent = AgentOrchestrator(settings=_settings(), llm=llm, registry=_registry_with_stub_search())
    result = agent.run("   ")
    assert result.error
    assert llm.calls == 0
    assert "enter" in result.reply.lower()


def test_agent_executes_search_web_when_llm_requests_it():
    llm = ScriptedLLM(
        [
            ChatResult(
                content="",
                tool_calls=[ToolCall(name="search_web", arguments={"query": "ollama tools"})],
                message={
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search_web",
                                "arguments": {"query": "ollama tools"},
                            }
                        }
                    ],
                },
            ),
            ChatResult(
                content="I found a stub source about Ollama tools.",
                tool_calls=[],
                message={"role": "assistant", "content": "I found a stub source about Ollama tools."},
            ),
        ]
    )
    agent = AgentOrchestrator(settings=_settings(), llm=llm, registry=_registry_with_stub_search())
    result = agent.run("Look up Ollama tool calling.")

    assert llm.calls == 2
    assert result.traces[0].name == "search_web"
    assert result.traces[0].result["ok"] is True
    assert "stub source" in result.reply.lower()

    roles = [m["role"] for m in llm.messages_seen[1]]
    assert roles[0] == "system"
    assert "user" in roles
    assert "assistant" in roles
    assert "tool" in roles


def test_agent_executes_scrape_page_when_llm_requests_it():
    def stub_scrape(**kwargs: Any) -> dict[str, Any]:
        return {
            "ok": True,
            "data": {
                "status": 200,
                "url": kwargs.get("url", ""),
                "title": "Stub page",
                "text": "Cleaned page text.",
                "char_count": 18,
                "truncated": False,
            },
            "error": None,
        }

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="scrape_page",
            description="Scrape a page",
            parameters={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
            handler=stub_scrape,
        )
    )
    llm = ScriptedLLM(
        [
            ChatResult(
                content="",
                tool_calls=[
                    ToolCall(name="scrape_page", arguments={"url": "https://example.com/article"})
                ],
                message={"role": "assistant", "content": "", "tool_calls": []},
            ),
            ChatResult(
                content="The page discusses cleaned page text.",
                tool_calls=[],
                message={"role": "assistant", "content": "The page discusses cleaned page text."},
            ),
        ]
    )
    agent = AgentOrchestrator(settings=_settings(), llm=llm, registry=registry)
    result = agent.run("Read https://example.com/article")

    assert result.traces[0].name == "scrape_page"
    assert result.traces[0].result["ok"] is True
    assert "cleaned page text" in result.reply.lower()
    assert "tool" in [m["role"] for m in llm.messages_seen[1]]


def test_agent_can_answer_without_tools():
    llm = ScriptedLLM(
        [
            ChatResult(
                content="Hello — how can I help you research?",
                tool_calls=[],
                message={"role": "assistant", "content": "Hello — how can I help you research?"},
            )
        ]
    )
    agent = AgentOrchestrator(settings=_settings(), llm=llm, registry=_registry_with_stub_search())
    result = agent.run("Hello")
    assert result.traces == []
    assert "hello" in result.reply.lower()
    assert "No tools called" in result.trace_markdown()
