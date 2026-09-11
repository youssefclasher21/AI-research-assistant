from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from research_assistant.config import Settings, load_settings
from research_assistant.llm.client import OllamaClient
from research_assistant.llm.prompts import SYSTEM_PROMPT
from research_assistant.llm.types import LLMClient
from research_assistant.reliability.validation import validate_user_input
from research_assistant.tools.registry import ToolRegistry, build_default_registry


@dataclass
class ToolTrace:
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]


@dataclass
class AgentResult:
    reply: str
    traces: list[ToolTrace] = field(default_factory=list)
    error: str | None = None

    def trace_markdown(self) -> str:
        if self.error and not self.traces:
            return f"**Input error:** {self.error}"
        if not self.traces:
            return "_No tools called. The model answered directly._"
        lines = ["### Tool trace"]
        for i, trace in enumerate(self.traces, start=1):
            ok = trace.result.get("ok")
            status = "ok" if ok else "error"
            lines.append(f"{i}. **`{trace.name}`** — {status}")
            lines.append(f"   - arguments: `{json.dumps(trace.arguments, ensure_ascii=False)}`")
            if ok:
                data = trace.result.get("data")
                if isinstance(data, list):
                    lines.append(f"   - results: {len(data)}")
                elif isinstance(data, dict) and "char_count" in data:
                    extra = " (truncated)" if data.get("truncated") else ""
                    title = data.get("title") or "(no title)"
                    lines.append(f"   - title: {title}")
                    lines.append(f"   - characters: {data.get('char_count')}{extra}")
                else:
                    lines.append("   - results: n/a")
            else:
                lines.append(f"   - error: {trace.result.get('error')}")
        return "\n".join(lines)


class AgentOrchestrator:
    def __init__(
        self,
        settings: Settings | None = None,
        llm: LLMClient | None = None,
        registry: ToolRegistry | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.llm = llm or OllamaClient(self.settings)
        self.registry = registry or build_default_registry()

    def run(
        self,
        user_text: str,
        history: list[dict[str, str]] | None = None,
    ) -> AgentResult:
        check = validate_user_input(user_text, self.settings.max_user_message_chars)
        if not check.ok:
            return AgentResult(reply=check.error or "Invalid input.", error=check.error)

        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in history or []:
            if isinstance(turn, dict):
                role = turn.get("role")
                content = turn.get("content")
            else:
                role = getattr(turn, "role", None)
                content = getattr(turn, "content", None)
            if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": check.text})

        tools = self.registry.ollama_tools()
        traces: list[ToolTrace] = []

        try:
            for _ in range(self.settings.max_tool_rounds):
                result = self.llm.chat(messages, tools)
                if result.tool_calls:
                    messages.append(result.message)
                    for call in result.tool_calls:
                        tool_result = self.registry.execute(call.name, call.arguments)
                        traces.append(
                            ToolTrace(
                                name=call.name,
                                arguments=call.arguments,
                                result=tool_result,
                            )
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_name": call.name,
                                "content": json.dumps(tool_result, ensure_ascii=False),
                            }
                        )
                    continue

                reply = (result.content or "").strip()
                if not reply:
                    reply = "I did not produce a response. Please try again."
                return AgentResult(reply=reply, traces=traces)

            return AgentResult(
                reply="Stopped after the maximum number of tool steps. Please try a simpler question.",
                traces=traces,
            )
        except Exception as exc:
            return AgentResult(
                reply=f"The assistant failed to complete this turn: {exc}",
                traces=traces,
                error=str(exc),
            )
