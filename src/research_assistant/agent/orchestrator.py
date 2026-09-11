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


def _compact_tool_result(result: dict[str, Any], max_chars: int) -> dict[str, Any]:
    encoded = json.dumps(result, ensure_ascii=False)
    if len(encoded) <= max_chars:
        return result
    compact = json.loads(encoded)
    data = compact.get("data")
    if isinstance(data, dict) and isinstance(data.get("text"), str):
        keep = max(500, max_chars - 800)
        data["text"] = data["text"][:keep]
        data["truncated_for_context"] = True
        compact["data"] = data
    encoded = json.dumps(compact, ensure_ascii=False)
    if len(encoded) > max_chars:
        return {
            "ok": compact.get("ok"),
            "error": compact.get("error"),
            "summary": compact.get("summary"),
            "title": compact.get("title"),
            "data": "Tool result truncated to fit the model context window.",
        }
    return compact


def _reply_from_report(result: dict[str, Any]) -> str:
    title = result.get("title") or "Research report"
    introduction = result.get("introduction") or ""
    findings = result.get("key_findings") or []
    conclusion = result.get("conclusion") or ""
    sources = result.get("sources") or []
    lines = [f"# {title}", "", "## Introduction", introduction, "", "## Key Findings"]
    if isinstance(findings, list):
        for item in findings:
            lines.append(f"- {item}")
    else:
        lines.append(str(findings))
    lines.extend(["", "## Conclusion", conclusion, "", "## Sources"])
    if isinstance(sources, list):
        for src in sources:
            if isinstance(src, dict):
                label = src.get("title") or "Source"
                url = src.get("url") or ""
                lines.append(f"- {label}{f' ({url})' if url else ''}")
            else:
                lines.append(f"- {src}")
    return "\n".join(lines).strip()


def _latest_report_reply(traces: list[ToolTrace]) -> str | None:
    for trace in reversed(traces):
        if trace.name == "generate_report" and trace.result.get("ok"):
            return _reply_from_report(trace.result)
    return None


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
                elif isinstance(data, dict) and "summary" in data:
                    points = data.get("key_points") or []
                    lines.append(f"   - summary chars: {len(data.get('summary') or '')}")
                    lines.append(f"   - key points: {len(points) if isinstance(points, list) else 0}")
                elif isinstance(data, dict) and (
                    "similarities" in data or "differences" in data or "key_takeaways" in data
                ):
                    lines.append(f"   - similarities: {len(data.get('similarities') or [])}")
                    lines.append(f"   - differences: {len(data.get('differences') or [])}")
                    lines.append(f"   - takeaways: {len(data.get('key_takeaways') or [])}")
                elif isinstance(data, dict) and "key_findings" in data:
                    findings = data.get("key_findings") or []
                    lines.append(f"   - report: {data.get('title') or '(untitled)'}")
                    lines.append(f"   - findings: {len(findings) if isinstance(findings, list) else 0}")
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
                        arguments = call.arguments if isinstance(call.arguments, dict) else {}
                        tool_result = self.registry.execute(call.name, arguments)
                        traces.append(
                            ToolTrace(
                                name=call.name,
                                arguments=arguments,
                                result=tool_result,
                            )
                        )
                        compact = _compact_tool_result(
                            tool_result, self.settings.max_tool_result_chars
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_name": call.name,
                                "content": json.dumps(compact, ensure_ascii=False),
                            }
                        )
                    continue

                reply = (result.content or "").strip()
                if not reply:
                    reply = _latest_report_reply(traces) or (
                        "I did not produce a response. Please try again."
                    )
                return AgentResult(reply=reply, traces=traces)

            report_reply = _latest_report_reply(traces)
            return AgentResult(
                reply=report_reply
                or "Stopped after the maximum number of tool steps. Please try a simpler question.",
                traces=traces,
            )
        except Exception as exc:
            return AgentResult(
                reply=f"The assistant failed to complete this turn: {exc}",
                traces=traces,
                error=str(exc),
            )
