from __future__ import annotations

import json
from typing import Any

from ollama import Client

from research_assistant.config import Settings
from research_assistant.llm.types import ChatResult, LLMClient, ToolCall


def _message_to_dict(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        data = message.model_dump()
    elif isinstance(message, dict):
        data = dict(message)
    else:
        data = {
            "role": getattr(message, "role", "assistant"),
            "content": getattr(message, "content", "") or "",
        }
        if getattr(message, "tool_calls", None):
            data["tool_calls"] = message.tool_calls
    cleaned: dict[str, Any] = {
        "role": data.get("role") or "assistant",
        "content": data.get("content") or "",
    }
    tool_calls = data.get("tool_calls")
    if tool_calls:
        cleaned["tool_calls"] = tool_calls
    return cleaned


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


class OllamaClient(LLMClient):
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = Client(host=settings.ollama_host)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        format: str | None = None,
    ) -> ChatResult:
        response = self._client.chat(
            model=self._settings.ollama_model,
            messages=messages,
            tools=tools or None,
            think=False,
            format=format or None,
        )
        message = response.message
        tool_calls: list[ToolCall] = []
        raw_calls = getattr(message, "tool_calls", None) or []
        for call in raw_calls:
            function = getattr(call, "function", None)
            if function is None and isinstance(call, dict):
                function = call.get("function") or call
                name = function.get("name", "")
                arguments = _parse_arguments(function.get("arguments"))
            else:
                name = getattr(function, "name", "") or ""
                arguments = _parse_arguments(getattr(function, "arguments", None))
            if name:
                tool_calls.append(ToolCall(name=name, arguments=arguments))

        content = getattr(message, "content", None) or ""
        return ChatResult(
            content=content,
            tool_calls=tool_calls,
            message=_message_to_dict(message),
        )
