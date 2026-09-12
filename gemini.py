from __future__ import annotations

import os
from typing import Any

from google import genai

from research_assistant.llm.types import ChatResult, LLMClient


class GeminiClient(LLMClient):
    def __init__(self, settings):
        self._model = settings.gemini_model
        self._client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY")
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        format: str | None = None,
    ) -> ChatResult:

        response = self._client.models.generate_content(
            model=self._model,
            contents=messages,
        )

        content = response.text or ""

        return ChatResult(
            content=content,
            tool_calls=[],
            message={
                "role": "assistant",
                "content": content,
            },
        )