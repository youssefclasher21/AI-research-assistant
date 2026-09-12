from __future__ import annotations

import os
import json
from typing import Any

from google import genai
from google.genai import types

from research_assistant.llm.types import ChatResult, ToolCall, LLMClient


class GeminiClient(LLMClient):

    def __init__(self, settings):
        self._model = settings.gemini_model
        self._client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY")
        )

    def _convert_tools(self, tools: list[dict[str, Any]]):
        functions = []

        for tool in tools:
            functions.append(
                types.FunctionDeclaration(
                    name=tool["function"]["name"],
                    description=tool["function"].get("description", ""),
                    parameters=tool["function"].get(
                        "parameters",
                        {}
                    ),
                )
            )

        return [
            types.Tool(
                function_declarations=functions
            )
        ]

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        format: str | None = None,
    ) -> ChatResult:

        response = self._client.models.generate_content(
            model=self._model,
            contents=messages,
            config=types.GenerateContentConfig(
                tools=self._convert_tools(tools)
            ),
        )

        candidate = response.candidates[0]

        parts = candidate.content.parts

        for part in parts:
            if part.function_call:
                call = part.function_call

                return ChatResult(
                    content="",
                    message={
                        "role": "assistant",
                        "content": "",
                    },
                    tool_calls=[
                        ToolCall(
                            name=call.name,
                            arguments=dict(call.args),
                        )
                    ],
                )

        return ChatResult(
            content=response.text or "",
            message={
                "role": "assistant",
                "content": response.text or "",
            },
            tool_calls=[],
        )