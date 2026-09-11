from dataclasses import dataclass
from typing import Any, Callable

ToolHandler = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def ollama_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def names(self) -> list[str]:
        return sorted(self._tools)

    def ollama_tools(self) -> list[dict[str, Any]]:
        return [spec.ollama_schema() for spec in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
        spec = self._tools.get(name)
        if spec is None:
            return {
                "ok": False,
                "data": None,
                "error": f"Unknown tool: {name}",
            }
        args = arguments or {}
        if not isinstance(args, dict):
            return {
                "ok": False,
                "data": None,
                "error": "Tool arguments must be an object.",
            }
        try:
            result = spec.handler(**args)
        except TypeError as exc:
            return {
                "ok": False,
                "data": None,
                "error": f"Invalid arguments for {name}: {exc}",
            }
        except Exception as exc:
            return {
                "ok": False,
                "data": None,
                "error": f"{name} failed: {exc}",
            }
        if not isinstance(result, dict) or "ok" not in result:
            return {"ok": True, "data": result, "error": None}
        return result


def build_default_registry() -> ToolRegistry:
    from research_assistant.tools.scrape_page import SCRAPE_PAGE_SPEC
    from research_assistant.tools.search_web import SEARCH_WEB_SPEC

    registry = ToolRegistry()
    registry.register(SEARCH_WEB_SPEC)
    registry.register(SCRAPE_PAGE_SPEC)
    return registry
