from pathlib import Path
from typing import Any

from research_assistant.tools.registry import ToolSpec


KNOWLEDGE_PATH = Path(
    "src/research_assistant/tools/neuroscience/data"
)


def retrieve_neuroscience_context(
    query: str
) -> dict[str, Any]:
    """
    Retrieve relevant neuroscience information
    from local knowledge files.
    """

    if not query:
        return {
            "ok": False,
            "data": None,
            "error": "Query cannot be empty."
        }

    query_words = query.lower().split()

    results = []

    try:
        for file in KNOWLEDGE_PATH.glob("*.txt"):

            content = file.read_text(
                encoding="utf-8"
            )

            text = content.lower()

            score = 0

            filename = file.stem.lower()

            # Give higher priority to matching filename
            for word in query_words:

                if word in filename:
                    score += 5

                if word in text:
                    score += 1

            if score > 0:
                results.append(
                    {
                        "file": file.name,
                        "score": score,
                        "content": content
                    }
                )

        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        if results:
            return {
                "ok": True,
                "data": {
                    "source": results[0]["file"],
                    "context": results[0]["content"]
                },
                "error": None
            }

        return {
            "ok": True,
            "data": {
                "message": "No relevant information found."
            },
            "error": None
        }

    except Exception as exc:
        return {
            "ok": False,
            "data": None,
            "error": str(exc)
        }


RAG_RETRIEVER_SPEC = ToolSpec(
    name="retrieve_neuroscience_context",
    description=(
        "Retrieve relevant neuroscience knowledge "
        "from a local research database."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Neuroscience topic to search."
            }
        },
        "required": [
            "query"
        ]
    },
    handler=retrieve_neuroscience_context
)