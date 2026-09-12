from typing import Any

from research_assistant.tools.registry import ToolSpec


def generate_citation(
    title: str,
    author: str,
    year: str,
    journal: str = "",
    doi: str = ""
) -> dict[str, Any]:
    """
    Generate APA style citation for scientific papers.
    """

    try:
        citation = f"{author} ({year}). {title}."

        if journal:
            citation += f" {journal}."

        if doi:
            citation += f" https://doi.org/{doi}"

        return {
            "ok": True,
            "data": {
                "style": "APA",
                "citation": citation
            },
            "error": None
        }

    except Exception as exc:
        return {
            "ok": False,
            "data": None,
            "error": str(exc)
        }


CITATION_GENERATOR_SPEC = ToolSpec(
    name="generate_citation",
    description=(
        "Generate academic citations for neuroscience research papers "
        "using APA format. Use this when the user needs references "
        "or bibliography entries."
    ),
    parameters={
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Research paper title."
            },
            "author": {
                "type": "string",
                "description": "Paper author or authors."
            },
            "year": {
                "type": "string",
                "description": "Publication year."
            },
            "journal": {
                "type": "string",
                "description": "Journal name."
            },
            "doi": {
                "type": "string",
                "description": "Digital Object Identifier."
            }
        },
        "required": [
            "title",
            "author",
            "year"
        ]
    },
    handler=generate_citation
)