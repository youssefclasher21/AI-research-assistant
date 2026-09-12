from typing import Any
import requests

from research_assistant.tools.registry import ToolSpec


def search_pubmed(query: str, max_results: int = 5) -> dict[str, Any]:
    """
    Search PubMed for neuroscience research papers.
    """

    try:
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": max_results,
        }

        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        data = response.json()

        papers = data["esearchresult"]["idlist"]

        return {
            "ok": True,
            "data": {
                "query": query,
                "paper_ids": papers
            },
            "error": None
        }

    except Exception as exc:
        return {
            "ok": False,
            "data": None,
            "error": str(exc)
        }


PUBMED_SEARCH_SPEC = ToolSpec(
    name="search_pubmed",
    description=(
        "Search scientific neuroscience papers using PubMed. "
        "Use this when the user asks for research papers, "
        "studies, or scientific evidence."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The neuroscience topic to search for."
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of papers to return."
            }
        },
        "required": [
            "query"
        ]
    },
    handler=search_pubmed
)