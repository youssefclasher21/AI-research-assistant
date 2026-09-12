from research_assistant.tools.registry import build_default_registry
from research_assistant.tools.neuroscience.brain_knowledge import (
    explain_brain_concept,
)
from research_assistant.tools.neuroscience.rag_retriever import (
    retrieve_neuroscience_context,
)


def test_neuroscience_tools_registered():
    registry = build_default_registry()

    tools = registry.names()

    assert "search_pubmed" in tools
    assert "brain_knowledge" in tools
    assert "read_research_paper" in tools
    assert "generate_citation" in tools
    assert "retrieve_neuroscience_context" in tools


def test_brain_knowledge():
    result = explain_brain_concept("hippocampus")

    assert result["ok"] is True
    assert "memory" in result["data"]["function"].lower()


def test_rag_retriever():
    result = retrieve_neuroscience_context(
        "hippocampus memory"
    )

    assert result["ok"] is True
    assert "context" in result["data"]