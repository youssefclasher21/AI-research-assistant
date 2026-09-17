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


def test_retrieve_neuroscience_context_returns_hippocampus_for_memory_query():
    result = retrieve_neuroscience_context(
        "what part of the brain is involved in memory formation?"
    )

    assert result["ok"] is True
    chunks = result["data"]["retrieved_chunks"]
    assert len(chunks) > 0
    sources = [c["source"] for c in chunks]
    assert "hippocampus.txt" in sources


def test_retrieve_neuroscience_context_respects_top_k_one():
    result = retrieve_neuroscience_context("dopamine", top_k=1)

    assert result["ok"] is True
    assert len(result["data"]["retrieved_chunks"]) == 1


def test_retrieve_neuroscience_context_empty_query_fails():
    result = retrieve_neuroscience_context("")

    assert result["ok"] is False
    assert result["error"]
