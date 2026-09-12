from research_assistant.tools.registry import build_default_registry


def test_default_registry_includes_original_tools():
    registry = build_default_registry()

    names = registry.names()

    assert "compare_sources" in names
    assert "generate_report" in names
    assert "scrape_page" in names
    assert "search_web" in names
    assert "summarize_source" in names


def test_default_registry_includes_neuroscience_tools():
    registry = build_default_registry()

    names = registry.names()

    assert "search_pubmed" in names
    assert "brain_knowledge" in names
    assert "read_research_paper" in names
    assert "generate_citation" in names
    assert "retrieve_neuroscience_context" in names


def test_ollama_tool_schemas_exist():
    registry = build_default_registry()

    schemas = registry.ollama_tools()

    names = {
        schema["function"]["name"]
        for schema in schemas
    }

    assert "brain_knowledge" in names
    assert "search_pubmed" in names
    assert "retrieve_neuroscience_context" in names