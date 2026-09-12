# NeuroResearch Assistant 🧠

An AI-powered neuroscience research assistant built with **Python, Ollama, Gradio, Tool Calling, and RAG**.

The project extends a general AI research assistant into a specialized neuroscience assistant capable of:
- Explaining neuroscience concepts
- Searching scientific literature
- Reading research papers
- Generating citations
- Retrieving knowledge from a local neuroscience database using RAG
- Combining multiple tools automatically through LLM tool calling

---

# Features

## 🤖 Local LLM with Ollama

- Uses Ollama as the main LLM backend
- Supports local models such as `qwen3:8b`
- The model decides when to use available tools

---

# 🛠️ Available Tools

## Research Tools

- `search_web`
  - Search the web for research information

- `scrape_page`
  - Extract content from web pages

- `summarize_source`
  - Generate summaries from sources

- `compare_sources`
  - Compare multiple research sources

- `generate_report`
  - Create structured research reports


## Neuroscience Tools

- `brain_knowledge`
  - Explain brain regions, neurotransmitters, and disorders

- `search_pubmed`
  - Search scientific neuroscience papers

- `read_research_paper`
  - Extract information from research PDFs

- `generate_citation`
  - Generate academic citations

- `retrieve_neuroscience_context`
  - Retrieve relevant neuroscience information using RAG

---

# Advanced AI Concept: Retrieval-Augmented Generation (RAG)

The project integrates RAG to improve reliability.

Instead of depending only on the LLM memory, the assistant retrieves relevant information from a local neuroscience knowledge base and uses it as additional context before generating answers.

Example:
