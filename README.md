# NeuroResearch Assistant 🧠

An AI-powered neuroscience research assistant built with **Python, Ollama, Gradio, agentic tool-calling, and Retrieval-Augmented Generation (RAG)**.

The assistant extends a general-purpose research assistant into a neuroscience-focused tool capable of explaining concepts, searching scientific literature, reading papers, generating citations, and retrieving grounded context from a local knowledge base — combining these capabilities automatically through LLM tool-calling.

---

## Key Features

- **Local LLM via Ollama** — runs entirely on a local model (default: `qwen3:8b`), no external API required
- **Agentic tool-calling** — the model decides which tool(s) to use based on the user's question, rather than following a fixed script
- **RAG-based retrieval** — semantic search over a local neuroscience knowledge base using sentence embeddings and a vector database
- **10 registered tools** spanning general web research and neuroscience-specific tasks
- **Tool execution tracing** — every tool call, its arguments, and result are logged and shown alongside the conversation in the UI
- **Automated test suite** — 48 passing tests covering the agent loop, tool registry, and individual tools, including RAG retrieval relevance

---

## Architecture / Agent Workflow

1. The user submits a question through the Gradio chat interface.
2. `AgentOrchestrator` sends the conversation (plus the full tool registry) to the LLM via Ollama.
3. If the model requests a tool call, the orchestrator executes it through `ToolRegistry`, appends the result back into the conversation as a `tool` message, and loops.
4. This repeats — up to a configurable maximum (`max_tool_rounds`) — until the model produces a final answer grounded in the tool results.
5. Tool results are automatically compacted if they exceed the model's context window, to avoid truncation errors.

This is a general **agent loop**, not a hardcoded pipeline: the model chooses which of the 10 available tools to call, in what order, and how many times, based on the system prompt's tool-selection guidance.

---

## RAG Implementation

The `retrieve_neuroscience_context` tool implements Retrieval-Augmented Generation over a local neuroscience document set:

- **Embedding model:** `all-MiniLM-L6-v2` (via `sentence-transformers`), with normalized embeddings
- **Vector store:** ChromaDB (`PersistentClient`), using **cosine similarity** for retrieval
- **Chunking:** character-based sliding window (800 chars per chunk, 120-char overlap)
- **Indexing:** documents are automatically chunked, embedded, and indexed into ChromaDB the first time the tool runs; the collection persists on disk afterward
- **Retrieval:** top-k semantic search (default `k=3`, capped at 5), with retrieved chunks assembled into a single context string
- **Agent integration:** the LLM autonomously chooses to call this tool when a question requires knowledge grounded in the local dataset, rather than relying on model memory alone

> **Note:** The current knowledge base is a small, curated set of local `.txt` files (covering topics such as the hippocampus, dopamine, and Alzheimer's disease) intended to demonstrate the RAG pipeline end-to-end. It is not a large-scale or production knowledge base.

---

## Available Tools

**General research tools**
- `search_web` — searches the web for information
- `scrape_page` — extracts readable text from a given URL
- `summarize_source` — summarizes source text into key points
- `compare_sources` — compares two sources for similarities/differences
- `generate_report` — compiles a structured research report from collected material

**Neuroscience tools**
- `brain_knowledge` — simple predefined explanations of basic neuroscience concepts
- `retrieve_neuroscience_context` — semantic RAG retrieval from the local neuroscience knowledge base
- `search_pubmed` — searches PubMed for scientific literature
- `read_research_paper` — extracts information from a research paper
- `generate_citation` — generates citations from sources

---

## Setup & Run

**Prerequisites:** Python 3.12+, Ollama installed and running locally.

Clone the repository:
git clone https://github.com/youssefclasher21/AI-research-assistant.git
cd AI-research-assistant

Create and activate a virtual environment:
python -m venv .venv
.venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt

Pull the local model:
ollama pull qwen3:8b

Run the app:
python app.py

The Gradio interface will launch locally (and generate a temporary public share link).

---

## Environment Variables

Set these in a `.env` file (or use the defaults):

OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3:8b

---

## Testing

python -m pytest -v

**48 tests passing**, covering the agent orchestration loop, tool registry, individual tools (web search, scraping, summarization, report generation), and neuroscience tools including RAG retrieval — with dedicated tests verifying that semantically relevant chunks are actually returned for a given query, not just that the pipeline runs without errors.

---

## Example Usage

**Input:** "Explain the hippocampus and find recent Alzheimer's research"

**What happens:**
1. The agent calls `retrieve_neuroscience_context` to pull relevant local context on the hippocampus
2. The agent calls `search_pubmed` (or `search_web`) to find recent Alzheimer's-related literature
3. The agent combines both results into a single grounded response, with the tool trace panel showing exactly which tools were called and what they returned