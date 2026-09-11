# AI Research Assistant (Team 1)

First implementation slice only: Gradio chat, Ollama tool calling, and `search_web`.

## Setup (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Confirm Ollama is running and `qwen3:8b` is installed:

```powershell
ollama list
```

## Run the app

```powershell
python app.py
```

Open the local Gradio URL. Use the single chat box — the model decides whether to call `search_web`. The tool-trace panel shows calls when they happen.

## Tests

```powershell
pytest
```

Tests use a mocked LLM. They do not require Ollama or the network.
