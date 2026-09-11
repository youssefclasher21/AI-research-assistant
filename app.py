from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import gradio as gr

from research_assistant.agent.orchestrator import AgentOrchestrator

_orchestrator = AgentOrchestrator()


def _history_as_dicts(history) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    for item in history or []:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
        else:
            role = getattr(item, "role", None)
            content = getattr(item, "content", None)
        if role in {"user", "assistant"} and isinstance(content, str):
            turns.append({"role": role, "content": content})
    return turns


def respond(message: str, history: list | None):
    prior = _history_as_dicts(history)
    result = _orchestrator.run(message, prior)
    prior.append({"role": "user", "content": message})
    prior.append({"role": "assistant", "content": result.reply})
    return prior, result.trace_markdown()


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="AI Research Assistant") as demo:
        gr.Markdown(
            "## AI Research Assistant\n"
            "Send a message. Ollama decides whether to call `search_web`. "
            "There are no per-tool buttons."
        )
        chatbot = gr.Chatbot(label="Conversation")
        trace = gr.Markdown(value="_Tool trace will appear here._", label="Tool trace")
        with gr.Row():
            box = gr.Textbox(
                label="Research topic or question",
                placeholder="e.g. What is Ollama tool calling?",
                scale=4,
            )
            send = gr.Button("Send", variant="primary", scale=1)

        send.click(respond, inputs=[box, chatbot], outputs=[chatbot, trace]).then(
            lambda: "", outputs=box
        )
        box.submit(respond, inputs=[box, chatbot], outputs=[chatbot, trace]).then(
            lambda: "", outputs=box
        )
    return demo


if __name__ == "__main__":
    build_ui().launch()
