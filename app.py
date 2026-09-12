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
    with gr.Blocks(title="NeuroResearch Assistant") as demo:
        gr.Markdown(
            "## NeuroResearch Assistant 🧠\n"
            "AI-powered neuroscience research assistant using "
            "Ollama, Tool Calling, and RAG.\n\n"
            "The model automatically selects tools when needed:\n"
            "- brain_knowledge\n"
            "- retrieve_neuroscience_context\n"
            "- search_pubmed\n"
            "- read_research_paper\n"
            "- generate_citation\n"
            "- web research tools"
        )

        chatbot = gr.Chatbot(label="Conversation")

        trace = gr.Markdown(
            value="_Tools used will appear here._",
            label="Agent Tool Trace"
        )

        with gr.Row():
            box = gr.Textbox(
                label="Research topic or question",
                placeholder=(
                    "e.g. Explain hippocampus and find "
                    "Alzheimer's research"
                ),
                scale=4,
            )

            send = gr.Button(
                "Send",
                variant="primary",
                scale=1,
            )

        send.click(
            respond,
            inputs=[box, chatbot],
            outputs=[chatbot, trace]
        ).then(
            lambda: "",
            outputs=box
        )

        box.submit(
            respond,
            inputs=[box, chatbot],
            outputs=[chatbot, trace]
        ).then(
            lambda: "",
            outputs=box
        )

    return demo


if __name__ == "__main__":
    build_ui().launch(share=True)
