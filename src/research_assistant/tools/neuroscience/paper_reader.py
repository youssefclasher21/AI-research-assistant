from typing import Any

from research_assistant.tools.registry import ToolSpec


def read_paper(file_path: str) -> dict[str, Any]:
    """
    Extract text from a neuroscience research paper PDF.
    """

    try:
        from pypdf import PdfReader

        reader = PdfReader(file_path)

        text = ""

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        return {
            "ok": True,
            "data": {
                "file": file_path,
                "characters": len(text),
                "content": text[:5000]
            },
            "error": None
        }

    except Exception as exc:
        return {
            "ok": False,
            "data": None,
            "error": str(exc)
        }


PAPER_READER_SPEC = ToolSpec(
    name="read_research_paper",
    description=(
        "Read and extract text from a neuroscience research paper PDF. "
        "Use this tool when the user provides a scientific paper "
        "and wants analysis or summarization."
    ),
    parameters={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the PDF research paper."
            }
        },
        "required": [
            "file_path"
        ]
    },
    handler=read_paper
)