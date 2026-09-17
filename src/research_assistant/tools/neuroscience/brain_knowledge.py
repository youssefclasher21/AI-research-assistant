from typing import Any

from research_assistant.tools.registry import ToolSpec


BRAIN_DATABASE = {
    "hippocampus": {
        "aliases": ["hippocampal", "memory center"],
        "function": (
            "A brain region important for memory formation, "
            "learning, and spatial navigation."
        ),
        "related_conditions": [
            "Alzheimer's disease",
            "Memory impairment"
        ]
    },

    "amygdala": {
        "aliases": ["emotion center"],
        "function": (
            "A brain structure involved in emotion processing, "
            "fear responses, and threat detection."
        ),
        "related_conditions": [
            "Anxiety disorders",
            "PTSD"
        ]
    },

    "dopamine": {
        "aliases": [
            "dopaminergic system",
            "dopamine pathway"
        ],
        "function": (
            "A neurotransmitter involved in reward, motivation, "
            "learning, and movement control."
        ),
        "related_conditions": [
            "Parkinson's disease",
            "Addiction"
        ]
    },

    "prefrontal cortex": {
        "aliases": [
            "pfc",
            "frontal cortex"
        ],
        "function": (
            "A brain region responsible for planning, "
            "decision making, attention, and cognitive control."
        ),
        "related_conditions": [
            "ADHD",
            "Executive dysfunction"
        ]
    }
}


def explain_brain_concept(topic: str) -> dict[str, Any]:
    """
    Explain neuroscience concepts using local knowledge.
    """

    if not topic:
        return {
            "ok": False,
            "data": None,
            "error": "Topic cannot be empty."
        }

    query = topic.lower().strip()

    for concept, info in BRAIN_DATABASE.items():

        searchable_terms = [
            concept,
            *info["aliases"]
        ]

        for term in searchable_terms:
            if term in query:
                return {
                    "ok": True,
                    "data": {
                        "concept": concept,
                        "function": info["function"],
                        "related_conditions": info["related_conditions"],
                        "source": "Local neuroscience knowledge base"
                    },
                    "error": None
                }

    return {
        "ok": True,
        "data": {
            "concept": topic,
            "message": (
                "No local information found. "
                "Use PubMed search tool for scientific research."
            )
        },
        "error": None
    }


BRAIN_KNOWLEDGE_SPEC = ToolSpec(
    name="brain_knowledge",
    description=(
"Simple predefined lookup for a small fixed set of neuroscience concepts. "
"Use this tool ONLY for a basic direct lookup when the concept is explicitly "
"covered by the local predefined dictionary. Do NOT use it for broad, "
"detailed, semantic, evidence-grounded, or knowledge-retrieval questions. "
"For those questions, use retrieve_neuroscience_context instead."
    ),
    parameters={
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": (
                    "The neuroscience concept to explain, "
                    "for example hippocampus or dopamine."
                )
            }
        },
        "required": [
            "topic"
        ]
    },
    handler=explain_brain_concept
)