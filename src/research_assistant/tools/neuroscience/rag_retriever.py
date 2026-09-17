from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from research_assistant.tools.registry import ToolSpec


BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_PATH = BASE_DIR / "data"
CHROMA_PATH = BASE_DIR / "chroma_db"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "neuroscience_knowledge"

_embedding_model: SentenceTransformer | None = None
_collection = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model

    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return _embedding_model


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    text = text.strip()

    if not text:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def _get_collection():
    global _collection

    if _collection is not None:
        return _collection

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    _collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    _build_index_if_needed()

    return _collection


def _build_index_if_needed() -> None:
    if _collection is None:
        return

    existing = _collection.count()

    if existing > 0:
        return

    documents: list[str] = []
    ids: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for file in sorted(KNOWLEDGE_PATH.glob("*.txt")):
        content = file.read_text(encoding="utf-8")
        chunks = _chunk_text(content)

        for index, chunk in enumerate(chunks):
            documents.append(chunk)
            ids.append(f"{file.stem}-{index}")
            metadatas.append(
                {
                    "source": file.name,
                    "chunk": index,
                }
            )

    if not documents:
        return

    model = _get_embedding_model()
    embeddings = model.encode(
        documents,
        normalize_embeddings=True,
    ).tolist()

    _collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )


def retrieve_neuroscience_context(
    query: str,
    top_k: int = 3,
) -> dict[str, Any]:
    """Retrieve semantically relevant neuroscience chunks using embeddings and ChromaDB."""

    if not isinstance(query, str) or not query.strip():
        return {
            "ok": False,
            "data": None,
            "error": "Query cannot be empty.",
        }

    try:
        collection = _get_collection()

        if collection.count() == 0:
            return {
                "ok": True,
                "data": {
                    "message": "No neuroscience documents are available."
                },
                "error": None,
            }

        model = _get_embedding_model()

        query_embedding = model.encode(
            [query.strip()],
            normalize_embeddings=True,
        ).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=max(1, min(int(top_k), 5)),
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved_chunks = []

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):
            retrieved_chunks.append(
                {
                    "source": metadata.get("source", "unknown"),
                    "chunk": metadata.get("chunk", -1),
                    "similarity_distance": distance,
                    "content": document,
                }
            )

        return {
            "ok": True,
            "data": {
                "query": query,
                "top_k": len(retrieved_chunks),
                "retrieved_chunks": retrieved_chunks,
                "context": "\n\n--- Retrieved Chunk ---\n\n".join(
                    item["content"] for item in retrieved_chunks
                ),
            },
            "error": None,
        }

    except Exception as exc:
        return {
            "ok": False,
            "data": None,
            "error": str(exc),
        }


RAG_RETRIEVER_SPEC = ToolSpec(
    name="retrieve_neuroscience_context",
    description=(
        "Retrieve semantically relevant neuroscience knowledge from "
        "a local vector database using embeddings and similarity search. "
        "Use this tool when the user asks about neuroscience topics that "
        "should be answered using the project's collected knowledge."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Neuroscience topic or question to retrieve.",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of relevant chunks to retrieve.",
                "minimum": 1,
                "maximum": 5,
            },
        },
        "required": ["query"],
    },
    handler=retrieve_neuroscience_context,
)
