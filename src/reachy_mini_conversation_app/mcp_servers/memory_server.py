"""MCP server for persistent memory storage.

Run standalone (stdio transport)::

    python -m reachy_mini_conversation_app.mcp_servers.memory_server

Memories are stored in a local JSON file.  The path can be overridden via
the MEMORY_FILE environment variable.

TODOs:
- Replace JSON file store with SQLite or a vector store (ChromaDB / Qdrant)
  for semantic recall.
- Wire an embedding model so recall_memory() can do similarity search
  rather than keyword matching.
- Integrate with Home Assistant input_text helpers for cross-device
  persistence via python-homeassistant-api or webhooks.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("memory")

_DEFAULT_MEMORY_FILE = Path.home() / ".reachy_mini" / "memories.json"
_MEMORY_FILE = Path(os.environ.get("MEMORY_FILE", str(_DEFAULT_MEMORY_FILE)))


def _load() -> list[dict]:
    """Load memories from disk, returning an empty list on any error."""
    try:
        return json.loads(_MEMORY_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save(memories: list[dict]) -> None:
    """Persist memories to disk, creating parent directories as needed."""
    _MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    _MEMORY_FILE.write_text(json.dumps(memories, indent=2, ensure_ascii=False))


@mcp.tool()
async def store_memory(content: str, tags: Optional[List[str]] = None) -> str:
    """Store a new memory entry.

    Args:
        content: The text to remember.
        tags: Optional list of tags for later filtering.

    Returns:
        Confirmation string with the assigned memory ID.

    TODO: embed content for semantic retrieval.
    """
    memories = _load()
    entry = {
        "id": len(memories),
        "content": content,
        "tags": tags or [],
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }
    memories.append(entry)
    _save(memories)
    logger.info("Stored memory id=%d: %r", entry["id"], content[:60])
    return f"Stored as memory #{entry['id']}."


@mcp.tool()
async def recall_memory(query: str, limit: int = 5) -> List[dict]:
    """Retrieve memories matching a query.

    Args:
        query: Keyword or phrase to search for (case-insensitive substring
               match for now).
        limit: Maximum number of results to return.

    Returns:
        List of matching memory dicts (id, content, tags, timestamp).

    TODO: replace substring match with embedding similarity search.
    """
    memories = _load()
    q = query.lower()
    results = [m for m in memories if q in m.get("content", "").lower()]
    return results[:limit]


@mcp.tool()
async def list_memories(limit: int = 20) -> List[dict]:
    """List the most recent memories.

    Args:
        limit: Maximum number of memories to return (newest first).

    Returns:
        List of memory dicts.
    """
    memories = _load()
    return memories[-limit:][::-1]


if __name__ == "__main__":
    mcp.run(transport="stdio")
