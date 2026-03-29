import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class RecallMemory(Tool):
    """Search memories for information about the user."""

    name = "recall_memory"
    description = (
        "Search your memory for facts about the user. "
        "Use at the start of conversations to recall their name/preferences, "
        "or when you need to check if you know something about them."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for (e.g., 'user name', 'music preferences', 'previous conversations')",
            },
        },
        "required": ["query"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Search memories via Mem0."""
        query = kwargs.get("query", "")
        if not query:
            return {"memories": [], "note": "no query provided"}

        from reachy_mini_conversation_app.memory.mem0_client import Mem0Client

        client = Mem0Client()
        try:
            items = await client.search_memories(query, limit=5)
            memories = [
                {"content": m.get("content", ""), "categories": m.get("categories", [])}
                for m in items
            ]
            return {"memories": memories, "count": len(memories)}
        except Exception as exc:
            logger.warning("recall_memory failed: %s", exc)
            return {"memories": [], "error": str(exc)}
        finally:
            await client.close()
