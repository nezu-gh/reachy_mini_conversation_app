import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class StoreMemory(Tool):
    """Store a fact or memory about the user for future conversations."""

    name = "store_memory"
    description = (
        "Remember something about the user for future conversations. "
        "Use when the user shares personal facts, preferences, or asks you to remember something. "
        "Pass the full context as text — the system extracts the key facts automatically."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text containing facts to remember (e.g., 'User said their name is Alex and they like jazz')",
            },
        },
        "required": ["text"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Store a memory via Mem0."""
        text = kwargs.get("text", "")
        if not text:
            return {"error": "no text provided"}

        from reachy_mini_conversation_app.memory.mem0_client import Mem0Client

        client = Mem0Client()
        try:
            result = await client.add_memory(text)
            return {"status": "stored", "result": result}
        except Exception as exc:
            logger.warning("store_memory failed: %s", exc)
            return {"error": str(exc)}
        finally:
            await client.close()
