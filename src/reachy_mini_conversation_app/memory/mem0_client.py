"""Async HTTP client for the OpenMemory (Mem0) REST API.

Wraps the REST API at MEM0_BASE_URL to add, search, list, and delete
memories.  Uses aiohttp for non-blocking calls from the pipecat pipeline.

Env vars:
    MEM0_BASE_URL  — default http://192.168.178.155:8765
    MEM0_USER_ID   — default "default"
    MEM0_APP_NAME  — default "r3mn1"
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_BASE_URL = os.environ.get("MEM0_BASE_URL", "http://192.168.178.155:8765")
_DEFAULT_USER = os.environ.get("MEM0_USER_ID", "default")
_DEFAULT_APP = os.environ.get("MEM0_APP_NAME", "r3mn1")


class Mem0Client:
    """Thin async wrapper around the OpenMemory REST API."""

    def __init__(
        self,
        base_url: str = _BASE_URL,
        user_id: str = _DEFAULT_USER,
        app_name: str = _DEFAULT_APP,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.user_id = user_id
        self.app_name = app_name
        self._session: Any | None = None

    async def _get_session(self) -> Any:
        if self._session is None or self._session.closed:
            import aiohttp

            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def add_memory(
        self,
        text: str,
        *,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add a memory.  Mem0 auto-extracts facts when infer=True."""
        session = await self._get_session()
        payload = {
            "user_id": user_id or self.user_id,
            "text": text,
            "infer": True,
            "app": self.app_name,
        }
        if metadata:
            payload["metadata"] = metadata
        try:
            async with session.post(
                f"{self.base_url}/api/v1/memories/",
                json=payload,
            ) as resp:
                result = await resp.json()
                logger.info("Mem0 add_memory: %s", result)
                return result
        except Exception as exc:
            logger.warning("Mem0 add_memory failed: %s", exc)
            return {"error": str(exc)}

    async def search_memories(
        self,
        query: str,
        *,
        user_id: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Search memories by query string."""
        session = await self._get_session()
        params = {
            "user_id": user_id or self.user_id,
            "search_query": query,
            "size": limit,
        }
        try:
            async with session.get(
                f"{self.base_url}/api/v1/memories/",
                params=params,
            ) as resp:
                data = await resp.json()
                items = data.get("items", [])
                logger.debug("Mem0 search '%s': %d results", query, len(items))
                return items
        except Exception as exc:
            logger.warning("Mem0 search failed: %s", exc)
            return []

    async def list_memories(
        self,
        *,
        user_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List all memories for a user."""
        session = await self._get_session()
        params = {
            "user_id": user_id or self.user_id,
            "size": limit,
        }
        try:
            async with session.get(
                f"{self.base_url}/api/v1/memories/",
                params=params,
            ) as resp:
                data = await resp.json()
                return data.get("items", [])
        except Exception as exc:
            logger.warning("Mem0 list failed: %s", exc)
            return []

    async def delete_memory(
        self,
        memory_id: str,
        *,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Delete a specific memory by ID."""
        session = await self._get_session()
        payload = {
            "memory_ids": [memory_id],
            "user_id": user_id or self.user_id,
        }
        try:
            async with session.delete(
                f"{self.base_url}/api/v1/memories/",
                json=payload,
            ) as resp:
                result = await resp.json()
                logger.info("Mem0 delete: %s", result)
                return result
        except Exception as exc:
            logger.warning("Mem0 delete failed: %s", exc)
            return {"error": str(exc)}
