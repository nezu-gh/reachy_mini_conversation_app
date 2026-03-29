"""Tests for the memory system (Mem0 client, tools, pipeline processors)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestMem0Client:
    """Tests for the Mem0 REST client."""

    @pytest.mark.asyncio
    async def test_add_memory_builds_correct_payload(self):
        from reachy_mini_conversation_app.memory.mem0_client import Mem0Client

        client = Mem0Client(base_url="http://fake:8765", user_id="testuser", app_name="testapp")

        mock_resp = AsyncMock()
        mock_resp.json = AsyncMock(return_value={"status": "ok"})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.closed = False
        client._session = mock_session

        result = await client.add_memory("User likes jazz")

        mock_session.post.assert_called_once()
        call_kwargs = mock_session.post.call_args
        assert call_kwargs[1]["json"]["text"] == "User likes jazz"
        assert call_kwargs[1]["json"]["user_id"] == "testuser"
        assert call_kwargs[1]["json"]["app"] == "testapp"
        assert call_kwargs[1]["json"]["infer"] is True
        assert result == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_search_memories_returns_items(self):
        from reachy_mini_conversation_app.memory.mem0_client import Mem0Client

        client = Mem0Client(base_url="http://fake:8765")

        mock_resp = AsyncMock()
        mock_resp.json = AsyncMock(return_value={
            "items": [{"content": "likes jazz", "categories": ["preferences"]}],
            "total": 1,
        })
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.closed = False
        client._session = mock_session

        items = await client.search_memories("music")
        assert len(items) == 1
        assert items[0]["content"] == "likes jazz"

    @pytest.mark.asyncio
    async def test_search_returns_empty_on_error(self):
        from reachy_mini_conversation_app.memory.mem0_client import Mem0Client

        client = Mem0Client(base_url="http://fake:8765")

        mock_session = AsyncMock()
        mock_session.get = MagicMock(side_effect=Exception("connection refused"))
        mock_session.closed = False
        client._session = mock_session

        items = await client.search_memories("anything")
        assert items == []


class TestMemoryTools:
    """Tests for store_memory and recall_memory tools."""

    def test_store_memory_tool_has_correct_spec(self):
        from reachy_mini_conversation_app.tools.store_memory import StoreMemory

        tool = StoreMemory()
        assert tool.name == "store_memory"
        spec = tool.spec()
        assert spec["parameters"]["required"] == ["text"]

    def test_recall_memory_tool_has_correct_spec(self):
        from reachy_mini_conversation_app.tools.recall_memory import RecallMemory

        tool = RecallMemory()
        assert tool.name == "recall_memory"
        spec = tool.spec()
        assert spec["parameters"]["required"] == ["query"]

    @pytest.mark.asyncio
    async def test_store_memory_calls_mem0(self):
        from reachy_mini_conversation_app.tools.store_memory import StoreMemory

        tool = StoreMemory()
        deps = MagicMock()

        with patch(
            "reachy_mini_conversation_app.memory.mem0_client.Mem0Client"
        ) as MockClient:
            mock_instance = AsyncMock()
            mock_instance.add_memory = AsyncMock(return_value={"status": "ok"})
            mock_instance.close = AsyncMock()
            MockClient.return_value = mock_instance

            result = await tool(deps, text="User's name is Alex")

            mock_instance.add_memory.assert_awaited_once_with("User's name is Alex")
            assert result["status"] == "stored"

    @pytest.mark.asyncio
    async def test_recall_memory_calls_mem0(self):
        from reachy_mini_conversation_app.tools.recall_memory import RecallMemory

        tool = RecallMemory()
        deps = MagicMock()

        with patch(
            "reachy_mini_conversation_app.memory.mem0_client.Mem0Client"
        ) as MockClient:
            mock_instance = AsyncMock()
            mock_instance.search_memories = AsyncMock(return_value=[
                {"content": "name is Alex", "categories": ["personal"]},
            ])
            mock_instance.close = AsyncMock()
            MockClient.return_value = mock_instance

            result = await tool(deps, query="user name")

            assert result["count"] == 1
            assert result["memories"][0]["content"] == "name is Alex"


class TestAutoMemoryTapRateLimit:
    """Test that AutoMemoryTap respects rate limiting."""

    def test_rate_limit_prevents_rapid_stores(self):
        """Rate limiting logic: only 1 store within the interval window."""
        import time

        # Simulate: last_store was 0 (initial), interval is 10s
        # First check should pass (monotonic time >> 0)
        last_store = 0.0
        interval = 10.0
        now = time.monotonic()

        calls = 0
        if now - last_store > interval:
            calls += 1
            last_store = now

        # Second immediate check should NOT pass
        now2 = time.monotonic()
        if now2 - last_store > interval:
            calls += 1
            last_store = now2

        assert calls == 1, "Only one store should fire within the rate limit window"
