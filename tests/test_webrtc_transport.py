"""Tests for WebRTC transport module."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestWebRTCRoutes:
    """Test WebRTC route mounting and session management."""

    def test_mount_webrtc_routes_registers_endpoints(self):
        """mount_webrtc_routes should register /webrtc/* endpoints."""
        from fastapi import FastAPI

        app = FastAPI()
        handler = MagicMock()

        from reachy_mini_conversation_app.transports.webrtc_transport import (
            mount_webrtc_routes,
        )

        mount_webrtc_routes(app, handler)

        # Check routes were registered
        route_paths = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/webrtc" in route_paths
        assert "/webrtc/offer" in route_paths
        assert "/webrtc/sessions" in route_paths

    def test_mount_graceful_without_pipecat(self):
        """Should not crash if pipecat SmallWebRTC is not importable."""
        from fastapi import FastAPI

        app = FastAPI()
        handler = MagicMock()

        with patch.dict("sys.modules", {
            "pipecat.transports.smallwebrtc.connection": None,
            "pipecat.transports.smallwebrtc.request_handler": None,
            "pipecat.transports.smallwebrtc.transport": None,
        }):
            # Re-import to trigger the ImportError path
            import importlib

            from reachy_mini_conversation_app.transports import webrtc_transport

            importlib.reload(webrtc_transport)
            # Should not raise
            webrtc_transport.mount_webrtc_routes(app, handler)


class TestSessionManagement:
    """Test concurrent session limiting."""

    def test_max_clients_default(self):
        from reachy_mini_conversation_app.transports.webrtc_transport import MAX_CLIENTS

        assert MAX_CLIENTS >= 1

    @pytest.mark.asyncio
    async def test_session_tracking(self):
        """Active sessions dict should track connections."""
        from reachy_mini_conversation_app.transports.webrtc_transport import (
            _active_sessions,
            _session_lock,
        )

        async with _session_lock:
            _active_sessions["test-1"] = "conn1"
            assert len(_active_sessions) == 1
            _active_sessions.pop("test-1")
            assert len(_active_sessions) == 0


class TestWebRTCUI:
    """Test the WebRTC HTML UI file exists and is valid."""

    def test_webrtc_html_exists(self):
        from pathlib import Path

        html_path = (
            Path(__file__).parent.parent
            / "src"
            / "reachy_mini_conversation_app"
            / "static"
            / "webrtc.html"
        )
        assert html_path.exists(), f"WebRTC UI not found at {html_path}"

    def test_webrtc_html_contains_key_elements(self):
        from pathlib import Path

        html_path = (
            Path(__file__).parent.parent
            / "src"
            / "reachy_mini_conversation_app"
            / "static"
            / "webrtc.html"
        )
        content = html_path.read_text()
        assert "RTCPeerConnection" in content
        assert "/webrtc/offer" in content
        assert "getUserMedia" in content
        assert "Reachy Mini" in content
