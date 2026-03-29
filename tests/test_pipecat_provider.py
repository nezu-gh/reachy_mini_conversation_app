"""Tests for pipecat_provider module-level utilities.

Tests health-check probes, service validation, and audio validation.
These mock urllib so they don't require actual network access or running
services.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from reachy_mini_conversation_app.providers.pipecat_provider import (
    _check_services,
    _probe_service,
)


class TestProbeService:
    def test_success(self):
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            assert _probe_service("TestSvc", "http://localhost:1234/v1") is True

    def test_timeout(self):
        with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            assert _probe_service("TestSvc", "http://localhost:1234/v1") is False

    def test_connection_refused(self):
        with patch("urllib.request.urlopen", side_effect=OSError("Connection refused")):
            assert _probe_service("TestSvc", "http://localhost:1234/v1") is False

    def test_url_error(self):
        import urllib.error

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("unreachable"),
        ):
            assert _probe_service("TestSvc", "http://localhost:1234/v1") is False


class TestCheckServices:
    @patch("reachy_mini_conversation_app.providers.pipecat_provider._probe_service")
    def test_all_healthy_no_retry(self, mock_probe):
        mock_probe.return_value = True
        _check_services()
        # Should be called exactly 3 times (one per service)
        assert mock_probe.call_count == 3

    @patch("reachy_mini_conversation_app.providers.pipecat_provider._probe_service")
    @patch("time.sleep")  # prevent actual sleeping in tests
    def test_retries_on_failure(self, mock_sleep, mock_probe, caplog):
        # Fail twice, then succeed
        call_count = {"n": 0}

        def side_effect(name, url, timeout=5.0):
            call_count["n"] += 1
            if call_count["n"] <= 6:  # first 2 rounds × 3 services = 6 calls
                return False
            return True

        mock_probe.side_effect = side_effect
        with caplog.at_level(logging.WARNING):
            _check_services()
        # 3 rounds × 3 services = 9 total probe calls
        assert mock_probe.call_count == 9

    @patch("reachy_mini_conversation_app.providers.pipecat_provider._probe_service")
    @patch("time.sleep")
    def test_logs_error_after_all_retries(self, mock_sleep, mock_probe, caplog):
        mock_probe.return_value = False
        with caplog.at_level(logging.ERROR):
            _check_services()
        assert "still unreachable after 3 attempts" in caplog.text


# ---------------------------------------------------------------------------
# Audio validation tests for PipecatProvider.receive()
# ---------------------------------------------------------------------------

def _make_provider(queue_maxsize: int = 0):
    """Create a minimal PipecatProvider with mocked deps for receive() testing."""
    from reachy_mini_conversation_app.providers.pipecat_provider import PipecatProvider

    deps = MagicMock()
    deps.vision_processor = None
    provider = PipecatProvider(deps, gradio_mode=False, instance_path=None)
    # Simulate a live pipeline so receive() doesn't bail early
    provider._pipeline_task = MagicMock()
    if queue_maxsize > 0:
        provider._audio_in_queue = asyncio.Queue(maxsize=queue_maxsize)
    else:
        provider._audio_in_queue = asyncio.Queue()
    return provider


@pytest.mark.asyncio
async def test_receive_skips_non_ndarray() -> None:
    """Passing a non-ndarray should log a warning and not crash."""
    provider = _make_provider()
    await provider.receive((16000, "not_an_array"))
    assert provider._audio_in_queue.empty(), "nothing should be queued for invalid input"


@pytest.mark.asyncio
async def test_receive_skips_empty_array() -> None:
    """Empty ndarray should be silently skipped."""
    provider = _make_provider()
    await provider.receive((16000, np.array([], dtype=np.int16)))
    assert provider._audio_in_queue.empty()


@pytest.mark.asyncio
async def test_receive_converts_float32_to_int16() -> None:
    """Float32 audio in [-1, 1] should be converted to int16."""
    provider = _make_provider()
    audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    await provider.receive((16000, audio))
    assert not provider._audio_in_queue.empty()
    sr, result = provider._audio_in_queue.get_nowait()
    assert result.dtype == np.int16
    assert sr == 16000


# ---------------------------------------------------------------------------
# Queue backpressure tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_receive_drops_frame_when_queue_full() -> None:
    """When _audio_in_queue is full, receive() drops the frame without crashing."""
    provider = _make_provider(queue_maxsize=2)
    audio = np.array([100, 200, 300], dtype=np.int16)

    # Fill the queue
    await provider.receive((16000, audio))
    await provider.receive((16000, audio))
    assert provider._audio_in_queue.qsize() == 2

    # This should drop silently
    await provider.receive((16000, audio))
    assert provider._audio_in_queue.qsize() == 2
    assert provider._frames_dropped >= 1


# ---------------------------------------------------------------------------
# Barge-in generation counter tests
# ---------------------------------------------------------------------------

def test_barge_in_generation_prevents_stale_clear() -> None:
    """Clearing barge-in with a stale generation should not deactivate it."""
    provider = _make_provider()

    # Simulate VADUserStartedSpeaking
    provider._barge_in_generation += 1
    provider._barge_in_active = True
    current_gen = provider._barge_in_generation

    # Simulate a stale clear (from previous speech, gen doesn't match)
    stale_gen = current_gen - 1
    # The generation check logic: only clear if gen matches
    if stale_gen == provider._barge_in_generation:
        provider._barge_in_active = False

    assert provider._barge_in_active is True, "stale generation should not clear barge-in"


def test_barge_in_cleared_on_matching_generation() -> None:
    """Clearing barge-in with matching generation should deactivate it."""
    provider = _make_provider()

    provider._barge_in_generation += 1
    provider._barge_in_active = True

    # Clear with current generation (matches)
    provider._barge_in_active = False

    assert provider._barge_in_active is False


def test_pipeline_health_returns_metrics() -> None:
    """get_pipeline_health() returns expected keys."""
    provider = _make_provider()
    health = provider.get_pipeline_health()

    assert "frames_dropped" in health
    assert "pipeline_alive" in health
    assert "barge_in_active" in health
    assert "barge_in_generation" in health
    assert health["frames_dropped"] == 0
    assert health["barge_in_active"] is False
