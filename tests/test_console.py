"""Tests for the headless console stream."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fastrtc import AdditionalOutputs
from reachy_mini.media.media_manager import MediaBackend

from reachy_mini_conversation_app.console import LocalStream


def test_clear_audio_queue_prefers_clear_player_when_available() -> None:
    """Local GStreamer audio should use the lower-level player flush when available."""
    handler = MagicMock()
    audio = SimpleNamespace(
        clear_player=MagicMock(),
        clear_output_buffer=MagicMock(),
    )
    robot = SimpleNamespace(media=SimpleNamespace(audio=audio, backend=MediaBackend.LOCAL))
    stream = LocalStream(handler, robot)

    stream.clear_audio_queue()

    audio.clear_player.assert_called_once()
    audio.clear_output_buffer.assert_not_called()


def test_clear_audio_queue_uses_output_buffer_for_webrtc() -> None:
    """WebRTC audio should flush queued playback via the output buffer API."""
    handler = MagicMock()
    audio = SimpleNamespace(
        clear_player=MagicMock(),
        clear_output_buffer=MagicMock(),
    )
    robot = SimpleNamespace(media=SimpleNamespace(audio=audio, backend=MediaBackend.WEBRTC))
    stream = LocalStream(handler, robot)

    stream.clear_audio_queue()

    audio.clear_output_buffer.assert_called_once()
    audio.clear_player.assert_not_called()


def test_clear_audio_queue_falls_back_when_backend_is_unknown() -> None:
    """Unknown backends should still best-effort flush pending playback."""
    handler = MagicMock()
    audio = SimpleNamespace(clear_output_buffer=MagicMock())
    robot = SimpleNamespace(media=SimpleNamespace(audio=audio, backend=None))
    stream = LocalStream(handler, robot)

    stream.clear_audio_queue()

    audio.clear_output_buffer.assert_called_once()


# ---------------------------------------------------------------------------
# Error boundary tests
# ---------------------------------------------------------------------------

def _make_stream(handler=None):
    """Create a LocalStream with a mocked robot for testing loops."""
    handler = handler or MagicMock()
    audio = SimpleNamespace(
        clear_player=MagicMock(),
        clear_output_buffer=MagicMock(),
    )
    robot = SimpleNamespace(
        media=SimpleNamespace(
            audio=audio,
            backend=MediaBackend.LOCAL,
            get_input_audio_samplerate=MagicMock(return_value=16000),
            get_output_audio_samplerate=MagicMock(return_value=16000),
            get_audio_sample=MagicMock(return_value=None),
            push_audio_sample=MagicMock(),
        ),
    )
    return LocalStream(handler, robot)


@pytest.mark.asyncio
async def test_record_loop_survives_receive_exception() -> None:
    """A transient error in receive() must not crash the record loop."""
    handler = MagicMock()
    call_count = 0

    async def _receive(frame):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("transient failure")

    handler.receive = _receive
    stream = _make_stream(handler)

    # Return audio on every call so receive() is invoked
    stream._robot.media.get_audio_sample.return_value = np.zeros(160, dtype=np.int16)

    # Stop after a few iterations
    iteration = 0

    def _is_set():
        nonlocal iteration
        iteration += 1
        return iteration > 3

    stream._stop_event.is_set = _is_set

    await stream.record_loop()

    assert call_count >= 2, "record_loop should have continued after the exception"


@pytest.mark.asyncio
async def test_play_loop_survives_bad_audio_frame() -> None:
    """A non-ndarray audio payload must be skipped, not crash play_loop."""
    handler = MagicMock()
    emissions = [
        ("not_a_rate", "not_an_array"),  # bad frame
        (16000, np.zeros((1, 160), dtype=np.int16)),  # good frame
    ]
    emit_idx = 0

    async def _emit():
        nonlocal emit_idx
        if emit_idx < len(emissions):
            result = emissions[emit_idx]
            emit_idx += 1
            return result
        await asyncio.sleep(10)  # block until cancelled

    handler.emit = _emit
    handler._barge_in = False
    stream = _make_stream(handler)

    iteration = 0

    def _is_set():
        nonlocal iteration
        iteration += 1
        return iteration > 3

    stream._stop_event.is_set = _is_set

    await stream.play_loop()

    assert emit_idx == 2, "play_loop should have processed both frames"


@pytest.mark.asyncio
async def test_play_loop_skips_empty_audio() -> None:
    """Zero-length audio arrays must be skipped without error."""
    handler = MagicMock()

    async def _emit():
        return (24000, np.array([], dtype=np.int16))

    handler.emit = _emit
    handler._barge_in = False
    stream = _make_stream(handler)

    iteration = 0

    def _is_set():
        nonlocal iteration
        iteration += 1
        return iteration > 2

    stream._stop_event.is_set = _is_set

    # Should not raise
    await stream.play_loop()
    # push_audio_sample should never be called for empty frames
    stream._robot.media.push_audio_sample.assert_not_called()
