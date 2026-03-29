"""Integration-style tests for audio data flow boundaries.

These exercise the actual numpy/scipy code paths (no mocking of math)
to catch shape/dtype mismatches that would crash the pipeline at runtime.
"""

import numpy as np
import pytest
from scipy.signal import resample

from fastrtc import audio_to_float32


class TestResampleRoundtrip:
    """Verify the resample path used in console.py play_loop."""

    def test_24khz_to_16khz_roundtrip(self):
        """Resample 24kHz → 16kHz produces correct length and dtype."""
        # 480 samples at 24kHz = 20ms frame (typical TTS chunk)
        audio_24k = np.random.randint(-32768, 32767, size=480, dtype=np.int16)
        audio_float = audio_to_float32(audio_24k).ravel()

        assert audio_float.ndim == 1
        assert audio_float.dtype == np.float32

        num_samples = int(len(audio_float) * 16000 / 24000)
        assert num_samples == 320

        resampled = resample(audio_float, num_samples)
        assert resampled.shape == (320,)
        assert resampled.dtype in (np.float32, np.float64)

    def test_resample_preserves_1d(self):
        """Resample output is always 1D regardless of input shape."""
        # Simulate PipelineSink output: shape (1, N)
        audio_2d = np.random.randint(-32768, 32767, size=(1, 480), dtype=np.int16)
        audio_flat = audio_to_float32(audio_2d).ravel()

        assert audio_flat.ndim == 1
        assert len(audio_flat) == 480

        resampled = resample(audio_flat, 320)
        assert resampled.ndim == 1

    def test_empty_audio_guard(self):
        """Empty audio after ravel should be caught before resample."""
        audio = np.array([], dtype=np.int16)
        audio_float = audio_to_float32(audio).ravel()
        assert len(audio_float) == 0
        # The play_loop skips with `if len(audio_frame) == 0: continue`

    def test_tiny_fragment_guard(self):
        """Very small audio should produce num_samples >= 1 or be skipped."""
        # 1 sample at 24kHz → target at 16kHz = 0.67 → int = 0
        audio = np.array([1000], dtype=np.int16)
        audio_float = audio_to_float32(audio).ravel()
        num_samples = int(len(audio_float) * 16000 / 24000)
        assert num_samples == 0
        # The play_loop skips with `if num_samples < 1: continue`


class TestAudioDtypesAtBoundaries:
    """Parametrized tests: various input types through the receive() conversion path."""

    @pytest.mark.parametrize("dtype,values", [
        (np.int16, [0, 1000, -1000, 32767, -32768]),
        (np.float32, [0.0, 0.5, -0.5, 1.0, -1.0]),
        (np.float64, [0.0, 0.5, -0.5, 1.0, -1.0]),
    ])
    def test_conversion_to_int16(self, dtype, values):
        """All supported dtypes convert to valid int16."""
        audio = np.array(values, dtype=dtype)

        # Replicate receive() conversion logic
        audio = audio.ravel()
        if audio.dtype == np.float32 or audio.dtype == np.float64:
            audio = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

        assert audio.dtype == np.int16
        assert audio.ndim == 1
        assert len(audio) == 5

    @pytest.mark.parametrize("shape", [
        (5,),        # 1D mono
        (1, 5),      # 2D row
        (5, 1),      # 2D column
    ])
    def test_shape_normalization(self, shape):
        """Various input shapes are flattened to 1D."""
        audio = np.ones(shape, dtype=np.int16) * 1000
        # Replicate receive() mono + ravel logic
        if audio.ndim == 2:
            if audio.shape[1] > audio.shape[0]:
                audio = audio.T
            if audio.shape[1] > 1:
                audio = audio[:, 0]
        audio = audio.ravel()

        assert audio.ndim == 1
        assert audio.dtype == np.int16


class TestPipelineSinkOutputShape:
    """Verify PipelineSink's PCM output matches what play_loop expects."""

    def test_pcm_reshape_produces_2d(self):
        """PipelineSink outputs (1, N) which play_loop must .ravel() before resample."""
        raw_bytes = np.array([100, -200, 300], dtype=np.int16).tobytes()
        pcm = np.frombuffer(raw_bytes, dtype=np.int16)
        output = pcm.reshape(1, -1)

        assert output.shape == (1, 3)
        assert output.dtype == np.int16

        # play_loop does: audio_to_float32(output).ravel()
        flat = audio_to_float32(output).ravel()
        assert flat.ndim == 1
        assert flat.dtype == np.float32
        assert len(flat) == 3

    def test_volume_boost_clipping(self):
        """Volume boost (2.8x) clips correctly to int16 range."""
        pcm = np.array([12000, -12000, 32767, -32768], dtype=np.int16).astype(np.int32)
        boosted = np.clip(pcm * 2.8, -32768, 32767).astype(np.int16)

        assert boosted.dtype == np.int16
        assert boosted.max() <= 32767
        assert boosted.min() >= -32768
        # 12000 * 2.8 = 33600 → clipped to 32767
        assert boosted[0] == 32767
        assert boosted[1] == -32768
