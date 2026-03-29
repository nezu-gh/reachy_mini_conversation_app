"""Tests for the procedural sound library."""

import numpy as np
import pytest

from reachy_mini_conversation_app.audio.sound_library import (
    SAMPLE_RATE,
    SoundLibrary,
    _GENERATORS,
    _envelope,
    _to_int16,
)


class TestGenerators:
    @pytest.mark.parametrize("name", list(_GENERATORS.keys()))
    def test_generator_produces_float64(self, name):
        audio = _GENERATORS[name]()
        assert audio.dtype == np.float64

    @pytest.mark.parametrize("name", list(_GENERATORS.keys()))
    def test_generator_nonzero(self, name):
        audio = _GENERATORS[name]()
        assert len(audio) > 0
        assert np.any(audio != 0)

    @pytest.mark.parametrize("name", list(_GENERATORS.keys()))
    def test_generator_in_range(self, name):
        audio = _GENERATORS[name]()
        assert np.all(audio >= -1.0)
        assert np.all(audio <= 1.0)


class TestToInt16:
    def test_converts_correctly(self):
        signal = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float64)
        result = _to_int16(signal)
        assert result.dtype == np.int16
        assert result[0] == 0
        assert result[3] == 32767
        assert result[4] == -32767  # np.clip keeps -32767, not -32768

    def test_clips_out_of_range(self):
        signal = np.array([2.0, -2.0], dtype=np.float64)
        result = _to_int16(signal)
        assert result[0] == 32767
        assert result[1] == -32768


class TestEnvelope:
    def test_shape(self):
        n = 1000
        env = _envelope(n)
        assert len(env) == n

    def test_attack_ramps_up(self):
        env = _envelope(1000, attack_ms=10, decay_ms=10)
        # First sample should be near 0, middle should be 1
        assert env[0] < 0.01
        assert env[500] == pytest.approx(1.0)

    def test_decay_ramps_down(self):
        env = _envelope(1000, attack_ms=10, decay_ms=10)
        assert env[-1] < 0.01


class TestSoundLibrary:
    def test_all_default_expressions_loaded(self):
        lib = SoundLibrary(sounds_dir=None)
        for name in _GENERATORS:
            sound = lib.get(name)
            assert sound is not None, f"Missing expression: {name}"
            assert sound.dtype == np.int16

    def test_list_expressions(self):
        lib = SoundLibrary(sounds_dir=None)
        expressions = lib.list_expressions()
        assert set(_GENERATORS.keys()).issubset(set(expressions))

    def test_unknown_expression_returns_none(self):
        lib = SoundLibrary(sounds_dir=None)
        assert lib.get("nonexistent_expression") is None

    def test_custom_wav_override(self, tmp_path):
        """A WAV file in the sounds dir should override the procedural default."""
        import struct
        import wave

        wav_path = tmp_path / "happy.wav"
        n_samples = 100
        with wave.open(str(wav_path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(struct.pack(f"<{n_samples}h", *([1000] * n_samples)))

        lib = SoundLibrary(sounds_dir=tmp_path)
        sound = lib.get("happy")
        assert sound is not None
        # Custom WAV should produce different output than procedural
        default_lib = SoundLibrary(sounds_dir=None)
        default_sound = default_lib.get("happy")
        assert not np.array_equal(sound, default_sound)
