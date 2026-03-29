"""Procedural micro-expression sound library.

Generates short non-verbal sounds (chirps, hums, tones) that the robot
can play as quick emotional reactions — faster and more natural than
full TTS for simple acknowledgments.

Sounds are pure numpy arrays (int16, 24 kHz, mono).  Custom WAV files
in the ``sounds/`` directory override the procedural defaults.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000
_SOUNDS_DIR = Path(__file__).parent.parent / "sounds"


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def _envelope(n: int, attack_ms: float = 8, decay_ms: float = 40) -> np.ndarray:
    """Smooth amplitude envelope to avoid clicks."""
    env = np.ones(n, dtype=np.float64)
    a = min(int(SAMPLE_RATE * attack_ms / 1000), n // 2)
    d = min(int(SAMPLE_RATE * decay_ms / 1000), n // 2)
    if a > 0:
        env[:a] = np.linspace(0, 1, a)
    if d > 0:
        env[-d:] = np.linspace(1, 0, d)
    return env


def _tone(freq: float, dur_ms: float, amp: float = 0.3, harmonics: int = 3) -> np.ndarray:
    """Tone with natural harmonics and rolloff."""
    n = int(SAMPLE_RATE * dur_ms / 1000)
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    sig = np.zeros(n, dtype=np.float64)
    for h in range(1, harmonics + 1):
        sig += (amp / (h * h)) * np.sin(2 * np.pi * freq * h * t)
    return sig * _envelope(n)


def _sweep(f0: float, f1: float, dur_ms: float, amp: float = 0.3) -> np.ndarray:
    """Frequency sweep (chirp)."""
    n = int(SAMPLE_RATE * dur_ms / 1000)
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    freq = np.linspace(f0, f1, n)
    phase = np.cumsum(2 * np.pi * freq / SAMPLE_RATE)
    return amp * np.sin(phase) * _envelope(n)


def _silence(dur_ms: float) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * dur_ms / 1000), dtype=np.float64)


def _vibrato_tone(freq: float, dur_ms: float, amp: float = 0.25,
                  vib_hz: float = 5.0, vib_depth: float = 8.0) -> np.ndarray:
    """Tone with vibrato for organic feel."""
    n = int(SAMPLE_RATE * dur_ms / 1000)
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    mod = vib_depth * np.sin(2 * np.pi * vib_hz * t)
    phase = np.cumsum(2 * np.pi * (freq + mod) / SAMPLE_RATE)
    sig = amp * np.sin(phase) + (amp * 0.3) * np.sin(2 * phase)
    return sig * _envelope(n, attack_ms=20, decay_ms=80)


def _to_int16(signal: np.ndarray) -> np.ndarray:
    """Float64 [-1, 1] → int16, with clipping."""
    return np.clip(signal * 32767, -32768, 32767).astype(np.int16)


# ---------------------------------------------------------------------------
# Expression generators
# ---------------------------------------------------------------------------

def _gen_acknowledge() -> np.ndarray:
    """Short rising two-note — "mm-hmm"."""
    return np.concatenate([
        _tone(300, 100, amp=0.20),
        _silence(30),
        _tone(380, 130, amp=0.25),
    ])


def _gen_think() -> np.ndarray:
    """Sustained wavering hum — "hmm..."."""
    return _vibrato_tone(200, 500, amp=0.20, vib_hz=4.0, vib_depth=6.0)


def _gen_surprise() -> np.ndarray:
    """Quick upward sweep — "oh!"."""
    return np.concatenate([
        _sweep(250, 520, 120, amp=0.30),
        _tone(520, 80, amp=0.20),
    ])


def _gen_happy() -> np.ndarray:
    """Bright ascending double chirp."""
    return np.concatenate([
        _tone(400, 90, amp=0.25),
        _silence(25),
        _tone(520, 110, amp=0.30),
    ])


def _gen_sad() -> np.ndarray:
    """Slow descending tone."""
    return _sweep(320, 180, 350, amp=0.20)


def _gen_curious() -> np.ndarray:
    """Rising inflection — question tone."""
    return np.concatenate([
        _tone(280, 120, amp=0.20),
        _sweep(280, 420, 180, amp=0.25),
    ])


def _gen_laugh() -> np.ndarray:
    """Playful bouncing between two pitches."""
    parts = []
    for i in range(4):
        f = 380 if i % 2 == 0 else 450
        parts.append(_tone(f, 70, amp=0.22))
        parts.append(_silence(20))
    return np.concatenate(parts)


def _gen_concerned() -> np.ndarray:
    """Low wavering pulse."""
    return _vibrato_tone(170, 400, amp=0.20, vib_hz=3.5, vib_depth=10.0)


_GENERATORS = {
    "acknowledge": _gen_acknowledge,
    "think": _gen_think,
    "surprise": _gen_surprise,
    "happy": _gen_happy,
    "sad": _gen_sad,
    "curious": _gen_curious,
    "laugh": _gen_laugh,
    "concerned": _gen_concerned,
}


# ---------------------------------------------------------------------------
# WAV loader
# ---------------------------------------------------------------------------

def _load_wav(path: Path) -> np.ndarray | None:
    """Load a WAV file and return as float64 mono at SAMPLE_RATE."""
    try:
        import wave
        with wave.open(str(path), "rb") as wf:
            sr = wf.getframerate()
            n_ch = wf.getnchannels()
            sw = wf.getsampwidth()
            raw = wf.readframes(wf.getnframes())

        if sw == 2:
            pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
        elif sw == 4:
            pcm = np.frombuffer(raw, dtype=np.int32).astype(np.float64) / 2147483648.0
        else:
            logger.warning("Unsupported sample width %d in %s", sw, path)
            return None

        # Mono
        if n_ch > 1:
            pcm = pcm.reshape(-1, n_ch)[:, 0]

        # Resample to SAMPLE_RATE
        if sr != SAMPLE_RATE:
            from scipy.signal import resample
            n_out = int(len(pcm) * SAMPLE_RATE / sr)
            pcm = resample(pcm, n_out)

        return pcm
    except Exception as e:
        logger.warning("Failed to load WAV %s: %s", path, e)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class SoundLibrary:
    """Loads or generates micro-expression sounds.

    Custom WAVs in ``sounds/<expression>.wav`` override procedural defaults.
    """

    EXPRESSIONS = list(_GENERATORS.keys())

    def __init__(self, sounds_dir: Path | None = None) -> None:
        self._dir = sounds_dir or _SOUNDS_DIR
        self._cache: Dict[str, np.ndarray] = {}
        self._load_all()

    def _load_all(self) -> None:
        for name, gen_fn in _GENERATORS.items():
            wav_path = self._dir / f"{name}.wav"
            if wav_path.exists():
                pcm = _load_wav(wav_path)
                if pcm is not None:
                    self._cache[name] = _to_int16(pcm)
                    logger.info("Loaded custom sound: %s", wav_path)
                    continue
            # Procedural fallback
            self._cache[name] = _to_int16(gen_fn())

        # Also load any extra WAVs not in the generator list
        if self._dir.is_dir():
            for wav_path in self._dir.glob("*.wav"):
                name = wav_path.stem
                if name not in self._cache:
                    pcm = _load_wav(wav_path)
                    if pcm is not None:
                        self._cache[name] = _to_int16(pcm)
                        logger.info("Loaded extra sound: %s", name)

    def get(self, expression: str) -> np.ndarray | None:
        """Return int16 PCM array for the given expression, or None."""
        return self._cache.get(expression)

    def list_expressions(self) -> list[str]:
        """Return all available expression names."""
        return sorted(self._cache.keys())
