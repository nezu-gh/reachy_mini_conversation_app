"""Pipecat-based local conversation provider.

Bridges fastrtc's AsyncStreamHandler audio contract to a pipecat Pipeline
running three OpenAI-compatible ``/v1`` Docker services on a local VM:

  STT : Qwen-ASR  → ASR_BASE_URL  (default :8015/v1)
  LLM : Qwen3.5   → LLM_BASE_URL  (default :3443/v1)
  TTS : Qwen3-TTS → TTS_BASE_URL  (default :7034/v1)
  VAD : SileroVADAnalyzer (runs locally, no endpoint)

Audio flows:
  fastrtc receive() → _audio_in_queue → PipelineSource → [VAD → STT → LLM → TTS]
                                                                              ↓
  fastrtc emit()    ← _output_queue  ← PipelineSink  ←  audio + transcripts

Speech energy for head-wobble is fed from TTS output audio into HeadWobbler
before it reaches the output queue.
"""

from __future__ import annotations
import os
import re
import json
import base64
import random
import asyncio
import logging
from typing import TYPE_CHECKING, Any, List, Tuple, Optional

import numpy as np
from fastrtc import AdditionalOutputs, wait_for_item
from numpy.typing import NDArray
from scipy.signal import resample

from reachy_mini_conversation_app.providers.base import ConversationProvider


if TYPE_CHECKING:
    from reachy_mini_conversation_app.tools.core_tools import ToolDependencies

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipecat availability guard
# ---------------------------------------------------------------------------
try:
    import pipecat  # noqa: F401

    _PIPECAT_AVAILABLE = True
except ImportError:
    _PIPECAT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Endpoint configuration (read from env; overridable per-instance via .env)
# ---------------------------------------------------------------------------
_DEFAULT_VM = os.environ.get("LOCAL_VM_IP", "192.168.178.155")

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", f"http://{_DEFAULT_VM}:3443/v1")
LLM_MODEL = os.environ.get("MODEL_NAME", "/models/Qwen3.5-35B-A3B-GGUF/IQ4_XS/Qwen3.5-35B-A3B-IQ4_XS-00001-of-00002.gguf")

TTS_BASE_URL = os.environ.get("TTS_BASE_URL", f"http://{_DEFAULT_VM}:7034/v1")
TTS_MODEL = os.environ.get("TTS_MODEL", "qwen3-tts")

ASR_BASE_URL = os.environ.get("ASR_BASE_URL", f"http://{_DEFAULT_VM}:8015/v1")
ASR_MODEL = os.environ.get("ASR_MODEL", "Qwen/Qwen3-ASR-0.6B")

# ---------------------------------------------------------------------------
# Audio constants
# ---------------------------------------------------------------------------
# fastrtc uses 24 kHz; pipecat's Silero VAD requires 16 kHz.
# PipelineParams defaults already match: audio_in=16000, audio_out=24000.
FASTRTC_SAMPLE_RATE = 24000
PIPELINE_SAMPLE_RATE = 16000

# ---------------------------------------------------------------------------
# Testable pure functions (extracted from inner FrameProcessor classes)
# ---------------------------------------------------------------------------

# CJK Unicode ranges for ASR noise detection
_CJK_RE = re.compile(
    r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af\u3400-\u4dbf]'
)

_ASR_MIN_LENGTH = 2  # minimum chars for a valid transcript


def _is_noise(text: str) -> bool:
    """Return True if a transcript looks like noise rather than real speech.

    Qwen3-ASR hallucinates CJK characters on ambient sound (fans, hum).
    Also rejects empty, too-short, and punctuation-only transcripts.
    """
    if not text or len(text) < _ASR_MIN_LENGTH:
        return True
    cjk_count = len(_CJK_RE.findall(text))
    if cjk_count > 0 and cjk_count / len(text) > 0.3:
        return True
    if not any(c.isalnum() for c in text):
        return True
    return False


def _trim_context(messages: list[dict], max_turns: int = 40) -> list[dict] | None:
    """Trim old non-system messages if the context exceeds *max_turns*.

    Returns the trimmed list, or ``None`` if no trimming was needed.
    """
    non_system = [m for m in messages if m.get("role") != "system"]
    if len(non_system) <= max_turns:
        return None
    system_msgs = [m for m in messages if m.get("role") == "system"]
    return system_msgs + non_system[-max_turns:]


# Regex to catch tool calls written as plain text by the LLM
_TOOL_TEXT_RE = re.compile(
    r'(micro_expression|play_emotion|move_head|dance|stop_dance|stop_emotion|do_nothing)'
    r'\(([^)]*)\)',
)
# Markdown-style: "*Micro-expression: laugh*"
_MARKDOWN_EXPR_RE = re.compile(
    r'\*[Mm]icro[- _]expression:\s*(\w+)\*',
)


def _extract_inline_tool_calls(text: str) -> list[tuple[str, str]]:
    """Extract tool-call-like patterns from LLM output text.

    Returns a list of ``(function_name, raw_arg)`` tuples found in *text*.
    Does **not** dispatch them — the caller is responsible for that.
    """
    results: list[tuple[str, str]] = []
    for m in _TOOL_TEXT_RE.finditer(text):
        results.append((m.group(1), m.group(2).strip().strip("'\"")))
    for m in _MARKDOWN_EXPR_RE.finditer(text):
        results.append(("micro_expression", m.group(1).lower()))
    return results


def _strip_inline_tool_calls(text: str) -> str:
    """Remove tool-call patterns from *text* (for TTS output)."""
    cleaned = _TOOL_TEXT_RE.sub("", text)
    return _MARKDOWN_EXPR_RE.sub("", cleaned).strip()


def _probe_service(name: str, base_url: str, timeout: float = 2.0) -> bool:
    """HTTP GET ``base_url/models`` to check if a service is reachable."""
    import urllib.request
    import urllib.error

    url = f"{base_url}/models"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout):
            pass
        logger.info("Health check OK: %s @ %s", name, base_url)
        return True
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        logger.warning("Health check FAILED: %s @ %s — %s", name, base_url, exc)
        return False


def _check_services() -> None:
    """Pre-flight health check for all three VM services.

    Retries up to 3 times with 2s backoff.  Logs clear errors but does
    **not** abort — the pipeline may still recover if the service comes
    up later.  Designed to run in a background thread.
    """
    import time as _time

    services = [
        ("ASR", ASR_BASE_URL),
        ("LLM", LLM_BASE_URL),
        ("TTS", TTS_BASE_URL),
    ]
    for attempt in range(1, 4):
        failed = [(n, u) for n, u in services if not _probe_service(n, u)]
        if not failed:
            return
        if attempt < 3:
            logger.warning(
                "Service health check attempt %d/3: %s unreachable — retrying in 2s",
                attempt, ", ".join(n for n, _ in failed),
            )
            _time.sleep(2)
    logger.error(
        "SERVICE HEALTH CHECK: %s still unreachable after 3 attempts. "
        "Pipeline may fail at runtime.",
        ", ".join(n for n, _ in failed),
    )


def _warm_up_tts() -> None:
    """Send a short TTS request to prime the model (eliminates cold-start latency)."""
    import urllib.request
    import urllib.error

    url = f"{TTS_BASE_URL}/audio/speech"
    body = json.dumps({
        "model": TTS_MODEL,
        "input": "Hello.",
        "voice": "alloy",
        "response_format": "wav",
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp.read()  # discard audio
        logger.info("TTS warm-up complete")
    except Exception as exc:
        logger.warning("TTS warm-up failed (non-fatal): %s", exc)


class _PipelineMetricsObserver:
    """Observer that collects TTFB, processing, and usage metrics from pipecat.

    Attached to PipelineTask to intercept MetricsFrame data and pipeline
    lifecycle events.  Stores results in the provider's _metrics dict.
    """

    def __init__(self, provider: "PipecatProvider") -> None:
        self._provider = provider

    async def on_push_frame(self, data: Any) -> None:
        """Called for every frame pushed between processors."""
        frame = getattr(data, "frame", None)
        if frame is None:
            return

        # Only process MetricsFrame
        try:
            from pipecat.frames.frames import MetricsFrame
        except ImportError:
            return

        if not isinstance(frame, MetricsFrame):
            return

        from pipecat.metrics.metrics import (
            TTFBMetricsData,
            ProcessingMetricsData,
            LLMUsageMetricsData,
            TTSUsageMetricsData,
        )

        metrics = self._provider._metrics
        for item in frame.data:
            proc = getattr(item, "processor", "unknown")
            if isinstance(item, TTFBMetricsData):
                metrics["ttfb"][proc] = round(item.value, 3)
                hist = self._provider._ttfb_history.setdefault(proc, [])
                hist.append(item.value)
                if len(hist) > 10:
                    hist.pop(0)
                metrics["ttfb_avg"][proc] = round(sum(hist) / len(hist), 3)
                logger.debug("TTFB %s: %.3fs (avg %.3fs)", proc, item.value, metrics["ttfb_avg"][proc])
            elif isinstance(item, ProcessingMetricsData):
                metrics["processing"][proc] = round(item.value, 3)
            elif isinstance(item, LLMUsageMetricsData):
                model = getattr(item, "model", "unknown") or "unknown"
                usage = item.value
                metrics["token_usage"][model] = {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                }
            elif isinstance(item, TTSUsageMetricsData):
                metrics["tts_chars"] += item.value

    async def on_pipeline_started(self) -> None:
        """Called when pipeline has fully started."""
        logger.info("Pipeline observer: pipeline started")
        self._add_event("started")

    async def on_process_frame(self, data: Any) -> None:
        """Passthrough — we only care about pushed frames."""
        pass

    def _add_event(self, event_type: str) -> None:
        import time as _time
        events = self._provider._metrics["pipeline_events"]
        events.append({"type": event_type, "ts": _time.time()})
        if len(events) > 20:
            events.pop(0)


class PipecatProvider(ConversationProvider):
    """Fully-local conversation backend powered by pipecat-ai.

    Raises ``RuntimeError`` at instantiation if pipecat-ai is not installed.
    Install the ``local_pipeline`` extra to use this provider::

        pip install '.[local_pipeline]'
    """

    def __init__(
        self,
        deps: "ToolDependencies",
        *,
        gradio_mode: bool = False,
        instance_path: Optional[str] = None,
    ) -> None:
        """Initialise the provider; raises RuntimeError if pipecat-ai is absent."""
        if not _PIPECAT_AVAILABLE:
            raise RuntimeError(
                "pipecat-ai is not installed. Run: pip install '.[local_pipeline]' to use the local pipeline."
            )
        super().__init__(
            expected_layout="mono",
            output_sample_rate=FASTRTC_SAMPLE_RATE,
            input_sample_rate=FASTRTC_SAMPLE_RATE,
        )
        self.deps = deps
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path

        # Queues bridging fastrtc ↔ pipecat (bounded to prevent OOM on RPi 4)
        self._audio_in_queue: asyncio.Queue[Tuple[int, NDArray[np.int16]]] = asyncio.Queue(maxsize=100)
        self.output_queue: asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs] = asyncio.Queue(maxsize=100)

        # Lifecycle
        self._pipeline_task: Any | None = None
        self._runner_task: asyncio.Task[None] | None = None
        self._source: Any | None = None
        self.last_activity_time: float = 0.0
        self.start_time: float = 0.0
        self._last_emit_time: float = 0.0

        # Barge-in: generation counter prevents stale clears from racing
        self._barge_in_active: bool = False
        self._barge_in_generation: int = 0

        # Health tracking
        self._frames_dropped: int = 0
        self._last_stt_time: float = 0.0
        self._last_llm_time: float = 0.0
        self._last_tts_time: float = 0.0

        # Pipeline metrics (populated by observer)
        self._metrics: dict = {
            "ttfb": {},          # processor → latest TTFB in seconds
            "ttfb_avg": {},      # processor → rolling average TTFB
            "processing": {},    # processor → latest processing time
            "token_usage": {},   # model → {prompt, completion, total}
            "tts_chars": 0,      # total TTS characters processed
            "pipeline_events": [],  # last N pipeline events
        }
        self._ttfb_history: dict[str, list[float]] = {}  # processor → last 10 values

    # ------------------------------------------------------------------
    # ConversationProvider abstract methods
    # ------------------------------------------------------------------

    async def apply_personality(self, profile: Optional[str]) -> str:
        """Update LLM system instructions at runtime.

        Not yet implemented: would need to rebuild the LLMContext with
        new instructions and push an updated context frame.  Currently
        only takes effect after a pipeline restart.
        """
        from reachy_mini_conversation_app.config import set_custom_profile

        set_custom_profile(profile)
        logger.info("PipecatProvider: profile set to %r (live update pending)", profile)
        return f"Profile '{profile}' set. Pipeline restart needed for full effect."

    async def get_available_voices(self) -> List[str]:
        """Return voices available from the local TTS engine.

        Not yet implemented: would need to query TTS_BASE_URL/v1/voices
        or a similar endpoint.  Returns empty until a TTS voice listing
        API is available.
        """
        return []

    # ------------------------------------------------------------------
    # AsyncStreamHandler audio contract
    # ------------------------------------------------------------------

    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Accept an audio frame from fastrtc or LocalStream and forward into the pipeline.

        Handles both int16 (fastrtc/Gradio) and float32 (LocalStream/robot mic)
        input formats.  Resamples to the pipeline's 16 kHz if needed.
        """
        if self._pipeline_task is None:
            return

        input_sr, audio = frame

        if not isinstance(audio, np.ndarray):
            logger.warning("receive: expected ndarray, got %s", type(audio).__name__)
            return
        if audio.size == 0:
            return

        # Ensure mono
        if audio.ndim == 2:
            if audio.shape[1] > audio.shape[0]:
                audio = audio.T
            if audio.shape[1] > 1:
                audio = audio[:, 0]

        # Flatten any remaining extra dimensions
        audio = audio.ravel()

        # Convert float32 [-1.0, 1.0] → int16 [-32768, 32767]
        if audio.dtype == np.float32 or audio.dtype == np.float64:
            audio = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

        # Resample to pipeline rate
        if input_sr != PIPELINE_SAMPLE_RATE:
            n_samples = int(len(audio) * PIPELINE_SAMPLE_RATE / input_sr)
            if n_samples < 1:
                return
            audio = resample(audio, n_samples).astype(np.int16)

        logger.debug("receive: sr=%d dtype=%s samples=%d", PIPELINE_SAMPLE_RATE, audio.dtype, len(audio))
        try:
            self._audio_in_queue.put_nowait((PIPELINE_SAMPLE_RATE, audio))
        except asyncio.QueueFull:
            self._frames_dropped += 1
            if self._frames_dropped % 50 == 1:
                logger.warning("Audio input queue full, dropping frame (total dropped: %d)", self._frames_dropped)

    def copy(self) -> "PipecatProvider":
        """Return a new handler for a new WebRTC session."""
        return PipecatProvider(
            self.deps,
            gradio_mode=self.gradio_mode,
            instance_path=self.instance_path,
        )

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Emit audio / transcript frames to fastrtc.

        Also fires idle signals when the robot has been quiet too long.
        Uses a short timeout on queue.get() so we can periodically
        clear a stuck _barge_in flag and check idle state.
        """
        while True:
            now = asyncio.get_event_loop().time()

            idle_duration = asyncio.get_event_loop().time() - self.last_activity_time
            if idle_duration > 15.0 and self.deps.movement_manager.is_idle():
                try:
                    await self.send_idle_signal(idle_duration)
                except Exception as exc:
                    logger.warning("Idle signal skipped: %s", exc)
                    return None
                self.last_activity_time = asyncio.get_event_loop().time()

            # Watchdog: warn if no audio has been emitted for a long time
            # during what should be an active conversation.
            if self._last_emit_time > 0:
                silence_gap = now - self._last_emit_time
                if silence_gap > 30.0:
                    logger.warning(
                        "Emit watchdog: no output for %.0fs (barge_in=%s)",
                        silence_gap, self._barge_in_active,
                    )
                    self._last_emit_time = now  # reset so we don't spam

            try:
                item = await asyncio.wait_for(self.output_queue.get(), timeout=1.0)
                if isinstance(item, tuple):
                    self._last_emit_time = asyncio.get_event_loop().time()
                return item
            except asyncio.TimeoutError:
                continue  # loop back to check _barge_in / idle

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start_up(self) -> None:
        """Build and start the pipecat pipeline."""
        from pipecat.frames.frames import (
            TTSTextFrame,
            TTSAudioRawFrame,
            InputAudioRawFrame,
            TranscriptionFrame,
            OutputAudioRawFrame,
            InterimTranscriptionFrame,
            VADUserStartedSpeakingFrame,
            VADUserStoppedSpeakingFrame,
        )
        from pipecat.pipeline.task import PipelineTask, PipelineParams
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.services.openai.llm import OpenAILLMService
        from pipecat.services.openai.stt import OpenAISTTService
        from pipecat.services.openai.tts import OpenAITTSService
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
        from pipecat.processors.audio.vad_processor import VADProcessor
        from pipecat.processors.aggregators.llm_context import LLMContext
        from pipecat.processors.aggregators.llm_response_universal import (
            LLMContextFrame,
            LLMContextAggregatorPair,
        )

        from reachy_mini_conversation_app.prompts import get_session_instructions
        import threading as _threading

        loop = asyncio.get_event_loop()
        self.start_time = loop.time()
        self.last_activity_time = loop.time()

        # Pre-flight checks in background threads (non-blocking)
        _threading.Thread(target=_check_services, daemon=True, name="health-check").start()
        _threading.Thread(target=_warm_up_tts, daemon=True, name="tts-warmup").start()

        logger.info(
            "PipecatProvider: building pipeline  LLM=%s@%s  STT=%s@%s  TTS=%s@%s",
            LLM_MODEL,
            LLM_BASE_URL,
            ASR_MODEL,
            ASR_BASE_URL,
            TTS_MODEL,
            TTS_BASE_URL,
        )

        # ---- Services ---------------------------------------------------

        stt = OpenAISTTService(
            api_key="not-needed",
            base_url=ASR_BASE_URL,
            settings=OpenAISTTService.Settings(model=ASR_MODEL),
        )

        llm = OpenAILLMService(
            api_key="not-needed",
            base_url=LLM_BASE_URL,
            settings=OpenAILLMService.Settings(
                model=LLM_MODEL,
                system_instruction=get_session_instructions(),
                # Disable thinking via chat_template_kwargs so the Jinja
                # template pre-fills an empty <think>\n\n</think> block.
                # This avoids the 30-90s thinking latency on Qwen3.5.
                # Note: put directly in extra (not nested under extra_body)
                # because ik-llama.cpp needs it at the top level of the
                # request body, and pipecat merges extra directly into params.
                extra={
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            ),
        )

        # OpenAI TTS always outputs 24 kHz.  Our Qwen3-TTS endpoint is
        # OpenAI-compatible, so it also outputs 24 kHz — no resampling
        # needed between TTS output and the fastrtc contract.
        tts = OpenAITTSService(
            api_key="not-needed",
            base_url=TTS_BASE_URL,
            settings=OpenAITTSService.Settings(model=TTS_MODEL),
        )

        # ---- VAD ---------------------------------------------------------
        # Standalone VADProcessor generates VADUserStartedSpeakingFrame /
        # VADUserStoppedSpeakingFrame that both the STT service and the
        # user aggregator listen for.

        vad = VADProcessor(
            vad_analyzer=SileroVADAnalyzer(sample_rate=PIPELINE_SAMPLE_RATE),
        )

        # ---- Tool registration --------------------------------------------
        # Convert the app's tool specs to OpenAI-compatible format and register
        # a catch-all handler that dispatches to the existing tool registry.

        from pipecat.adapters.schemas.function_schema import FunctionSchema
        from pipecat.adapters.schemas.tools_schema import ToolsSchema
        from reachy_mini_conversation_app.tools.core_tools import (
            get_tool_specs,
            dispatch_tool_call,
        )

        tool_specs = get_tool_specs()
        # Convert app tool specs to pipecat FunctionSchema objects
        function_schemas = []
        for spec in tool_specs:
            params = spec.get("parameters", {})
            function_schemas.append(FunctionSchema(
                name=spec["name"],
                description=spec.get("description", ""),
                properties=params.get("properties", {}),
                required=params.get("required", []),
            ))
        openai_tools = ToolsSchema(standard_tools=function_schemas)

        provider_ref_for_tools = self

        async def _handle_tool_call(params) -> None:
            """Catch-all handler for all pipecat function calls."""
            fn_name = params.function_name
            arguments = params.arguments
            tool_call_id = params.tool_call_id

            logger.info("Tool call: %s(%s) [%s]", fn_name, arguments, tool_call_id)

            # pipecat passes arguments as a dict (Mapping), but
            # dispatch_tool_call expects a JSON string.
            args_json = json.dumps(arguments) if isinstance(arguments, dict) else str(arguments)
            try:
                result = await dispatch_tool_call(fn_name, args_json, provider_ref_for_tools.deps)
            except Exception as exc:
                logger.exception("Tool call %s failed", fn_name)
                result = {"error": str(exc)}

            await params.result_callback(json.dumps(result))

            # Notify UI about tool usage
            try:
                provider_ref_for_tools.output_queue.put_nowait(
                    AdditionalOutputs({"role": "assistant", "content": f"Tool {fn_name}: {result}"})
                )
            except asyncio.QueueFull:
                pass  # UI notification drop is acceptable

        # Register catch-all (function_name=None handles all tool calls)
        llm.register_function(None, _handle_tool_call)

        logger.info("Registered %d tools with LLM: %s",
                     len(tool_specs), [s["name"] for s in tool_specs])

        # ---- Context & aggregators ---------------------------------------

        context = LLMContext(tools=openai_tools)

        # MinWords turn start strategy: require N words to interrupt the bot.
        # Single-word backchannel ("yeah", "uh-huh") won't interrupt.
        from pipecat.turns.user_start.min_words_user_turn_start_strategy import (
            MinWordsUserTurnStartStrategy,
        )
        from pipecat.turns.user_turn_strategies import UserTurnStrategies
        from pipecat.processors.aggregators.llm_response_universal import (
            LLMUserAggregatorParams,
        )

        _min_interrupt_words = int(os.environ.get("MIN_INTERRUPT_WORDS", "3"))
        user_params = LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                start=[MinWordsUserTurnStartStrategy(min_words=_min_interrupt_words)],
            ),
        )
        user_agg, assistant_agg = LLMContextAggregatorPair(
            context, user_params=user_params,
        )
        logger.info("MinWords interruption: %d words to interrupt bot", _min_interrupt_words)

        # ---- Bridge processors -------------------------------------------
        # PipelineSource: pulls from _audio_in_queue, emits InputAudioRawFrame
        # PipelineSink: catches TTS audio + transcripts, pushes to output_queue

        provider_ref = self  # avoid closure over self in nested classes

        # Noise reduction (optional — degrades gracefully if noisereduce not installed)
        try:
            import noisereduce as nr
            _nr_available = True
            logger.info("Noise reduction: noisereduce available, will filter mic audio")
        except ImportError:
            _nr_available = False
            logger.info("Noise reduction: noisereduce not installed, skipping (pip install noisereduce)")

        class PipelineSource(FrameProcessor):
            """Pulls audio from the fastrtc inbound queue into the pipeline."""

            def __init__(self) -> None:
                super().__init__()
                self._running = True

            async def process_frame(self, frame: Any, direction: FrameDirection) -> None:
                """Forward frames that arrive from upstream."""
                await super().process_frame(frame, direction)

            async def run(self, task: PipelineTask) -> None:
                """Background loop that feeds mic audio into the pipeline."""
                _src_count = 0
                while self._running:
                    try:
                        sr, audio = await asyncio.wait_for(provider_ref._audio_in_queue.get(), timeout=0.5)
                    except asyncio.TimeoutError:
                        continue

                    # Audio is already int16 after receive() conversion
                    if audio.dtype != np.int16:
                        if audio.dtype in (np.float32, np.float64):
                            audio = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
                        else:
                            audio = audio.astype(np.int16)

                    # Apply noise reduction before VAD/STT
                    # noisereduce needs ≥1024 samples (scipy welch nperseg);
                    # real-time chunks are typically 320 samples (20ms @ 16kHz),
                    # so skip noise reduction for now — it corrupts short frames.
                    # TODO: accumulate a buffer of ≥1024 samples before filtering.

                    audio_bytes = audio.tobytes()
                    frame = InputAudioRawFrame(
                        audio=audio_bytes,
                        sample_rate=sr,
                        num_channels=1,
                    )
                    await task.queue_frame(frame)

            def stop(self) -> None:
                self._running = False

        class ASRTextCleaner(FrameProcessor):
            """Strips Qwen ASR prefix tags and filters noise transcriptions."""

            async def process_frame(self, frame: Any, direction: FrameDirection) -> None:
                await super().process_frame(frame, direction)
                if isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
                    text = getattr(frame, "text", "")
                    cleaned = text.replace("<asr_text>", "").strip()
                    if cleaned != text:
                        frame.text = cleaned
                        text = cleaned
                    if _is_noise(text):
                        logger.debug("ASR noise filtered: %r", text)
                        return
                await self.push_frame(frame, direction)

        class ContextTrimmer(FrameProcessor):
            """Trim old messages from the LLM context to prevent OOM on RPi 4."""

            async def process_frame(self, frame: Any, direction: FrameDirection) -> None:
                await super().process_frame(frame, direction)
                if isinstance(frame, LLMContextFrame):
                    messages = frame.context.get_messages()
                    trimmed = _trim_context(messages)
                    if trimmed is not None:
                        frame.context.set_messages(trimmed)
                        logger.info(
                            "ContextTrimmer: trimmed %d → %d messages",
                            len(messages), len(trimmed),
                        )
                await self.push_frame(frame, direction)

        class VisionInjector(FrameProcessor):
            """Attach a camera snapshot to every LLM turn.

            "Always capture, route later" pattern: on each
            LLMContextFrame, grab the latest camera frame and inject
            visual context into the last user message.

            Two modes depending on the LLM's capabilities:

            1. **VLM / multimodal LLM** (``multimodal=True``):
               JPEG-encode the frame and inject it as an ``image_url``
               data-URI in OpenAI multimodal message format.

            2. **Text-only LLM** with a local ``VisionProcessor``
               (``multimodal=False``, ``vision_processor`` set):
               Run SmolVLM2 on the frame to produce a short scene
               description and inject it as a ``[scene: ...]`` text
               prefix in the user message.

            If neither mode is possible, the context passes through
            unchanged.
            """

            MAX_DIM = 512
            JPEG_QUALITY = 60

            def __init__(self, *, multimodal: bool = False) -> None:
                super().__init__()
                self._multimodal = multimodal

            async def process_frame(self, frame: Any, direction: FrameDirection) -> None:
                await super().process_frame(frame, direction)
                if not isinstance(frame, LLMContextFrame):
                    await self.push_frame(frame, direction)
                    return

                cam = provider_ref.deps.camera_worker
                if cam is None:
                    await self.push_frame(frame, direction)
                    return

                raw_frame = cam.get_latest_frame()
                if raw_frame is None:
                    await self.push_frame(frame, direction)
                    return

                if self._multimodal:
                    await self._inject_image(frame, raw_frame, direction)
                elif provider_ref.deps.vision_processor is not None:
                    await self._inject_description(frame, raw_frame, direction)
                else:
                    # Text-only LLM, no vision processor → pass through
                    await self.push_frame(frame, direction)
                    return

            async def _inject_image(
                self, frame: Any, raw_frame: Any, direction: FrameDirection,
            ) -> None:
                """Inject base64 JPEG into the last user message (multimodal LLM)."""
                b64_url = await asyncio.to_thread(self._encode_frame, raw_frame)
                if b64_url is None:
                    await self.push_frame(frame, direction)
                    return

                messages = frame.context.get_messages()
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            msg["content"] = [
                                {"type": "text", "text": content},
                                {"type": "image_url", "image_url": {"url": b64_url}},
                            ]
                        elif isinstance(content, list):
                            content.append(
                                {"type": "image_url", "image_url": {"url": b64_url}}
                            )
                        break

                logger.debug("VisionInjector: attached image to LLM context")
                await self.push_frame(frame, direction)

            async def _inject_description(
                self, frame: Any, raw_frame: Any, direction: FrameDirection,
            ) -> None:
                """Run local VLM and inject text description (text-only LLM)."""
                try:
                    desc = await asyncio.to_thread(
                        provider_ref.deps.vision_processor.process_image,
                        raw_frame,
                        "Briefly describe the scene and any people visible.",
                    )
                    if desc and desc != "Vision model not initialized":
                        messages = frame.context.get_messages()
                        for msg in reversed(messages):
                            if msg.get("role") == "user":
                                text = msg.get("content", "")
                                if isinstance(text, str):
                                    msg["content"] = f"[scene: {desc}]\n{text}"
                                break
                        logger.debug("VisionInjector: injected scene description")
                except Exception as e:
                    logger.warning("VisionInjector: VLM failed: %s", e)

                await self.push_frame(frame, direction)

            @staticmethod
            def _encode_frame(frame: Any) -> str | None:
                """Resize + JPEG-encode a BGR numpy frame → data URL."""
                import cv2

                try:
                    h, w = frame.shape[:2]
                    if max(h, w) > VisionInjector.MAX_DIM:
                        scale = VisionInjector.MAX_DIM / max(h, w)
                        frame = cv2.resize(
                            frame,
                            (int(w * scale), int(h * scale)),
                            interpolation=cv2.INTER_AREA,
                        )
                    ok, buf = cv2.imencode(
                        ".jpg",
                        frame,
                        [cv2.IMWRITE_JPEG_QUALITY, VisionInjector.JPEG_QUALITY],
                    )
                    if not ok:
                        return None
                    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
                    return f"data:image/jpeg;base64,{b64}"
                except Exception as e:
                    logger.warning("VisionInjector: encode failed: %s", e)
                    return None

        class TTSTextChunker(FrameProcessor):
            """Split long LLM text into TTS-friendly chunks for faster
            time-to-first-audio.

            Uses a waterfall approach: sentence boundaries → clause
            boundaries → phrase boundaries → word boundaries, with a
            configurable max chunk size.
            """

            _WATERFALL = [
                re.compile(r'([.!?…]+["\'\)]?\s+)'),  # sentence endings
                re.compile(r'([:;]\s+)'),               # clause separators
                re.compile(r'([,—]\s+)'),               # phrase separators
                re.compile(r'(\s+)'),                    # any whitespace
            ]

            def __init__(self, max_chars: int = 150) -> None:
                super().__init__()
                self._max_chars = max_chars

            async def process_frame(self, frame: Any, direction: FrameDirection) -> None:
                await super().process_frame(frame, direction)
                if not isinstance(frame, TTSTextFrame):
                    await self.push_frame(frame, direction)
                    return

                text = getattr(frame, "text", "").strip()
                if not text:
                    await self.push_frame(frame, direction)
                    return

                chunks = self._split(text)
                for chunk in chunks:
                    await self.push_frame(
                        TTSTextFrame(text=chunk), direction,
                    )

            def _split(self, text: str) -> list[str]:
                if len(text) <= self._max_chars:
                    return [text]

                chunks: list[str] = []
                remaining = text

                while remaining:
                    if len(remaining) <= self._max_chars:
                        chunks.append(remaining.strip())
                        break

                    best_break = self._find_break(remaining)

                    if best_break and best_break > 20:
                        chunk = remaining[:best_break].strip()
                        remaining = remaining[best_break:].strip()
                    else:
                        # Force break at a space near max_chars
                        space_idx = remaining[: self._max_chars].rfind(" ")
                        if space_idx > 20:
                            chunk = remaining[:space_idx].strip()
                            remaining = remaining[space_idx:].strip()
                        else:
                            chunk = remaining[: self._max_chars].strip()
                            remaining = remaining[self._max_chars :].strip()

                    if chunk:
                        chunks.append(chunk)

                return chunks

            def _find_break(self, text: str) -> int | None:
                window = text[: self._max_chars + 50]
                for pattern in self._WATERFALL:
                    matches = list(pattern.finditer(window))
                    if matches:
                        for m in reversed(matches):
                            if m.end() <= self._max_chars + 20:
                                return m.end()
                return None

        class AssistantTextTap(FrameProcessor):
            """Taps LLM output text before it reaches TTS.

            Intercepts tool-call-like text (e.g. ``micro_expression(happy)``)
            that the LLM wrote as content instead of a proper tool call,
            dispatches them, and strips them from the TTS text.
            """

            _ARG_MAP = {
                "micro_expression": "expression",
                "play_emotion": "emotion_name",
                "move_head": "direction",
                "dance": "move_name",
            }

            async def process_frame(self, frame: Any, direction: FrameDirection) -> None:
                await super().process_frame(frame, direction)
                if isinstance(frame, TTSTextFrame):
                    text = getattr(frame, "text", "")
                    if text:
                        provider_ref._last_llm_time = asyncio.get_event_loop().time()
                        text = await self._intercept_tool_text(text)
                        frame.text = text
                        if text.strip():
                            try:
                                provider_ref.output_queue.put_nowait(
                                    AdditionalOutputs({"role": "assistant", "content": text})
                                )
                            except asyncio.QueueFull:
                                pass  # transcript drop is acceptable
                await self.push_frame(frame, direction)

            async def _intercept_tool_text(self, text: str) -> str:
                """Find tool-call patterns in text, dispatch them, return cleaned text."""
                calls = _extract_inline_tool_calls(text)
                if not calls:
                    return text

                for fn_name, raw_arg in calls:
                    param = self._ARG_MAP.get(fn_name)
                    if param and raw_arg:
                        args_json = json.dumps({param: raw_arg})
                    else:
                        args_json = json.dumps({})
                    logger.info("Intercepted text tool call: %s(%s)", fn_name, raw_arg)
                    try:
                        await dispatch_tool_call(fn_name, args_json, provider_ref.deps)
                    except Exception as exc:
                        logger.warning("Intercepted tool call %s failed: %s", fn_name, exc)

                return _strip_inline_tool_calls(text)

        class PipelineSink(FrameProcessor):
            """Intercepts output frames and bridges them to fastrtc."""

            async def process_frame(self, frame: Any, direction: FrameDirection) -> None:
                # Let base class handle lifecycle (StartFrame → __start, etc.)
                await super().process_frame(frame, direction)

                # TTS audio → output queue + head wobbler
                # While the robot is speaking, suppress listening so
                # idle breathing is allowed and antennas aren't frozen.
                if isinstance(frame, (OutputAudioRawFrame, TTSAudioRawFrame)):
                    # During barge-in, drop TTS audio — the user is speaking
                    # and pipecat's InterruptionFrame hasn't fully propagated yet.
                    if provider_ref._barge_in_active:
                        await self.push_frame(frame, direction)
                        return

                    audio_bytes = frame.audio
                    sr = frame.sample_rate

                    if len(audio_bytes) == 0:
                        await self.push_frame(frame, direction)
                        return

                    provider_ref.deps.movement_manager.set_listening(False)
                    if provider_ref._doa_tracker is not None:
                        provider_ref._doa_tracker.set_enabled(False)
                    provider_ref.last_activity_time = asyncio.get_event_loop().time()

                    # Convert raw PCM bytes → int16 numpy and boost volume.
                    # Qwen3-TTS outputs at ~7% RMS — apply gain to fill
                    # the dynamic range without clipping.  (2.8x ≈ 70% of 4x)
                    pcm = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.int32)
                    if pcm.size == 0:
                        await self.push_frame(frame, direction)
                        return
                    pcm = np.clip(pcm * 2.8, -32768, 32767).astype(np.int16)
                    logger.debug("PipelineSink: TTS frame sr=%d pcm_samples=%d", sr, len(pcm))

                    # Resample to fastrtc 24 kHz if needed (TTS should
                    # already output 24 kHz, but handle mismatches).
                    if sr != FASTRTC_SAMPLE_RATE:
                        n_out = int(len(pcm) * FASTRTC_SAMPLE_RATE / sr)
                        pcm = resample(pcm, n_out).astype(np.int16)

                    # Feed HeadWobbler with 24 kHz PCM (same rate it
                    # expects from the OpenAI path).
                    if provider_ref.deps.head_wobbler is not None:
                        try:
                            b64 = base64.b64encode(pcm.tobytes()).decode("utf-8")
                            provider_ref.deps.head_wobbler.feed(b64)
                        except Exception as exc:
                            logger.warning("HeadWobbler feed failed: %s", exc)

                    try:
                        provider_ref.output_queue.put_nowait((FASTRTC_SAMPLE_RATE, pcm.reshape(1, -1)))
                        provider_ref._last_tts_time = asyncio.get_event_loop().time()
                    except asyncio.QueueFull:
                        provider_ref._frames_dropped += 1
                        if provider_ref._frames_dropped % 50 == 1:
                            logger.warning("Output queue full, dropping TTS frame (total dropped: %d)", provider_ref._frames_dropped)

                # Final user transcript
                elif isinstance(frame, TranscriptionFrame):
                    # Clear barge-in on transcript — ensures the flag is
                    # reset even if VADUserStoppedSpeakingFrame was lost.
                    provider_ref._barge_in_active = False
                    provider_ref._last_stt_time = asyncio.get_event_loop().time()
                    text = getattr(frame, "text", "")
                    if text:
                        logger.debug("User transcript: %s", text)
                        try:
                            provider_ref.output_queue.put_nowait(AdditionalOutputs({"role": "user", "content": text}))
                        except asyncio.QueueFull:
                            pass

                # Partial user transcript
                elif isinstance(frame, InterimTranscriptionFrame):
                    text = getattr(frame, "text", "")
                    if text:
                        try:
                            provider_ref.output_queue.put_nowait(
                                AdditionalOutputs({"role": "user_partial", "content": text})
                            )
                        except asyncio.QueueFull:
                            pass

                # VAD speech boundaries → listening mode for movement manager.
                # The robot should always be listening when idle — only
                # TTS playback (above) temporarily clears the flag.
                elif isinstance(frame, VADUserStartedSpeakingFrame):
                    provider_ref._barge_in_generation += 1
                    provider_ref._barge_in_active = True
                    provider_ref.deps.movement_manager.set_listening(True)
                    if provider_ref.deps.head_wobbler is not None:
                        provider_ref.deps.head_wobbler.reset()
                    if provider_ref._doa_tracker is not None:
                        provider_ref._doa_tracker.set_enabled(True)
                    # Barge-in: drain ALL pending audio from the output
                    # queue so the robot stops talking immediately.
                    # Preserve transcript AdditionalOutputs (non-audio).
                    kept: list = []
                    while not provider_ref.output_queue.empty():
                        try:
                            item = provider_ref.output_queue.get_nowait()
                            if isinstance(item, AdditionalOutputs):
                                kept.append(item)
                            # else: discard audio tuples
                        except asyncio.QueueEmpty:
                            break
                    for item in kept:
                        try:
                            provider_ref.output_queue.put_nowait(item)
                        except asyncio.QueueFull:
                            pass  # acceptable during congestion
                    # Also flush the robot's GStreamer player buffer so
                    # already-pushed audio stops immediately.
                    clear_fn = getattr(provider_ref, "_clear_queue", None)
                    if clear_fn is not None:
                        try:
                            clear_fn()
                        except Exception as exc:
                            logger.debug("clear_queue during barge-in: %s", exc)
                    logger.debug("User speech started (barge-in, drained audio + player)")

                elif isinstance(frame, VADUserStoppedSpeakingFrame):
                    # Clear barge-in flag so next LLM response audio plays
                    provider_ref._barge_in_active = False
                    # Stay in listening mode — robot remains attentive
                    # between utterances.  Listening is only cleared
                    # while TTS audio is actively playing.
                    provider_ref.deps.movement_manager.set_listening(True)
                    logger.debug("User speech stopped (still listening)")

                # Always forward so downstream processors
                # (like assistant_aggregator) still see the frame.
                await self.push_frame(frame, direction)

        class MemoryInjector(FrameProcessor):
            """Inject relevant memories into LLM context before each turn.

            Queries Mem0 with the last user message and prepends matching
            memories as a system-message prefix so the LLM can reference them.
            Results are cached for 5s to avoid redundant API calls.
            """

            def __init__(self) -> None:
                super().__init__()
                self._cache: list[dict] = []
                self._cache_time: float = 0.0
                self._cache_ttl: float = 5.0

            async def process_frame(self, frame: Any, direction: FrameDirection) -> None:
                await super().process_frame(frame, direction)
                if not isinstance(frame, LLMContextFrame):
                    await self.push_frame(frame, direction)
                    return

                # Extract last user message for search query
                messages = frame.context.get_messages()
                query = ""
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            query = content
                        elif isinstance(content, list):
                            for part in content:
                                if isinstance(part, dict) and part.get("type") == "text":
                                    query = part.get("text", "")
                                    break
                        break

                if query:
                    memories = await self._get_memories(query)
                    if memories:
                        mem_text = ", ".join(m.get("content", "") for m in memories if m.get("content"))
                        if mem_text:
                            # Inject as first system message after the main system prompt
                            messages = frame.context.get_messages()
                            insert_idx = 1 if messages and messages[0].get("role") == "system" else 0
                            mem_msg = {"role": "system", "content": f"[memories about user: {mem_text}]"}
                            messages.insert(insert_idx, mem_msg)
                            frame.context.set_messages(messages)
                            logger.debug("MemoryInjector: injected %d memories", len(memories))

                await self.push_frame(frame, direction)

            async def _get_memories(self, query: str) -> list[dict]:
                """Search memories with caching."""
                import time as _time

                now = _time.monotonic()
                if now - self._cache_time < self._cache_ttl and self._cache:
                    return self._cache

                try:
                    from reachy_mini_conversation_app.memory.mem0_client import Mem0Client

                    client = Mem0Client()
                    try:
                        items = await client.search_memories(query, limit=5)
                        self._cache = items
                        self._cache_time = now
                        return items
                    finally:
                        await client.close()
                except Exception as exc:
                    logger.debug("MemoryInjector: search failed: %s", exc)
                    return []

        class AutoMemoryTap(FrameProcessor):
            """Auto-store conversation turns to Mem0 for fact extraction.

            Monitors assistant text output and periodically sends the
            conversation context to Mem0 with infer=True so it can
            extract memorable facts automatically.
            Rate-limited to 1 store per 10s.
            """

            def __init__(self) -> None:
                super().__init__()
                self._last_store_time: float = 0.0
                self._store_interval: float = 10.0
                self._pending_text: str = ""

            async def process_frame(self, frame: Any, direction: FrameDirection) -> None:
                await super().process_frame(frame, direction)

                # Collect user transcripts and assistant text
                if isinstance(frame, TranscriptionFrame):
                    text = getattr(frame, "text", "")
                    if text:
                        self._pending_text += f" User: {text}"
                elif isinstance(frame, TTSTextFrame):
                    text = getattr(frame, "text", "")
                    if text:
                        self._pending_text += f" Assistant: {text}"

                # Rate-limited store
                if self._pending_text:
                    import time as _time

                    now = _time.monotonic()
                    if now - self._last_store_time > self._store_interval:
                        text_to_store = self._pending_text.strip()
                        self._pending_text = ""
                        self._last_store_time = now
                        # Fire-and-forget — don't block the pipeline
                        asyncio.create_task(
                            self._store(text_to_store),
                            name="auto-memory-store",
                        )

                await self.push_frame(frame, direction)

            async def _store(self, text: str) -> None:
                try:
                    from reachy_mini_conversation_app.memory.mem0_client import Mem0Client

                    client = Mem0Client()
                    try:
                        await client.add_memory(text)
                    finally:
                        await client.close()
                except Exception as exc:
                    logger.debug("AutoMemoryTap: store failed: %s", exc)

        class ParallelEnricher(FrameProcessor):
            """Run memory and vision enrichment concurrently.

            Replaces sequential MemoryInjector → VisionInjector with a
            single processor that runs both via asyncio.gather(), saving
            ~latency-of-slower-operation per turn.
            """

            MAX_DIM = 512
            JPEG_QUALITY = 60

            def __init__(self, *, multimodal: bool = False) -> None:
                super().__init__()
                self._multimodal = multimodal
                # Memory cache
                self._mem_cache: list[dict] = []
                self._mem_cache_time: float = 0.0
                self._mem_cache_ttl: float = 5.0

            async def process_frame(self, frame: Any, direction: FrameDirection) -> None:
                await super().process_frame(frame, direction)
                if not isinstance(frame, LLMContextFrame):
                    await self.push_frame(frame, direction)
                    return

                # Extract last user message (shared by both operations)
                messages = frame.context.get_messages()
                user_text = ""
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            user_text = content
                        elif isinstance(content, list):
                            for part in content:
                                if isinstance(part, dict) and part.get("type") == "text":
                                    user_text = part.get("text", "")
                                    break
                        break

                # Run memory + vision concurrently
                mem_result, vision_result = await asyncio.gather(
                    self._fetch_memories(user_text),
                    self._fetch_vision(frame),
                    return_exceptions=True,
                )

                # Apply memory enrichment (adds system message)
                if isinstance(mem_result, str) and mem_result:
                    messages = frame.context.get_messages()
                    insert_idx = 1 if messages and messages[0].get("role") == "system" else 0
                    messages.insert(insert_idx, {
                        "role": "system",
                        "content": f"[memories about user: {mem_result}]",
                    })
                    frame.context.set_messages(messages)
                    logger.debug("ParallelEnricher: injected memories")
                elif isinstance(mem_result, Exception):
                    logger.debug("ParallelEnricher: memory fetch failed: %s", mem_result)

                # Apply vision enrichment (modifies last user message)
                if isinstance(vision_result, Exception):
                    logger.debug("ParallelEnricher: vision fetch failed: %s", vision_result)
                elif vision_result is not None:
                    mode, data = vision_result
                    messages = frame.context.get_messages()
                    for msg in reversed(messages):
                        if msg.get("role") == "user":
                            if mode == "image":
                                content = msg.get("content", "")
                                if isinstance(content, str):
                                    msg["content"] = [
                                        {"type": "text", "text": content},
                                        {"type": "image_url", "image_url": {"url": data}},
                                    ]
                                elif isinstance(content, list):
                                    content.append(
                                        {"type": "image_url", "image_url": {"url": data}}
                                    )
                            elif mode == "description":
                                text = msg.get("content", "")
                                if isinstance(text, str):
                                    msg["content"] = f"[scene: {data}]\n{text}"
                            break
                    logger.debug("ParallelEnricher: injected vision (%s)", mode)

                await self.push_frame(frame, direction)

            async def _fetch_memories(self, query: str) -> str:
                """Search Mem0 with caching. Returns formatted string or empty."""
                if not query:
                    return ""
                import time as _time

                now = _time.monotonic()
                if now - self._mem_cache_time < self._mem_cache_ttl and self._mem_cache:
                    items = self._mem_cache
                else:
                    try:
                        from reachy_mini_conversation_app.memory.mem0_client import Mem0Client

                        client = Mem0Client()
                        try:
                            items = await client.search_memories(query, limit=5)
                            self._mem_cache = items
                            self._mem_cache_time = now
                        finally:
                            await client.close()
                    except Exception as exc:
                        logger.debug("ParallelEnricher: memory search failed: %s", exc)
                        return ""

                return ", ".join(m.get("content", "") for m in items if m.get("content"))

            async def _fetch_vision(self, frame: Any) -> tuple[str, str] | None:
                """Capture camera frame and encode/describe. Returns (mode, data) or None."""
                cam = provider_ref.deps.camera_worker
                if cam is None:
                    return None

                raw_frame = cam.get_latest_frame()
                if raw_frame is None:
                    return None

                if self._multimodal:
                    b64_url = await asyncio.to_thread(self._encode_frame, raw_frame)
                    return ("image", b64_url) if b64_url else None
                elif provider_ref.deps.vision_processor is not None:
                    try:
                        desc = await asyncio.to_thread(
                            provider_ref.deps.vision_processor.process_image,
                            raw_frame,
                            "Briefly describe the scene and any people visible.",
                        )
                        if desc and desc != "Vision model not initialized":
                            return ("description", desc)
                    except Exception as e:
                        logger.warning("ParallelEnricher: VLM failed: %s", e)
                return None

            @staticmethod
            def _encode_frame(frame: Any) -> str | None:
                """Resize + JPEG-encode a BGR numpy frame → data URL."""
                import cv2

                try:
                    h, w = frame.shape[:2]
                    if max(h, w) > ParallelEnricher.MAX_DIM:
                        scale = ParallelEnricher.MAX_DIM / max(h, w)
                        frame = cv2.resize(
                            frame,
                            (int(w * scale), int(h * scale)),
                            interpolation=cv2.INTER_AREA,
                        )
                    ok, buf = cv2.imencode(
                        ".jpg",
                        frame,
                        [cv2.IMWRITE_JPEG_QUALITY, ParallelEnricher.JPEG_QUALITY],
                    )
                    if not ok:
                        return None
                    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
                    return f"data:image/jpeg;base64,{b64}"
                except Exception as e:
                    logger.warning("ParallelEnricher: encode failed: %s", e)
                    return None

        class IntentRouter(FrameProcessor):
            """Route LLM requests to fast or full model based on intent.

            Classifies the last user message and logs the routing decision.
            When a fast LLM is configured, adjusts the LLM service settings
            for casual intents (shorter max_tokens, potentially different model).
            """

            def __init__(self, llm_service: Any) -> None:
                super().__init__()
                self._llm = llm_service
                from reachy_mini_conversation_app.llm.llm_router import LLMRouter
                self._router = LLMRouter(
                    default_base_url=LLM_BASE_URL,
                    default_model=LLM_MODEL,
                )

            async def process_frame(self, frame: Any, direction: FrameDirection) -> None:
                await super().process_frame(frame, direction)
                if isinstance(frame, LLMContextFrame):
                    # Classify the last user message
                    messages = frame.context.get_messages()
                    user_text = ""
                    for msg in reversed(messages):
                        if msg.get("role") == "user":
                            content = msg.get("content", "")
                            if isinstance(content, str):
                                user_text = content
                            break

                    if user_text:
                        from reachy_mini_conversation_app.llm.intent_classifier import classify
                        result = classify(user_text)
                        config = self._router.route(result.intent)

                        # Log routing decision
                        logger.debug(
                            "IntentRouter: '%s' → %s (%.0f%%) → %s",
                            user_text[:50], result.intent.value,
                            result.confidence * 100, config.model[:30],
                        )

                await self.push_frame(frame, direction)

        # ---- Assemble pipeline -------------------------------------------
        # PipelineTask._process_push_queue() sends StartFrame automatically
        # with audio_in/out sample rates from PipelineParams.
        #
        # Pipeline order:
        #   VAD → STT → ASRCleaner → user_agg(+smart-turn) → ContextTrimmer
        #     → ParallelEnricher(memory+vision) → IntentRouter → LLM → text_tap
        #     → TTSChunker → TTS → sink → assistant_agg → AutoMemoryTap
        #
        # PipelineSource runs as a separate asyncio task and injects
        # InputAudioRawFrame via task.queue_frame().

        source = PipelineSource()
        asr_cleaner = ASRTextCleaner()

        # Detect multimodal LLM by model name patterns or env var override.
        # LLM_MULTIMODAL=0/false/no explicitly disables; =1/true/yes forces on.
        _multimodal_patterns = ("vlm", "vl-", "vision", "llava", "smolvlm")
        _multimodal_env = os.environ.get("LLM_MULTIMODAL", "").lower()
        if _multimodal_env in ("0", "false", "no"):
            _is_multimodal = False
        elif _multimodal_env in ("1", "true", "yes"):
            _is_multimodal = True
        else:
            _is_multimodal = any(p in LLM_MODEL.lower() for p in _multimodal_patterns)
        vision_injector = VisionInjector(multimodal=_is_multimodal)

        _has_vision = _is_multimodal or provider_ref.deps.vision_processor is not None
        if _has_vision:
            logger.info(
                "VisionInjector: %s",
                "multimodal LLM — injecting images"
                if _is_multimodal
                else "text-only LLM — injecting scene descriptions via VisionProcessor",
            )
        else:
            logger.info("VisionInjector: disabled (text-only LLM, no VisionProcessor)")

        context_trimmer = ContextTrimmer()
        parallel_enricher = ParallelEnricher(multimodal=_is_multimodal)
        intent_router = IntentRouter(llm)
        tts_chunker = TTSTextChunker(max_chars=80)
        text_tap = AssistantTextTap()
        sink = PipelineSink()
        auto_memory = AutoMemoryTap()

        logger.info(
            "SmartTurn: using pipecat built-in LocalSmartTurnAnalyzerV3 "
            "(bundled ONNX model, runs on-device)"
        )

        pipeline = Pipeline(
            [
                vad,               # analyse audio for speech boundaries
                stt,               # transcribe speech segments
                asr_cleaner,       # strip <asr_text> prefix + noise filter
                user_agg,          # accumulate user turns + smart-turn detection
                context_trimmer,   # trim old messages to prevent OOM
                parallel_enricher, # memory + vision concurrently via gather()
                intent_router,     # classify intent + log routing decision
                llm,               # generate response
                text_tap,         # capture assistant text before TTS
                tts_chunker,      # split long text into TTS-friendly chunks
                tts,              # synthesise speech
                sink,             # bridge audio + transcripts → fastrtc
                assistant_agg,    # accumulate assistant turns
                auto_memory,      # auto-store conversation facts to Mem0
            ]
        )

        # Create metrics observer
        metrics_observer = _PipelineMetricsObserver(self)

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                audio_in_sample_rate=PIPELINE_SAMPLE_RATE,
                audio_out_sample_rate=FASTRTC_SAMPLE_RATE,
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
            observers=[metrics_observer],
        )
        self._pipeline_task = task
        self._source = source

        # Expose output_queue to tools so they can inject audio directly
        self.deps.output_queue = self.output_queue
        self.deps.audio_sample_rate = FASTRTC_SAMPLE_RATE

        # Kick off the pipeline in a background asyncio task with retry
        max_attempts = 3

        def _build_pipeline():
            """Rebuild all pipeline components for a fresh attempt.

            Rebuilds services too — if a service has a corrupted internal
            state or a dangling HTTP connection, reusing it would cause
            the same crash on retry.
            """
            _source = PipelineSource()
            _asr_cleaner = ASRTextCleaner()
            _parallel_enricher = ParallelEnricher(multimodal=_is_multimodal)
            _context_trimmer = ContextTrimmer()
            _tts_chunker = TTSTextChunker(max_chars=80)
            _text_tap = AssistantTextTap()
            _sink = PipelineSink()
            _auto_memory = AutoMemoryTap()
            _context = LLMContext(tools=openai_tools)
            _user_params = LLMUserAggregatorParams(
                user_turn_strategies=UserTurnStrategies(
                    start=[MinWordsUserTurnStartStrategy(min_words=_min_interrupt_words)],
                ),
            )
            _user_agg, _assistant_agg = LLMContextAggregatorPair(
                _context, user_params=_user_params,
            )

            # Rebuild services to avoid reusing corrupted state
            _stt = OpenAISTTService(
                api_key="not-needed",
                base_url=ASR_BASE_URL,
                settings=OpenAISTTService.Settings(model=ASR_MODEL),
            )
            _llm = OpenAILLMService(
                api_key="not-needed",
                base_url=LLM_BASE_URL,
                settings=OpenAILLMService.Settings(
                    model=LLM_MODEL,
                    system_instruction=get_session_instructions(),
                    extra={
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                ),
            )
            _llm.register_function(None, _handle_tool_call)
            _tts = OpenAITTSService(
                api_key="not-needed",
                base_url=TTS_BASE_URL,
                settings=OpenAITTSService.Settings(model=TTS_MODEL),
            )
            _vad = VADProcessor(
                vad_analyzer=SileroVADAnalyzer(sample_rate=PIPELINE_SAMPLE_RATE),
            )

            _intent_router = IntentRouter(_llm)
            _pipeline = Pipeline([
                _vad, _stt, _asr_cleaner, _user_agg, _context_trimmer,
                _parallel_enricher, _intent_router, _llm,
                _text_tap, _tts_chunker, _tts, _sink, _assistant_agg,
                _auto_memory,
            ])
            _task = PipelineTask(
                _pipeline,
                params=PipelineParams(
                    allow_interruptions=True,
                    audio_in_sample_rate=PIPELINE_SAMPLE_RATE,
                    audio_out_sample_rate=FASTRTC_SAMPLE_RATE,
                    enable_metrics=True,
                    enable_usage_metrics=True,
                ),
                observers=[metrics_observer],
            )
            return _source, _task

        async def _run() -> None:
            nonlocal task, source
            for attempt in range(1, max_attempts + 1):
                runner = PipelineRunner(handle_sigint=False)
                try:
                    feeder = asyncio.create_task(source.run(task), name="pipecat-source-feeder")
                    try:
                        await runner.run(task)
                        return  # clean exit
                    finally:
                        source.stop()
                        feeder.cancel()
                        try:
                            await feeder
                        except asyncio.CancelledError:
                            pass
                except asyncio.CancelledError:
                    logger.info("Pipeline cancelled (attempt %d/%d)", attempt, max_attempts)
                    raise  # don't retry intentional shutdown
                except Exception:
                    logger.exception("Pipeline crashed (attempt %d/%d)", attempt, max_attempts)
                    if attempt < max_attempts:
                        delay = 2 ** (attempt - 1) + random.uniform(0, 0.5)
                        logger.info("Retrying pipeline in %.1fs...", delay)
                        await asyncio.sleep(delay)
                        # Rebuild entire pipeline with fresh stateful components
                        source, task = _build_pipeline()
                        self._source = source
                        self._pipeline_task = task
                    else:
                        logger.error("Pipeline failed after %d attempts", max_attempts)

        self._runner_task = asyncio.create_task(_run(), name="pipecat-pipeline")

        # Robot should be attentive from the start — always listening
        # unless actively speaking.
        self.deps.movement_manager.set_listening(True)

        # DoA speaker tracking (opt-in via ENABLE_DOA_TRACKING=1)
        self._doa_tracker = None
        if os.environ.get("ENABLE_DOA_TRACKING", "0") == "1":
            try:
                from reachy_mini_conversation_app.audio.doa_tracker import DoATracker

                self._doa_tracker = DoATracker(
                    robot=self.deps.reachy_mini,
                    set_offsets=self.deps.movement_manager.set_speech_offsets,
                )
                self._doa_tracker.start()
            except Exception as exc:
                logger.warning("DoA tracker init failed: %s", exc)

        logger.info("PipecatProvider: pipeline started (listening)")

    async def shutdown(self) -> None:
        """Stop the pipecat pipeline gracefully."""
        if getattr(self, "_doa_tracker", None) is not None:
            self._doa_tracker.stop()

        if self._source is not None:
            self._source.stop()

        if self._pipeline_task is not None:
            try:
                await self._pipeline_task.cancel()
            except Exception as exc:
                logger.debug("Pipeline cancel: %s", exc)
            self._pipeline_task = None

        if self._runner_task is not None:
            self._runner_task.cancel()
            try:
                await self._runner_task
            except asyncio.CancelledError:
                pass
            self._runner_task = None

        # Drain output queue
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        logger.info("PipecatProvider: shutdown complete")

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def get_pipeline_health(self) -> dict:
        """Return pipeline health metrics for the dashboard."""
        import time as _time

        now = _time.monotonic()
        return {
            "last_stt_s_ago": round(now - self._last_stt_time, 1) if self._last_stt_time else None,
            "last_llm_s_ago": round(now - self._last_llm_time, 1) if self._last_llm_time else None,
            "last_tts_s_ago": round(now - self._last_tts_time, 1) if self._last_tts_time else None,
            "frames_dropped": self._frames_dropped,
            "barge_in_active": self._barge_in_active,
            "barge_in_generation": self._barge_in_generation,
            "pipeline_alive": self._pipeline_task is not None,
            "ttfb": self._metrics.get("ttfb", {}),
            "ttfb_avg": self._metrics.get("ttfb_avg", {}),
            "processing": self._metrics.get("processing", {}),
            "token_usage": self._metrics.get("token_usage", {}),
            "tts_chars": self._metrics.get("tts_chars", 0),
        }

    def get_detailed_metrics(self) -> dict:
        """Return full metrics including pipeline events (for /api/metrics)."""
        health = self.get_pipeline_health()
        health["pipeline_events"] = self._metrics.get("pipeline_events", [])
        return health

    # ------------------------------------------------------------------
    # Idle behaviour
    # ------------------------------------------------------------------

    async def send_idle_signal(self, idle_duration: float) -> None:
        """Send an idle nudge to the LLM so the robot does something.

        Injects a synthetic system message into the pipeline so the LLM
        can initiate interaction (dance, comment on scene, greet, etc.).

        Tiered behaviour:
          15-45s  → subtle (micro_expression, small head movement)
          45-120s → medium (comment on scene, dance)
          >120s   → reduce frequency (cooldown in emit() handles this)
        """
        if self._pipeline_task is None:
            return

        # Get scene description if vision is available
        scene = ""
        cam = self.deps.camera_worker
        if cam is not None:
            raw = cam.get_latest_frame()
            if raw is not None and self.deps.vision_processor is not None:
                try:
                    desc = await asyncio.to_thread(
                        self.deps.vision_processor.process_image,
                        raw,
                        "Briefly describe the scene and any people visible.",
                    )
                    if desc and desc != "Vision model not initialized":
                        scene = f" You can see: {desc}."
                except Exception:
                    pass

        # Tiered prompts
        if idle_duration < 45:
            prompt = (
                f"[system: You've been idle for {int(idle_duration)}s.{scene} "
                "Do something subtle — a micro_expression, a small head movement, "
                "or just vibe. Keep it natural, don't speak unless you have "
                "something to say.]"
            )
        else:
            prompt = (
                f"[system: You've been idle for {int(idle_duration)}s.{scene} "
                "Do something — dance, comment on what you see, move around, "
                "or start a conversation. Don't be awkward about it.]"
            )

        try:
            from pipecat.frames.frames import (
                InputAudioRawFrame,
                TranscriptionFrame,
            )

            # Inject as a transcription frame (triggers the LLM pipeline)
            frame = TranscriptionFrame(
                text=prompt,
                user_id="system",
                timestamp="",
            )
            await self._pipeline_task.queue_frame(frame)
            logger.info("Idle signal sent (%.0fs idle)", idle_duration)
        except Exception as exc:
            logger.warning("Failed to send idle signal: %s", exc)
