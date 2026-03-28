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
import base64
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
LLM_MODEL = os.environ.get("MODEL_NAME", "qwen3.5-35b-a3b")

TTS_BASE_URL = os.environ.get("TTS_BASE_URL", f"http://{_DEFAULT_VM}:7034/v1")
TTS_MODEL = os.environ.get("TTS_MODEL", "tts-1")

ASR_BASE_URL = os.environ.get("ASR_BASE_URL", f"http://{_DEFAULT_VM}:8015/v1")
ASR_MODEL = os.environ.get("ASR_MODEL", "Qwen/Qwen3-ASR-1.7B")

# ---------------------------------------------------------------------------
# Audio constants
# ---------------------------------------------------------------------------
# fastrtc uses 24 kHz; pipecat's Silero VAD requires 16 kHz.
# PipelineParams defaults already match: audio_in=16000, audio_out=24000.
FASTRTC_SAMPLE_RATE = 24000
PIPELINE_SAMPLE_RATE = 16000


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

        # Queues bridging fastrtc ↔ pipecat
        self._audio_in_queue: asyncio.Queue[Tuple[int, NDArray[np.int16]]] = asyncio.Queue()
        self.output_queue: asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs] = asyncio.Queue()

        # Lifecycle
        self._pipeline_task: Any | None = None
        self._runner_task: asyncio.Task[None] | None = None
        self._source: Any | None = None
        self.last_activity_time: float = 0.0
        self.start_time: float = 0.0

    # ------------------------------------------------------------------
    # ConversationProvider abstract methods
    # ------------------------------------------------------------------

    async def apply_personality(self, profile: Optional[str]) -> str:
        """Update LLM system instructions at runtime.

        TODO: rebuild the LLMContext with new instructions from the
        profile and push an updated context frame into the pipeline.
        """
        from reachy_mini_conversation_app.config import set_custom_profile

        set_custom_profile(profile)
        logger.info("PipecatProvider: profile set to %r (live update pending)", profile)
        return f"Profile '{profile}' set. Pipeline restart needed for full effect."

    async def get_available_voices(self) -> List[str]:
        """Return voices available from the local TTS engine.

        TODO: query TTS_BASE_URL/v1/voices or similar for voice list.
        """
        return []

    # ------------------------------------------------------------------
    # AsyncStreamHandler audio contract
    # ------------------------------------------------------------------

    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Accept an audio frame from fastrtc and forward into the pipeline.

        Resamples from fastrtc's 24 kHz to the pipeline's 16 kHz.
        """
        if self._pipeline_task is None:
            return

        input_sr, audio = frame

        # Ensure mono
        if audio.ndim == 2:
            if audio.shape[1] > audio.shape[0]:
                audio = audio.T
            if audio.shape[1] > 1:
                audio = audio[:, 0]

        # Resample to pipeline rate
        if input_sr != PIPELINE_SAMPLE_RATE:
            n_samples = int(len(audio) * PIPELINE_SAMPLE_RATE / input_sr)
            audio = resample(audio, n_samples).astype(np.int16)

        await self._audio_in_queue.put((PIPELINE_SAMPLE_RATE, audio))

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
        """
        idle_duration = asyncio.get_event_loop().time() - self.last_activity_time
        if idle_duration > 15.0 and self.deps.movement_manager.is_idle():
            try:
                await self.send_idle_signal(idle_duration)
            except Exception as exc:
                logger.warning("Idle signal skipped: %s", exc)
                return None
            self.last_activity_time = asyncio.get_event_loop().time()

        return await wait_for_item(self.output_queue)  # type: ignore[no-any-return]

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
            LLMContextAggregatorPair,
        )

        from reachy_mini_conversation_app.prompts import get_session_instructions

        loop = asyncio.get_event_loop()
        self.start_time = loop.time()
        self.last_activity_time = loop.time()

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
            model=ASR_MODEL,
        )

        llm = OpenAILLMService(
            api_key="not-needed",
            base_url=LLM_BASE_URL,
            settings=OpenAILLMService.Settings(
                model=LLM_MODEL,
                system_instruction=get_session_instructions(),
            ),
        )

        # OpenAI TTS always outputs 24 kHz.  Our Qwen3-TTS endpoint is
        # OpenAI-compatible, so it also outputs 24 kHz — no resampling
        # needed between TTS output and the fastrtc contract.
        tts = OpenAITTSService(
            api_key="not-needed",
            base_url=TTS_BASE_URL,
            model=TTS_MODEL,
        )

        # ---- VAD ---------------------------------------------------------
        # Standalone VADProcessor generates VADUserStartedSpeakingFrame /
        # VADUserStoppedSpeakingFrame that both the STT service and the
        # user aggregator listen for.

        vad = VADProcessor(
            vad_analyzer=SileroVADAnalyzer(sample_rate=PIPELINE_SAMPLE_RATE),
        )

        # ---- Tool registration (deferred) --------------------------------
        # TODO: register robot tools with the LLM via llm.register_function()
        # so tool calls flow through pipecat's FunctionCallProcessor.
        # For now the LLM has no tools configured.

        # ---- Context & aggregators ---------------------------------------

        context = LLMContext()

        # No vad_analyzer in user_params — the standalone VADProcessor
        # already broadcasts VAD frames into the pipeline.
        user_agg, assistant_agg = LLMContextAggregatorPair(context)

        # ---- Bridge processors -------------------------------------------
        # PipelineSource: pulls from _audio_in_queue, emits InputAudioRawFrame
        # PipelineSink: catches TTS audio + transcripts, pushes to output_queue

        provider_ref = self  # avoid closure over self in nested classes

        class PipelineSource(FrameProcessor):
            """Pulls audio from the fastrtc inbound queue into the pipeline."""

            def __init__(self) -> None:
                super().__init__()
                self._running = True

            async def process_frame(self, frame: Any, direction: FrameDirection) -> None:
                """Forward frames that arrive from upstream."""
                await self.push_frame(frame, direction)

            async def run(self, task: PipelineTask) -> None:
                """Background loop that feeds mic audio into the pipeline."""
                while self._running:
                    try:
                        sr, audio = await asyncio.wait_for(provider_ref._audio_in_queue.get(), timeout=0.5)
                    except asyncio.TimeoutError:
                        continue

                    audio_bytes = audio.astype(np.int16).tobytes()
                    frame = InputAudioRawFrame(
                        audio=audio_bytes,
                        sample_rate=sr,
                        num_channels=1,
                    )
                    await task.queue_frame(frame)

            def stop(self) -> None:
                self._running = False

        class AssistantTextTap(FrameProcessor):
            """Taps LLM output text before it reaches TTS.

            Emits an AdditionalOutputs with role=assistant so the Gradio
            chatbot (or LocalStream logger) can display what the bot said.
            """

            async def process_frame(self, frame: Any, direction: FrameDirection) -> None:
                if isinstance(frame, TTSTextFrame):
                    text = getattr(frame, "text", "")
                    if text:
                        await provider_ref.output_queue.put(AdditionalOutputs({"role": "assistant", "content": text}))
                await self.push_frame(frame, direction)

        class PipelineSink(FrameProcessor):
            """Intercepts output frames and bridges them to fastrtc."""

            async def process_frame(self, frame: Any, direction: FrameDirection) -> None:
                # TTS audio → output queue + head wobbler
                # While the robot is speaking, suppress listening so
                # idle breathing is allowed and antennas aren't frozen.
                if isinstance(frame, (OutputAudioRawFrame, TTSAudioRawFrame)):
                    audio_bytes = frame.audio
                    sr = frame.sample_rate

                    provider_ref.deps.movement_manager.set_listening(False)
                    provider_ref.last_activity_time = asyncio.get_event_loop().time()

                    # Convert raw PCM bytes → int16 numpy
                    pcm = np.frombuffer(audio_bytes, dtype=np.int16)

                    # Resample to fastrtc 24 kHz if needed (TTS should
                    # already output 24 kHz, but handle mismatches).
                    if sr != FASTRTC_SAMPLE_RATE:
                        n_out = int(len(pcm) * FASTRTC_SAMPLE_RATE / sr)
                        pcm = resample(pcm, n_out).astype(np.int16)

                    # Feed HeadWobbler with 24 kHz PCM (same rate it
                    # expects from the OpenAI path).
                    if provider_ref.deps.head_wobbler is not None:
                        b64 = base64.b64encode(pcm.tobytes()).decode("utf-8")
                        provider_ref.deps.head_wobbler.feed(b64)

                    await provider_ref.output_queue.put((FASTRTC_SAMPLE_RATE, pcm.reshape(1, -1)))

                # Final user transcript
                elif isinstance(frame, TranscriptionFrame):
                    text = getattr(frame, "text", "")
                    if text:
                        logger.debug("User transcript: %s", text)
                        await provider_ref.output_queue.put(AdditionalOutputs({"role": "user", "content": text}))

                # Partial user transcript
                elif isinstance(frame, InterimTranscriptionFrame):
                    text = getattr(frame, "text", "")
                    if text:
                        await provider_ref.output_queue.put(
                            AdditionalOutputs({"role": "user_partial", "content": text})
                        )

                # VAD speech boundaries → listening mode for movement manager.
                # The robot should always be listening when idle — only
                # TTS playback (above) temporarily clears the flag.
                elif isinstance(frame, VADUserStartedSpeakingFrame):
                    provider_ref.deps.movement_manager.set_listening(True)
                    if provider_ref.deps.head_wobbler is not None:
                        provider_ref.deps.head_wobbler.reset()
                    # Barge-in: drain pending audio so the robot stops
                    # talking when the user interrupts.
                    while not provider_ref.output_queue.empty():
                        try:
                            item = provider_ref.output_queue.get_nowait()
                            if isinstance(item, AdditionalOutputs):
                                # Keep transcript outputs, discard audio
                                await provider_ref.output_queue.put(item)
                                break
                        except asyncio.QueueEmpty:
                            break
                    logger.debug("User speech started (barge-in)")

                elif isinstance(frame, VADUserStoppedSpeakingFrame):
                    # Stay in listening mode — robot remains attentive
                    # between utterances.  Listening is only cleared
                    # while TTS audio is actively playing.
                    provider_ref.deps.movement_manager.set_listening(True)
                    logger.debug("User speech stopped (still listening)")

                # TODO: capture FunctionCallResultFrame for tool call
                # dispatch once tools are registered with the LLM.

                # Always forward the frame so downstream processors
                # (like assistant_aggregator) still see it.
                await self.push_frame(frame, direction)

        # ---- Assemble pipeline -------------------------------------------
        # PipelineTask._process_push_queue() sends StartFrame automatically
        # with audio_in/out sample rates from PipelineParams.
        #
        # Pipeline order:
        #   VAD → STT → user_agg → LLM → text_tap → TTS → sink → assistant_agg
        #
        # PipelineSource runs as a separate asyncio task and injects
        # InputAudioRawFrame via task.queue_frame().

        source = PipelineSource()
        text_tap = AssistantTextTap()
        sink = PipelineSink()

        pipeline = Pipeline(
            [
                vad,            # analyse audio for speech boundaries
                stt,            # transcribe speech segments
                user_agg,       # accumulate user turns
                llm,            # generate response
                text_tap,       # capture assistant text before TTS
                tts,            # synthesise speech
                sink,           # bridge audio + transcripts → fastrtc
                assistant_agg,  # accumulate assistant turns
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                audio_in_sample_rate=PIPELINE_SAMPLE_RATE,
                audio_out_sample_rate=FASTRTC_SAMPLE_RATE,
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )
        self._pipeline_task = task
        self._source = source

        # Kick off the pipeline in a background asyncio task
        runner = PipelineRunner(handle_sigint=False)

        async def _run() -> None:
            try:
                # Start the source feeder alongside the runner
                feeder = asyncio.create_task(source.run(task), name="pipecat-source-feeder")
                try:
                    await runner.run(task)
                finally:
                    source.stop()
                    feeder.cancel()
                    try:
                        await feeder
                    except asyncio.CancelledError:
                        pass
            except Exception:
                logger.exception("PipecatProvider pipeline crashed")

        self._runner_task = asyncio.create_task(_run(), name="pipecat-pipeline")

        # Robot should be attentive from the start — always listening
        # unless actively speaking.
        self.deps.movement_manager.set_listening(True)
        logger.info("PipecatProvider: pipeline started (listening)")

    async def shutdown(self) -> None:
        """Stop the pipecat pipeline gracefully."""
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
    # Idle behaviour
    # ------------------------------------------------------------------

    async def send_idle_signal(self, idle_duration: float) -> None:
        """Send an idle nudge to the LLM so the robot does something.

        TODO: inject a user-turn message into the LLMContext and queue
        an LLMRunFrame so the model can invoke a tool (dance, emotion,
        etc.).  For now this is a no-op — idle tools require the tool
        registry to be wired first.
        """
