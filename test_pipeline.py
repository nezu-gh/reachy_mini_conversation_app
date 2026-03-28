#!/usr/bin/env python3
"""Smoke test for the pipecat STT→LLM→TTS pipeline without the robot.

1. Generates speech via TTS ("Hello, my name is R3-MN1")
2. Feeds that audio back through the full pipeline (VAD→STT→LLM→TTS)
3. Prints transcripts, LLM response, and TTS audio stats

Usage:
    .venv/bin/python test_pipeline.py
"""

import os
import sys
import wave
import asyncio
import struct
import numpy as np
from dotenv import load_dotenv

load_dotenv()

_VM = os.environ.get("LOCAL_VM_IP", "192.168.178.155")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", f"http://{_VM}:3443/v1")
LLM_MODEL = os.environ.get("MODEL_NAME", "/models/Qwen3.5-35B-A3B-GGUF/IQ4_XS/Qwen3.5-35B-A3B-IQ4_XS-00001-of-00002.gguf")
TTS_BASE_URL = os.environ.get("TTS_BASE_URL", f"http://{_VM}:7034/v1")
TTS_MODEL = os.environ.get("TTS_MODEL", "qwen3-tts")
ASR_BASE_URL = os.environ.get("ASR_BASE_URL", f"http://{_VM}:8015/v1")
ASR_MODEL = os.environ.get("ASR_MODEL", "Qwen/Qwen3-ASR-0.6B")

PIPELINE_SR = 16000
TTS_SR = 24000


def generate_test_audio() -> str:
    """Use the TTS endpoint to generate a real speech WAV file."""
    import urllib.request
    import json

    wav_path = "/tmp/r3mn1_test_speech.wav"
    req = urllib.request.Request(
        f"{TTS_BASE_URL}/audio/speech",
        data=json.dumps({
            "model": TTS_MODEL,
            "input": "Hello, can you hear me? My name is Manfred.",
            "voice": "alloy",
            "response_format": "wav",
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    print("Generating test speech via TTS endpoint...")
    with urllib.request.urlopen(req, timeout=15) as resp:
        with open(wav_path, "wb") as f:
            f.write(resp.read())

    with wave.open(wav_path) as w:
        dur = w.getnframes() / w.getframerate()
        print(f"  Generated {dur:.1f}s of speech at {w.getframerate()} Hz")
    return wav_path


def load_and_resample(wav_path: str, target_sr: int) -> np.ndarray:
    """Load a WAV file and resample to target_sr."""
    from scipy.signal import resample as scipy_resample

    with wave.open(wav_path) as w:
        sr = w.getframerate()
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)
        audio = np.array(struct.unpack(f"<{n_frames}h", raw), dtype=np.int16)

    if sr != target_sr:
        n_out = int(len(audio) * target_sr / sr)
        audio = scipy_resample(audio, n_out).astype(np.int16)
    return audio


async def main():
    from pipecat.frames.frames import (
        EndFrame,
        InputAudioRawFrame,
        InterimTranscriptionFrame,
        LLMTextFrame,
        OutputAudioRawFrame,
        TextFrame,
        TranscriptionFrame,
        TTSAudioRawFrame,
        VADUserStartedSpeakingFrame,
        VADUserStoppedSpeakingFrame,
    )
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.processors.audio.vad_processor import VADProcessor
    from pipecat.services.openai.llm import OpenAILLMService
    from pipecat.services.openai.stt import OpenAISTTService
    from pipecat.services.openai.tts import OpenAITTSService
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair,
    )

    print(f"LLM: {LLM_MODEL}")
    print(f"     @ {LLM_BASE_URL}")
    print(f"STT: {ASR_MODEL} @ {ASR_BASE_URL}")
    print(f"TTS: {TTS_MODEL} @ {TTS_BASE_URL}")
    print()

    # Step 1: Generate real speech audio
    wav_path = generate_test_audio()
    audio_16k = load_and_resample(wav_path, PIPELINE_SR)
    print(f"  Resampled to {PIPELINE_SR} Hz: {len(audio_16k)} samples ({len(audio_16k)/PIPELINE_SR:.1f}s)")
    print()

    # ---- Services ----
    stt = OpenAISTTService(
        api_key="not-needed", base_url=ASR_BASE_URL,
        settings=OpenAISTTService.Settings(model=ASR_MODEL),
    )
    llm = OpenAILLMService(
        api_key="not-needed", base_url=LLM_BASE_URL,
        settings=OpenAILLMService.Settings(
            model=LLM_MODEL,
            system_instruction="You are a friendly robot named R3-MN1. Keep responses to 1-2 sentences.",
            extra={
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            },
        ),
    )
    tts = OpenAITTSService(
        api_key="not-needed", base_url=TTS_BASE_URL,
        settings=OpenAITTSService.Settings(model=TTS_MODEL),
    )
    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer(sample_rate=PIPELINE_SR))

    # ---- Context ----
    context = LLMContext()
    user_agg, assistant_agg = LLMContextAggregatorPair(context)

    # ---- Collectors ----
    # Two collectors: one early (after STT, before LLM) to catch transcripts
    # and VAD events, one late (after TTS) to catch LLM text and TTS audio.
    collected = {"transcripts": [], "llm_text": [], "tts_frames": 0, "tts_bytes": 0, "vad_events": []}

    class EarlyCollector(FrameProcessor):
        """Captures STT transcripts and VAD events (before user_agg)."""
        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if isinstance(frame, TranscriptionFrame):
                collected["transcripts"].append(frame.text)
                print(f"  [STT] {frame.text}")
            elif isinstance(frame, InterimTranscriptionFrame):
                print(f"  [STT partial] {frame.text}")
            elif isinstance(frame, VADUserStartedSpeakingFrame):
                collected["vad_events"].append("start")
                print("  [VAD] speech started")
            elif isinstance(frame, VADUserStoppedSpeakingFrame):
                collected["vad_events"].append("stop")
                print("  [VAD] speech stopped")
            await self.push_frame(frame, direction)

    class LLMCollector(FrameProcessor):
        """Captures LLM text output (between LLM and TTS)."""
        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if isinstance(frame, (TextFrame, LLMTextFrame)):
                collected["llm_text"].append(frame.text)
                print(f"  [LLM] {frame.text}", end="", flush=True)
            await self.push_frame(frame, direction)

    class TTSCollector(FrameProcessor):
        """Captures TTS audio (after TTS)."""
        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if isinstance(frame, (OutputAudioRawFrame, TTSAudioRawFrame)):
                collected["tts_frames"] += 1
                collected["tts_bytes"] += len(frame.audio)
            await self.push_frame(frame, direction)

    early = EarlyCollector()
    llm_col = LLMCollector()
    tts_col = TTSCollector()

    pipeline = Pipeline([vad, stt, early, user_agg, llm, llm_col, tts, tts_col, assistant_agg])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            audio_in_sample_rate=PIPELINE_SR,
            audio_out_sample_rate=TTS_SR,
            enable_metrics=True,
        ),
    )

    runner = PipelineRunner(handle_sigint=False)
    runner_task = asyncio.create_task(runner.run(task))

    # Wait for pipeline to initialize
    await asyncio.sleep(1.5)

    # Step 2: Feed the speech audio through the pipeline
    chunk_ms = 20
    chunk_samples = PIPELINE_SR * chunk_ms // 1000  # 320 samples

    # Lead with 0.3s silence so VAD has a baseline
    silence = np.zeros(chunk_samples, dtype=np.int16)
    print("Feeding audio: 0.3s silence...")
    for _ in range(300 // chunk_ms):
        await task.queue_frame(InputAudioRawFrame(
            audio=silence.tobytes(), sample_rate=PIPELINE_SR, num_channels=1
        ))
        await asyncio.sleep(0.002)

    # Feed the real speech
    total_chunks = len(audio_16k) // chunk_samples
    print(f"Feeding audio: {len(audio_16k)/PIPELINE_SR:.1f}s of speech ({total_chunks} chunks)...")
    for i in range(total_chunks):
        chunk = audio_16k[i * chunk_samples:(i + 1) * chunk_samples]
        await task.queue_frame(InputAudioRawFrame(
            audio=chunk.tobytes(), sample_rate=PIPELINE_SR, num_channels=1
        ))
        await asyncio.sleep(0.002)

    # Trailing 1.5s silence to trigger VAD stop
    print("Feeding audio: 1.5s trailing silence...")
    for _ in range(1500 // chunk_ms):
        await task.queue_frame(InputAudioRawFrame(
            audio=silence.tobytes(), sample_rate=PIPELINE_SR, num_channels=1
        ))
        await asyncio.sleep(0.002)

    # Step 3: Wait for pipeline to process
    print("\nWaiting for response (up to 45s)...")
    for i in range(90):
        await asyncio.sleep(0.5)
        if collected["tts_frames"] > 0:
            # TTS started — wait for it to finish
            await asyncio.sleep(5)
            break
        if collected["llm_text"]:
            # LLM responded but TTS hasn't started yet — keep waiting
            continue
        if i % 10 == 9:
            print(f"  ...{(i+1)//2}s elapsed")

    # Shutdown
    await task.queue_frame(EndFrame())
    try:
        await asyncio.wait_for(runner_task, timeout=5)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        runner_task.cancel()
        try:
            await runner_task
        except asyncio.CancelledError:
            pass

    # ---- Results ----
    print("\n" + "=" * 60)
    print("PIPELINE TEST RESULTS")
    print("=" * 60)
    print(f"  VAD events:    {collected['vad_events']}")
    print(f"  Transcripts:   {collected['transcripts']}")
    print(f"  LLM response:  {''.join(collected['llm_text'])}")
    print(f"  TTS output:    {collected['tts_frames']} frames, {collected['tts_bytes']} bytes")

    ok = True
    if not collected["vad_events"]:
        print("\n  FAIL: VAD never triggered")
        ok = False
    if not collected["transcripts"]:
        print("\n  FAIL: No transcription from ASR")
        ok = False
    if not collected["llm_text"]:
        print("\n  FAIL: No LLM response")
        ok = False
    if collected["tts_frames"] == 0:
        print("\n  FAIL: No TTS audio generated")
        ok = False

    if ok:
        print("\n  OK: Full pipeline STT→LLM→TTS working end-to-end!")
    else:
        print("\n  Some stages failed — check endpoint logs above")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
