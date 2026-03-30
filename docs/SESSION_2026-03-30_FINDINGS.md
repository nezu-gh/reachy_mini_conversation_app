# Session Findings — 2026-03-30

## What We Tried to Solve

Get the robot (Reachy Mini + local Pipecat pipeline) to have responsive
conversations. Target: < 2 second end-to-end latency.

## What We Achieved

- **VAD fixed**: Was completely dead (min_volume=0.6 + pyloudnorm bug).
  Now triggers on speech with min_volume=0.0.
- **Profile loading fixed**: Project .env now overrides stale instance .env.
- **LLM thinking mode fixed**: extra_body wrapper correctly disables Qwen3.5
  thinking (TTFB dropped from 12s to 0.7s for first token).
- **Lean pipeline mode**: Strips 14→8 processors, 5000→83 prompt tokens,
  no tools. Pipeline infrastructure works correctly.
- **Dashboard model name fixed**: Shows actual Qwen3.5 model, not "gpt-realtime".
- **Micro-expression tool results**: No longer spoken as text by TTS.

## What We Did NOT Solve

**30+ second end-to-end latency persists** despite lean pipeline (83 prompt
tokens, no tools). The LLM metrics show TTFB 3.6s + processing 4.6s = ~5s
for the LLM alone. But the user measures 30-53 seconds wall-clock.

### The Missing 25-48 Seconds

The gap between pipeline metrics (~5s) and wall-clock time (~30-53s)
is NOT in the LLM. It's in the **audio path and turn detection**:

1. **VAD triggering delay**: Even with min_volume=0.0, the VAD may not
   trigger reliably on this mic. The logs show long gaps between barge_in
   events and "Emit watchdog: no output for 30s" warnings.

2. **SmartTurn V3 latency**: Pipecat's learned turn detector
   (LocalSmartTurnAnalyzerV3) adds unknown delay before deciding the user
   has finished speaking. On an RPi 4, the ONNX model inference may be slow.

3. **STT accumulation**: Audio is accumulated until VAD says "stopped speaking"
   AND SmartTurn says "end of turn". Only then is the full audio sent to STT.
   If SmartTurn is slow or doesn't trigger, audio keeps accumulating.

4. **Audio output path**: The `emit()` function polls `_output_queue`. TTS
   audio goes through `PipelineSink` → `output_queue` → `emit()` → fastrtc
   → ALSA speaker. Any blocking in this chain adds latency.

5. **"User intervention: flushing player queue"**: These log messages suggest
   the speaker playback queue has stale audio that gets flushed. The robot
   may be playing back old audio while new audio waits.

### Root Cause Hypothesis

The pipecat pipeline was designed for **WebRTC transports** (Daily, browser)
where audio I/O is handled by the transport layer with proper timing. Our
setup uses a **custom bridge** (console.py record_loop → receive() →
PipelineSource → pipeline → PipelineSink → emit() → fastrtc → ALSA) which
adds multiple queue hops and potential blocking points that don't exist in
native pipecat WebRTC transports.

The brevdev reference implementation uses **Daily WebRTC** as the transport —
audio goes directly through WebRTC, not through a custom bridge. This is
likely why their setup is responsive while ours isn't.

## Unsolved Problems

1. **30+ second latency** — primarily in audio path/turn detection, not LLM
2. **VAD reliability** — triggers inconsistently on the robot's mic
3. **SmartTurn overhead** — unknown latency on RPi 4
4. **Audio bridge overhead** — custom console.py → pipecat bridge adds queuing
5. **Choppy playback** — micro-expressions and voice still choppy
6. **Full pipeline too slow** — 5K prompt tokens + 35B model = 5.4 tok/s

## Architecture Comparison

| | brevdev (works) | r3-mn1 (30s latency) |
|---|---|---|
| Transport | Daily WebRTC | Custom bridge (console.py) |
| Audio I/O | WebRTC native | record_loop → queues → ALSA |
| STT | ElevenLabs cloud | Local Qwen-ASR |
| TTS | ElevenLabs cloud | Local Qwen3-TTS |
| LLM | NVIDIA NIM (50+ tok/s) | Qwen3.5-35B IQ4_XS (5-28 tok/s) |
| VAD | Silero defaults | Silero + min_volume workaround |
| Pipeline | 10 processors | 8 (lean) / 14 (full) |
| Turn detect | SmartTurn V3 | SmartTurn V3 (slow on RPi?) |

## Recommendations for Fresh Start

1. **Use pipecat's WebRTC transport directly** — eliminate the custom audio
   bridge (console.py record_loop/emit). Connect via browser or use
   SmallWebRTCTransport. This is how brevdev and pipecat are designed to work.

2. **Use a smaller/faster LLM** — Qwen3.5-7B or a 3B model. The 35B model
   at IQ4_XS is too slow even with lean prompt.

3. **Consider cloud STT/TTS** — ElevenLabs or Deepgram add network latency
   but are much faster than local Qwen models. Or use faster local models
   (Whisper for STT, Kokoro for TTS).

4. **Bypass SmartTurn on RPi** — use simple VAD-based turn detection
   (silence timeout) instead of the learned ONNX model.

5. **Start minimal** — begin with VAD → STT → LLM → TTS, nothing else.
   Add features only after sub-2s latency is confirmed.

## Files Changed This Session

| File | Changes |
|------|---------|
| `providers/pipecat_provider.py` | VAD min_volume, extra_body, lean pipeline mode, max_tokens, tool result suppression |
| `console.py` | Profile loading priority fix |
| `dashboard_api.py` | Model name resolution |
| `moves.py` | Breathing start log |
| `profiles/r3_mn1/tools.txt` | Reduced from 12 to 6 tools |
| `profiles/r3_mn1/instructions_lean.txt` | New: 4-line lean prompt |
| `prompts.py` | Lean prompt loading support |
| `tests/test_pipeline_mode.py` | New: 15 lean mode tests |
| `docs/SESSION_2026-03-30_CHANGES.md` | Change documentation |

## Key Learnings

1. **Don't over-optimize speculatively** — rechunking, aggressive TTS chunking,
   and context trimming all degraded the pipeline. Test on hardware first.

2. **The OpenAI SDK silently drops unknown kwargs** — `extra_body` is required
   to pass arbitrary parameters to the server.

3. **Pipecat's LLMContext crashes on tools=None** — must omit the argument.

4. **pyloudnorm returns -inf on 32ms audio blocks** — breaks pipecat's volume
   gate. This is a pipecat bug upstream.

5. **The real bottleneck isn't the LLM** — it's the audio path. The custom
   console.py bridge adds 25-48 seconds of unexplained latency. A WebRTC
   transport would likely solve this.

6. **prompt token count directly impacts generation speed** — 5K tokens = 5.4
   tok/s, 83 tokens = 16+ tok/s on the same hardware. KV cache attention
   overhead is the dominant factor.
