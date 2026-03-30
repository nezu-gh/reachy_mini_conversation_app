# Session Changes — 2026-03-30

## Summary

Fixed 5 bugs preventing the robot from having conversations, analyzed
brevdev's reference implementation, and created a plan for a lean pipeline mode.

---

## Changes (net diff from b4d23e1)

### 1. VAD volume gate disabled (`pipecat_provider.py`)

**What**: Set `VADParams(min_volume=0.1)` instead of the pipecat default `0.6`.

**Why**: Pipecat's `calculate_audio_volume()` uses pyloudnorm's EBU R128
loudness on 32ms chunks (512 samples). On such short blocks, pyloudnorm can
return `-inf`, which normalizes to volume=0.0 — always below the 0.6 threshold.
This meant `speaking = confidence >= 0.7 AND volume >= 0.6` was **never true**,
regardless of Silero's neural confidence. The VAD was dead.

Setting to 0.1 (not 0.0) filters very quiet ambient noise that would otherwise
cause constant false VAD triggers, wasting CPU on noise transcriptions.

**Tried and reverted**: `min_volume=0.0` worked but caused too many false
triggers on ambient noise, starving the RPi CPU (movement loop dropped from
50+ Hz to 14 Hz).

---

### 2. LLM thinking mode fix (`pipecat_provider.py`)

**What**: Changed `extra={"chat_template_kwargs": ...}` to
`extra={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}`.

**Why**: Pipecat merges `self._settings.extra` directly into the kwargs of
`client.chat.completions.create(**params)`. Without `extra_body`, the
`chat_template_kwargs` key becomes an unknown kwarg that the OpenAI SDK
silently ignores — it never reaches the ik-llama.cpp server. The Qwen3.5
model then spends 12-30+ seconds thinking and returns empty content.

`extra_body` is the OpenAI SDK's mechanism for injecting arbitrary fields
into the HTTP request body. With this wrapper, `chat_template_kwargs` reaches
the server and thinking is correctly disabled (TTFB dropped from 12s to 0.7s).

**History**: Commit 892da7b (previous session) REMOVED the `extra_body`
wrapper thinking it was wrong. It was actually correct — the test pipeline
(`test_pipeline.py`) already had it right.

---

### 3. Profile loading priority (`console.py`)

**What**: Project `.env` profile now takes priority over instance `.env`.

**Why**: `console.py:launch()` loads an instance `.env` (daemon state dir)
with `override=True`, which overwrites env vars from the project `.env`.
A stale `REACHY_MINI_CUSTOM_PROFILE=example` from a previous UI selection
was overriding the user's explicit `r3_mn1` setting in the project `.env`.

**Fix**: Before loading the instance `.env`, save the project-level profile
value. If the project `.env` had an explicit setting, restore it after the
instance `.env` is loaded. This preserves the instance `.env` mechanism for
UI persistence while letting the project `.env` be authoritative.

---

### 4. Dashboard model name (`dashboard_api.py`)

**What**: Health and config endpoints now resolve the actual LLM model name
from `pipecat_provider.LLM_MODEL` instead of raw `MODEL_NAME` env var.

**Why**: `config.py` defaults `MODEL_NAME` to `"gpt-realtime"` (for the
OpenAI provider). The Pipecat provider reads the same env var but has a
different default — the actual Qwen3.5 GGUF path. When `MODEL_NAME` is
unset in `.env`, the dashboard showed "gpt-realtime" while the actual model
was `Qwen3.5-35B-A3B-IQ4_XS-...`. Long GGUF paths are shortened to just
the filename.

---

### 5. Micro-expression tool results not spoken (`pipecat_provider.py`)

**What**: Suppressed `output_queue` broadcast for `micro_expression`,
`play_emotion`, and `do_nothing` tool results.

**Why**: After a tool call, the handler at line 628 pushed
`AdditionalOutputs({"role": "assistant", "content": f"Tool {fn_name}: {result}"})`
to the output queue. For micro_expression, the tool already plays audio
directly via `output_queue`. The result text (`"Tool micro_expression:
{'status': 'played', 'expression': 'happy'}"`) was being broadcast as a
transcript, which could interfere with TTS or appear as spurious text.

Also broadened the inline tool-call regex to catch `[bracket]` and
`(paren)` style patterns in addition to `*markdown*`.

---

### 6. Tool count reduction (`profiles/r3_mn1/tools.txt`)

**What**: Reduced from 12 to 6 active tools (dance, play_emotion, move_head,
micro_expression, store_memory, recall_memory).

**Why**: Each tool spec adds ~75-100 prompt tokens. With 12 tools, that's
~900 tokens of tool schemas that the LLM must process on every turn. At
5.4 tok/s generation speed, this adds significant prompt processing overhead.
Removed: `stop_dance`, `stop_emotion`, `camera`, `do_nothing` (plus
`task_status`, `task_cancel` are auto-loaded and couldn't be removed).

---

### 7. LLM max_tokens cap (`pipecat_provider.py`)

**What**: Added `max_tokens=150` to both LLM creation sites.

**Why**: Without a cap, the model can generate unbounded tokens. At 5.4 tok/s,
a 300-token response takes 55 seconds. Capping at 150 tokens limits worst-case
generation to ~28 seconds. The r3_mn1 profile already instructs "1-2 sentences"
but the model doesn't always comply.

---

### 8. Breathing start log (`moves.py`)

**What**: Added a one-time log when BreathingMove enters phase 2 (continuous).

**Why**: The health endpoint reported `breathing_active=true` but the user saw
no movement. The log confirms whether the breathing move is actually evaluating.
The original 5mm z-amplitude at 0.1 Hz is very subtle but the move IS running —
the issue was likely the robot's motor resolution rather than a code bug.

---

## Tried and Reverted

### Audio rechunking to 512-sample frames
**What**: Buffered 320-sample audio frames into 512-sample chunks before
sending to the pipeline, to align with Silero VAD's analysis window.

**Why reverted**: VAD's `_run_analyzer()` already buffers internally to
512 samples. The rechunking added unnecessary latency and complexity
without any benefit. The test pipeline (`test_pipeline.py`) works fine
with 320-sample frames.

### TTS chunk size 80→40
**Why reverted**: 40 characters is too small for natural speech. Caused
more network round-trips to the TTS service and potentially choppy output.
Reverted to the Phase 6 roadmap value of 80 chars.

### Context trimmer 40→16 turns
**Why reverted**: Too aggressive. After a few exchanges + tool calls, the
context would be trimmed, potentially splitting tool call/result pairs and
losing important conversation state. Reverted to 40.

### Breathing amplitude 5mm→12mm + 3deg pitch
**Why reverted**: User requested original behavior restored.

---

## Root Cause Analysis: Pipeline Latency

The 30-60 second response time is NOT a pipeline code issue. It's a
fundamental hardware/architecture mismatch:

| Factor | brevdev (reference) | r3-mn1 (ours) |
|--------|-------------------|---------------|
| Prompt tokens | ~200 | ~5000 |
| Gen speed | 50-100+ tok/s (NIM) | 5.4 tok/s |
| Tools | 0 | 8 |
| Pipeline processors | 10 | 14 |
| STT/TTS | Cloud (ElevenLabs) | Local (Qwen) |

**Fix**: Lean pipeline mode (see plan above). Strips to 8 processors,
~200 token prompt, no tools. Expected 5x generation speedup from reduced
KV cache attention overhead.
