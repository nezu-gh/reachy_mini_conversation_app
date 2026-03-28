# R3-MN1: Local-First Conversational Robot Stack

R3-MN1 is a fork of [reachy_mini_conversation_app](https://github.com/pollen-robotics/reachy_mini_conversation_app) (Pollen Robotics) reworked into a fully local conversational robot stack. All inference — speech recognition, language model, and text-to-speech — runs on a local VM with no cloud dependencies.

The upstream OpenAI Realtime API path is preserved as the default/fallback provider.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Infrastructure](#infrastructure)
- [Provider System](#provider-system)
- [Pipecat Pipeline](#pipecat-pipeline)
- [MCP Servers](#mcp-servers)
- [Profile System](#profile-system)
- [Configuration](#configuration)
- [Test Pipeline](#test-pipeline)
- [Performance](#performance)
- [Installation](#installation)
- [Usage](#usage)
- [Commit History](#commit-history)
- [Known Issues & TODOs](#known-issues--todos)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  Reachy Mini Robot (192.168.178.127)                                │
│  ┌───────────┐  ┌─────────────┐  ┌───────────────┐                 │
│  │  Camera    │  │  Head (pan/  │  │  Antenna LEDs │                 │
│  │           │  │  tilt/roll)  │  │  + emotions   │                 │
│  └─────┬─────┘  └──────┬──────┘  └───────┬───────┘                 │
│        └───────────────┼──────────────────┘                         │
│                        │                                            │
│              fastrtc audio (24 kHz)                                  │
└────────────────────────┼────────────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │      Provider Abstraction      │
         │  ┌──────────┐ ┌────────────┐  │
         │  │ OpenAI   │ │  Pipecat   │  │
         │  │ Realtime │ │  (local)   │  │
         │  └──────────┘ └─────┬──────┘  │
         └─────────────────────┼─────────┘
                               │
              ┌────────────────┼────────────────┐
              │    Pipecat Pipeline (16 kHz)     │
              │                                  │
              │  VAD → STT → ASRCleaner →        │
              │  UserAgg → LLM → TextTap →       │
              │  TTS → Sink → AssistantAgg       │
              └────────┬────────┬────────┬───────┘
                       │        │        │
              ┌────────┴─┐ ┌───┴───┐ ┌──┴──────┐
              │ ASR      │ │ LLM   │ │ TTS     │
              │ :8015/v1 │ │:3443  │ │:7034/v1 │
              │ Qwen3-ASR│ │Qwen3.5│ │Qwen3-TTS│
              └──────────┘ └───────┘ └─────────┘
                     Local VM (192.168.178.155)
```

---

## Infrastructure

All AI inference runs as Docker containers on a local VM, exposing OpenAI-compatible `/v1` endpoints.

| Service | Address | Model | Notes |
|---------|---------|-------|-------|
| **LLM** | `192.168.178.155:3443/v1` | Qwen3.5-35B-A3B-IQ4_XS | ik-llama.cpp with `--jinja` |
| **ASR/STT** | `192.168.178.155:8015/v1` | Qwen/Qwen3-ASR-0.6B | vLLM (qwen-asr-serve) |
| **TTS** | `192.168.178.155:7034/v1` | qwen3-tts | Qwen3-TTS-Openai-Fastapi |
| **Reachy Mini** | `192.168.178.127` | — | Robot daemon |
| **Home Assistant** | `192.168.178.77:8123` | — | HA instance (future integration) |

---

## Provider System

The provider abstraction (`providers/`) lets the application swap between conversation backends without changing any other code.

### Base Class (`providers/base.py`)

```python
class ConversationProvider(AsyncStreamHandler, ABC):
```

Extends fastrtc's `AsyncStreamHandler` with semantic methods:

| Method | Type | Purpose |
|--------|------|---------|
| `apply_personality(profile)` | abstract | Apply a personality profile at runtime |
| `get_available_voices()` | abstract | Return supported voice list |
| `send_idle_signal(idle_duration)` | default no-op | React when robot is idle |
| `get_tool_specs_override()` | default `None` | Custom tool specs for the provider |

**Design decision:** `ConversationProvider` defines no `__init__` to avoid MRO argument conflicts in the diamond with `AsyncStreamHandler`. Python's C3 MRO deduplicates cleanly.

### OpenAI Provider (`providers/openai_provider.py`)

```python
class OpenAIProvider(OpenaiRealtimeHandler, ConversationProvider):
    pass  # Empty body — MRO resolves all abstract methods
```

Zero-risk bridge. `OpenaiRealtimeHandler` already implements `apply_personality` and `get_available_voices`. This is the default provider (`--provider openai`).

### Pipecat Provider (`providers/pipecat_provider.py`)

```python
class PipecatProvider(ConversationProvider):
```

Fully-local conversation backend powered by pipecat-ai 0.0.108. Raises `RuntimeError` at instantiation if pipecat-ai is not installed.

Key responsibilities:
- **Audio bridging:** Resamples between fastrtc (24 kHz) and pipecat's internal rate (16 kHz)
- **Pipeline lifecycle:** Builds and manages the pipecat Pipeline, PipelineTask, and PipelineRunner
- **Always-listening:** Robot listens continuously; only mutes during TTS playback
- **Barge-in:** Drains output audio queue when user interrupts (VADUserStartedSpeakingFrame)
- **Head wobble:** Feeds TTS output audio into HeadWobbler for speech-synchronized head movement
- **Idle signals:** Placeholder for idle-triggered tool invocations (dance, emotion, etc.)

### Provider Selection

In `main.py`, the `_build_handler()` factory selects the provider:

```python
handler = _build_handler(args.provider, deps, args.gradio, instance_path)
```

CLI flag added to `utils.py`:

```bash
--provider {openai,pipecat}   # default: openai
```

Environment variable alternative (in `.env`):

```bash
PROVIDER=pipecat
```

---

## Pipecat Pipeline

The pipeline is built inside `PipecatProvider.start_up()` and runs as a background asyncio task.

### Pipeline Order

```
VAD → STT → ASRTextCleaner → UserAgg → LLM → TextTap → TTS → Sink → AssistantAgg
```

| Processor | Class | Purpose |
|-----------|-------|---------|
| **VAD** | `VADProcessor` | Silero VAD at 16 kHz; emits `VADUserStarted/StoppedSpeakingFrame` |
| **STT** | `OpenAISTTService` | Qwen3-ASR via OpenAI-compatible API; buffers audio, commits on VAD stop |
| **ASRTextCleaner** | Custom `FrameProcessor` | Strips `<asr_text>` prefix from Qwen ASR transcriptions |
| **UserAgg** | `LLMUserAggregator` | Accumulates user turns into LLM context |
| **LLM** | `OpenAILLMService` | Qwen3.5-35B via ik-llama.cpp; thinking disabled via `chat_template_kwargs` |
| **TextTap** | Custom `FrameProcessor` | Captures assistant text for Gradio chatbot display |
| **TTS** | `OpenAITTSService` | Qwen3-TTS; always outputs 24 kHz |
| **Sink** | Custom `FrameProcessor` | Bridges TTS audio → fastrtc output queue; feeds HeadWobbler |
| **AssistantAgg** | `LLMAssistantAggregator` | Accumulates assistant turns into LLM context |

### Audio Flow

```
Microphone (24 kHz) → fastrtc receive() → resample 24→16 kHz → _audio_in_queue
    → PipelineSource → task.queue_frame(InputAudioRawFrame)
    → [VAD → STT → ... → TTS]
    → PipelineSink → resample if needed → HeadWobbler (24 kHz) → output_queue
    → fastrtc emit() → Speaker (24 kHz)
```

### Thinking Model Handling

Qwen3.5 is a thinking model that streams `reasoning_content` before `content`. Two mechanisms address this:

1. **Thinking disabled at template level:** The LLM is configured with:
   ```python
   extra={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
   ```
   This passes through the OpenAI SDK's `extra_body` mechanism to ik-llama.cpp, where the Jinja chat template checks `enable_thinking` and pre-fills an empty `<think>\n\n</think>\n\n` block — the model skips reasoning entirely.

2. **Native pipecat fallback:** Even if thinking is enabled, pipecat's `OpenAILLMService` only forwards `delta.content` to downstream processors (line 543 of `base_llm.py`), silently skipping `reasoning_content`. This means thinking tokens never reach TTS — only actual response text does. The tradeoff is latency: the model must finish thinking before content tokens begin streaming.

**Impact:** With thinking disabled, LLM TTFB dropped from **85 seconds to 0.24 seconds**.

### ASR Text Cleaning

Qwen3-ASR prefixes all transcriptions with `<asr_text>` (e.g., `<asr_text>Hello world.`). The `ASRTextCleaner` processor strips this before the text reaches the LLM context, ensuring clean conversation history.

### PipelineSource

A custom `FrameProcessor` with a `run()` coroutine that pulls audio from `_audio_in_queue` and injects it into the pipeline via `task.queue_frame()`. Runs as a separate asyncio task alongside the pipeline runner.

### PipelineSink

Intercepts multiple frame types at the end of the pipeline:

| Frame Type | Action |
|------------|--------|
| `OutputAudioRawFrame` / `TTSAudioRawFrame` | Resample if needed, feed HeadWobbler, push to output queue |
| `TranscriptionFrame` | Push user transcript to output queue (for Gradio) |
| `InterimTranscriptionFrame` | Push partial transcript to output queue |
| `VADUserStartedSpeakingFrame` | Set listening=True, drain output queue (barge-in) |
| `VADUserStoppedSpeakingFrame` | Keep listening=True (stay attentive between utterances) |

---

## MCP Servers

Three stub MCP servers are scaffolded under `mcp_servers/`, using `fastmcp.FastMCP`. These are not yet integrated into the main application.

### Memory Server (`mcp_servers/memory_server.py`)

JSON-file-backed memory for the robot.

| Tool | Purpose |
|------|---------|
| `store_memory(key, value, category)` | Store a key-value memory |
| `recall_memory(key)` | Retrieve a stored memory |
| `list_memories(category)` | List memories, optionally filtered by category |

**TODO:** Replace JSON with SQLite or vector store (ChromaDB/Qdrant); add embedding model for semantic recall.

### Vision Server (`mcp_servers/vision_server.py`)

Camera and VLM integration stubs.

| Tool | Purpose |
|------|---------|
| `describe_scene()` | Capture and describe current camera view |
| `track_face()` | Detect and track faces in camera feed |

**TODO:** Wire `VisionProcessor` (SmolVLM2) and `CameraWorker`; add YOLO face-bbox resource.

### Robot Server (`mcp_servers/robot_server.py`)

Bridge to the existing tool registry in `tools/core_tools.py`.

| Tool | Purpose |
|------|---------|
| (dynamic) | Wraps `get_tool_specs()` from the core tool registry |

**TODO:** Wire `ToolDependencies`; robot connection ownership; `BackgroundToolManager` lifecycle.

### MCP Manager (`mcp_manager.py`)

Lifecycle coordinator that starts/stops all three servers as independent background asyncio tasks using stdio transport.

```python
manager = MCPManager()
await manager.start()   # starts memory, vision, robot servers
await manager.stop()    # cancels all tasks
```

**TODO:** Integrate into `main.py`'s `run()` function; switch to SSE transport for remote access.

---

## Profile System

### r3_mn1 Profile (`profiles/r3_mn1/`)

| File | Purpose |
|------|---------|
| `instructions.txt` | Robot identity, personality, hardware awareness |
| `tools.txt` | Enabled tools: dance, stop_dance, play_emotion, stop_emotion, move_head, camera, do_nothing |
| `r3_mn1.env.example` | Minimal env template pointing to `.env.r3mn1` |

**Key personality traits:**
- Self-aware about local inference setup (knows it runs Qwen3 locally)
- Concise — prefers 1-2 sentences
- Honest about offline-only constraints
- Dry wit, no performative enthusiasm
- Hardware-aware: has camera and articulated head, no arms or mobile base

Activate with:
```bash
REACHY_MINI_CUSTOM_PROFILE=r3_mn1
```

---

## Configuration

### Environment Variables

All endpoints and model IDs are read from environment variables. The canonical config file is `.env.r3mn1` at the repo root.

```bash
cp .env.r3mn1 .env   # or: ln -s .env.r3mn1 .env
```

| Variable | Default | Purpose |
|----------|---------|---------|
| `PROVIDER` | `openai` | Conversation backend (`openai` or `pipecat`) |
| `REACHY_MINI_CUSTOM_PROFILE` | — | Profile name (e.g., `r3_mn1`) |
| `LOCAL_VM_IP` | `192.168.178.155` | IP of inference VM |
| `LLM_BASE_URL` | `http://{VM}:3443/v1` | LLM endpoint |
| `MODEL_NAME` | (GGUF path) | LLM model ID (as returned by `/v1/models`) |
| `ASR_BASE_URL` | `http://{VM}:8015/v1` | ASR/STT endpoint |
| `ASR_MODEL` | `Qwen/Qwen3-ASR-0.6B` | ASR model ID |
| `TTS_BASE_URL` | `http://{VM}:7034/v1` | TTS endpoint |
| `TTS_MODEL` | `qwen3-tts` | TTS model ID |
| `LOCAL_VISION_MODEL` | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | Vision model for MCP server |
| `HA_URL` | `http://192.168.178.77:8123` | Home Assistant URL |
| `HA_TOKEN` | — | HA long-lived access token |

### pyproject.toml

New optional dependency group:

```toml
[project.optional-dependencies]
local_pipeline = [
    "pipecat-ai[silero,whisper,ollama,kokoro]>=0.0.49",
    "fastmcp>=2.0.0",
]
```

---

## Test Pipeline

`test_pipeline.py` is a standalone smoke test that exercises the full STT→LLM→TTS pipeline without the robot.

### What It Does

1. **Generates speech** via the TTS endpoint ("Hello, can you hear me? My name is Manfred.")
2. **Resamples** from TTS output (24 kHz) to pipeline rate (16 kHz)
3. **Feeds audio** through the full pipecat pipeline: silence → speech → trailing silence
4. **Collects results** via three specialized FrameProcessor collectors:
   - `EarlyCollector`: STT transcripts and VAD events (positioned after STT)
   - `LLMCollector`: LLM text output (positioned between LLM and TTS)
   - `TTSCollector`: TTS audio frames (positioned after TTS)
5. **Reports** pass/fail for each pipeline stage

### Running

```bash
.venv/bin/python test_pipeline.py
```

Requires all three inference endpoints to be running on the local VM.

### Expected Output

```
PIPELINE TEST RESULTS
  VAD events:    ['start', 'stop', 'start', 'stop']
  Transcripts:   ['<asr_text>Hello. Can you hear me?', '<asr_text>My name is Manfred.']
  LLM response:  Hello, Manfred! I can hear you perfectly, and I am R3-MN1. ...
  TTS output:    17 frames, 368640 bytes

  OK: Full pipeline STT→LLM→TTS working end-to-end!
```

---

## Performance

Measured with `test_pipeline.py` against the local inference VM:

| Stage | Metric | Value |
|-------|--------|-------|
| STT | TTFB | 0.3–0.6s |
| STT | Processing | 0.08–0.4s |
| LLM | TTFB | **0.24s** (thinking disabled) |
| LLM | Total generation | 3.9s |
| TTS | TTFB | 0.6s |
| TTS | Processing | 0.6–1.8s per sentence |
| **End-to-end** | **First audio after user stops speaking** | **~2.4s** |

For comparison, with thinking enabled (no `chat_template_kwargs`), LLM TTFB was **85 seconds** because the model generated ~1700 thinking tokens at 9.3 tok/s before producing any content.

---

## Installation

```bash
# Clone
git clone https://github.com/nezu-gh/reachy_mini_conversation_app.git r3-mn1
cd r3-mn1

# Create venv and install with local pipeline extras
uv venv
uv pip install -e '.[local_pipeline]'

# Configure environment
cp .env.r3mn1 .env
# Edit .env if your VM IP differs from 192.168.178.155

# Verify endpoints
curl http://192.168.178.155:3443/v1/models   # LLM
curl http://192.168.178.155:8015/v1/models   # ASR
curl http://192.168.178.155:7034/v1/models   # TTS

# Run the pipeline smoke test (no robot needed)
.venv/bin/python test_pipeline.py

# Run with robot
.venv/bin/python -m reachy_mini_conversation_app.main --provider pipecat
```

---

## Usage

```bash
# Default: OpenAI Realtime (unchanged from upstream)
python -m reachy_mini_conversation_app.main

# Local pipeline
python -m reachy_mini_conversation_app.main --provider pipecat

# With custom profile
REACHY_MINI_CUSTOM_PROFILE=r3_mn1 python -m reachy_mini_conversation_app.main --provider pipecat

# With Gradio UI
python -m reachy_mini_conversation_app.main --provider pipecat --gradio

# Smoke test (no robot)
.venv/bin/python test_pipeline.py
```

---

## Commit History

All changes from the initial upstream fork to the current state:

| # | Commit | Description |
|---|--------|-------------|
| 1 | `f3ffb0c` | `ConversationProvider` ABC and `OpenAIProvider` MRO bridge |
| 2 | `5267589` | `PipecatProvider` stub with deferred pipecat-ai import |
| 3 | `be7384f` | `--provider` CLI flag and `_build_handler()` factory in `main.py` |
| 4 | `0fd10d5` | MCP server scaffolds (memory, vision, robot) |
| 5 | `d3e89a6` | `MCPManager` lifecycle coordinator |
| 6 | `001d025` | `r3_mn1` profile (instructions, tools, env example) |
| 7 | `a9f53cf` | `local_pipeline` optional dependency group in `pyproject.toml` |
| 8 | `650a4e1` | Wire local VM endpoints into `PipecatProvider` |
| 9 | `d141ac1` | Full `PipecatProvider.start_up()` with pipecat pipeline |
| 10 | `1e255e4` | `AGENTS.md` project instructions and `.env.r3mn1` config |
| 11 | `cbddc41` | Reachy Mini and Home Assistant endpoints in env |
| 12 | `a54c9c6` | Fix model IDs from live `/v1/models` responses |
| 13 | `67124a3` | Fix VAD/STT/TTS flow (standalone VADProcessor, correct frame types) |
| 14 | `3bbdb0c` | Always-listening mode, barge-in, TTS muting |
| 15 | `c802b6e` | Fix model IDs, initial `enable_thinking: false` attempt |
| 16 | `a04f6d1` | Remove broken `enable_thinking` hack, document pipecat native handling |
| 17 | `e4c131a` | `chat_template_kwargs` fix (TTFB 85s→0.24s), ASR text cleaner, test fixes |

---

## Known Issues & TODOs

### Active Issues

- **`<asr_text>` prefix**: Qwen3-ASR prepends `<asr_text>` to all transcriptions. The `ASRTextCleaner` processor strips it in the provider pipeline, but the standalone `test_pipeline.py` still shows it (cosmetic — doesn't affect functionality).
- **Robot daemon**: Reachy Mini at 192.168.178.127 needs its daemon running to test the full application (provider + robot movement).
- **Smart Turn timeout**: Pipecat's Smart Turn analyzer adds ~5s delay after the last VAD stop before committing the user turn. This is by design (waits for multi-sentence input) but may feel slow for single-sentence queries.

### TODOs

| Area | Task | Priority |
|------|------|----------|
| **LLM Tools** | Register robot tools via `llm.register_function()` + `FunctionCallProcessor` | High |
| **MCP Integration** | Wire `MCPManager.start()` into `main.py` alongside `MovementManager` | Medium |
| **Home Assistant** | Add HA token, wire state events from robot_server to HA | Medium |
| **Vision** | Wire SmolVLM2 into vision_server, connect camera | Medium |
| **Memory** | Replace JSON store with SQLite/vector DB | Low |
| **TTS Voices** | Query TTS endpoint for available voices in `get_available_voices()` | Low |
| **Profile hot-reload** | Implement `apply_personality()` to rebuild LLMContext without pipeline restart | Low |
| **Idle behavior** | Inject idle nudge messages into LLM context for tool invocation | Low |

### Files NOT Modified

Per project rules, these files remain untouched:

- `moves.py` — MovementManager behavior unchanged
- `openai_realtime.py` — OpenAI provider path intact as stable baseline
- `tools/core_tools.py` — Tool registry read-only reference
- `config.py` — Env loading unchanged
- `audio/head_wobbler.py` — Speech-to-head-movement unchanged
- `audio/speech_tapper.py` — Audio signal processing unchanged
