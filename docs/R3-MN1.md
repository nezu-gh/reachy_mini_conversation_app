# R3-MN1: Local-First Conversational Robot Stack

R3-MN1 is a fork of [reachy_mini_conversation_app](https://github.com/pollen-robotics/reachy_mini_conversation_app) (Pollen Robotics) reworked into a fully local conversational robot stack. All inference — speech recognition, language model, and text-to-speech — runs on a local VM with no cloud dependencies.

The upstream OpenAI Realtime API path is preserved as the default/fallback provider.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Infrastructure](#infrastructure)
- [Provider System](#provider-system)
- [Pipecat Pipeline](#pipecat-pipeline)
- [Pipeline Augmentations](#pipeline-augmentations)
  - [Observability & Metrics](#observability--metrics)
  - [Audio Quality](#audio-quality)
  - [Conversation Memory](#conversation-memory)
  - [Proactive Engagement](#proactive-engagement)
  - [LLM Router](#llm-router)
  - [Parallel Enrichment](#parallel-enrichment)
  - [WebRTC Voice Chat](#webrtc-voice-chat)
- [Micro-Expressions](#micro-expressions)
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

> See also: [Interactive system diagram](scheme.mmd) (Mermaid) for the full component view including both OpenAI and Pipecat provider paths.

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
              │  UserAgg(+SmartTurn) →            │
              │  ContextTrimmer →                 │
              │  ParallelEnricher(mem+vision) →   │
              │  IntentRouter → LLM →             │
              │  TextTap → TTSChunker →           │
              │  TTS → Sink → AssistantAgg →     │
              │  AutoMemoryTap                    │
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
| **Reachy Mini** | `192.168.178.127` | — | Robot daemon (v1.6.0) |
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

Fully-local conversation backend powered by pipecat-ai (tested with 0.0.108, requires >=0.0.49). Raises `RuntimeError` at instantiation if pipecat-ai is not installed.

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
VAD → STT → ASRCleaner → UserAgg(+SmartTurn) → ContextTrimmer
  → ParallelEnricher(memory+vision) → IntentRouter → LLM
  → TextTap → TTSChunker → TTS → Sink → AssistantAgg → AutoMemoryTap
```

| Processor | Class | Purpose |
|-----------|-------|---------|
| **VAD** | `VADProcessor` | Silero VAD at 16 kHz; emits `VADUserStarted/StoppedSpeakingFrame` |
| **STT** | `OpenAISTTService` | Qwen3-ASR via OpenAI-compatible API; buffers audio, commits on VAD stop |
| **ASRTextCleaner** | Custom `FrameProcessor` | Strips `<asr_text>` prefix from Qwen ASR transcriptions |
| **UserAgg** | `LLMUserAggregator` | Accumulates user turns; uses pipecat's built-in Smart Turn v3.2 (ONNX) and `MinWordsUserTurnStartStrategy` (min 3 words to trigger interruption) |
| **ContextTrimmer** | Custom `FrameProcessor` | Trims old messages to prevent OOM; keeps system messages, drops oldest user/assistant turns |
| **ParallelEnricher** | Custom `FrameProcessor` | Runs memory search + vision capture concurrently via `asyncio.gather()`. Injects Mem0 memories as system message and camera frame (base64 JPEG or scene description) into user message. 5s memory cache TTL. |
| **IntentRouter** | Custom `FrameProcessor` | Classifies user text into `casual`/`command`/`reasoning`/`creative` intents. Routes casual intents to optional fast LLM model. Logs routing decisions. |
| **LLM** | `OpenAILLMService` | Qwen3.5-35B via ik-llama.cpp; thinking disabled via `chat_template_kwargs` |
| **TextTap** | Custom `FrameProcessor` | Captures assistant text for Gradio chatbot display |
| **TTSChunker** | Custom `FrameProcessor` | Waterfall text splitting (sentence→clause→phrase→word, max 80 chars) for faster time-to-first-audio |
| **TTS** | `OpenAITTSService` | Qwen3-TTS; always outputs 24 kHz |
| **Sink** | Custom `FrameProcessor` | Bridges TTS audio → fastrtc output queue; feeds HeadWobbler |
| **AssistantAgg** | `LLMAssistantAggregator` | Accumulates assistant turns into LLM context |
| **AutoMemoryTap** | Custom `FrameProcessor` | Fire-and-forget store of conversation turns to Mem0 with `infer=True` for automatic fact extraction. Rate-limited to 1 store per 10s. |

### Audio Flow

```
Microphone (24 kHz) → fastrtc receive() → resample 24→16 kHz → _audio_in_queue
    → PipelineSource → task.queue_frame(InputAudioRawFrame)
    → [VAD → STT → ... → TTS]
    → PipelineSink → resample if needed → HeadWobbler (24 kHz) → output_queue
    → fastrtc emit() → Speaker (24 kHz)
```

### TTS Warm-Up

A background thread fires a short TTS request at startup (`_warm_up_tts()`) to prime the model. This eliminates the cold-start latency on the first real utterance, which can otherwise be 2-3x slower than subsequent requests.

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

## Pipeline Augmentations

Eight augmentation phases layered on top of the base pipeline, each independently deployable and tested.

### Observability & Metrics

**Module:** `_PipelineMetricsObserver` (inner class in `pipecat_provider.py`)

Consumes pipecat's built-in metrics system (`enable_metrics=True`, `enable_usage_metrics=True`) via the observer pattern.

**How it works:**
- Registered as a `PipelineTask` observer — receives callbacks on `on_push_frame()` and `on_pipeline_started()`
- Intercepts `MetricsFrame` containing `TTFBMetricsData`, `ProcessingMetricsData`, `LLMUsageMetricsData`, and `TTSUsageMetricsData`
- Tracks TTFB per service (STT, LLM, TTS) with rolling 20-sample averages
- Stores processing times, token usage, TTS character counts, and pipeline lifecycle events

**API Endpoints:**
- `GET /api/metrics` — detailed metrics (TTFB, averages, processing times, token usage, events)
- `GET /api/health` — health endpoint enriched with pipeline metrics

**Metrics storage structure:**
```python
self._metrics = {
    "ttfb": {},          # Latest TTFB per service
    "ttfb_avg": {},      # Rolling averages
    "processing": {},    # Processing times per service
    "token_usage": {},   # LLM prompt/completion tokens
    "tts_chars": 0,      # Total TTS characters
    "pipeline_events": [], # Lifecycle events with timestamps
}
```

---

### Audio Quality

Two improvements to reduce false triggers and prevent backchannel interruptions.

#### Noise Suppression

**Library:** `noisereduce>=3.0.0` (optional dependency)

Applied in `PipelineSource.run()` before audio enters the VAD:

```python
audio_f = audio.astype(np.float32) / 32767.0
audio_f = nr.reduce_noise(y=audio_f, sr=sr, stationary=True, prop_decrease=0.75)
audio = np.clip(audio_f * 32767, -32768, 32767).astype(np.int16)
```

- Uses stationary noise reduction (assumes consistent background noise)
- `prop_decrease=0.75` preserves speech while reducing ambient noise
- Graceful fallback: if `noisereduce` is not installed, audio passes through unchanged
- Applied per-frame before `InputAudioRawFrame` is queued

#### Smart Interruption Strategy

**Class:** `MinWordsUserTurnStartStrategy` (pipecat built-in)

Prevents single-word backchannel responses ("yeah", "uh-huh", "ok") from interrupting the robot mid-sentence.

```python
user_params = LLMUserAggregatorParams(
    user_turn_strategies=UserTurnStrategies(
        start=[MinWordsUserTurnStartStrategy(min_words=_min_interrupt_words)],
    ),
)
```

- Default: 3 words minimum to trigger interruption (configurable via `MIN_INTERRUPT_WORDS` env var)
- Applied via `LLMContextAggregatorPair` which wraps the user/assistant aggregators
- Works alongside existing `allow_interruptions=True` — interruptions still work, just need more words

---

### Conversation Memory

**Backend:** OpenMemory (Mem0) running on the local VM at `192.168.178.155:8765`

The robot remembers facts across conversations. "Remember I like jazz" → next session: robot references jazz. Facts are automatically extracted by Mem0's inference engine.

#### Architecture

```
User says something    ParallelEnricher           AutoMemoryTap
       │                     │                         │
       │              ┌──────┴──────┐                  │
       │              │ search_memories()              │
       │              │ → GET /api/v1/memories/        │
       │              │ → inject as system msg         │
       │              └─────────────┘                  │
       │                                               │
       │              ┌────────────────────────────────┘
       │              │ After each turn:
       │              │ add_memory(text, infer=True)
       │              │ → POST /api/v1/memories/
       │              │ Rate limited: 1 store per 10s
       │              └────────────────────────────────
```

#### Mem0 Client

**File:** `src/reachy_mini_conversation_app/memory/mem0_client.py`

Async HTTP client wrapping the OpenMemory REST API using `aiohttp`:

| Method | HTTP | Endpoint | Purpose |
|--------|------|----------|---------|
| `add_memory(text)` | POST | `/api/v1/memories/` | Store fact with `infer=True` |
| `search_memories(query, limit)` | GET | `/api/v1/memories/?search_query=...` | Semantic search |
| `list_memories()` | GET | `/api/v1/memories/?user_id=...` | List all |
| `delete_memory(memory_id)` | DELETE | `/api/v1/memories/` | Remove specific memory |

**Env vars:** `MEM0_BASE_URL`, `MEM0_USER_ID` (default: `default`), `MEM0_APP_NAME` (default: `r3mn1`)

#### Memory Tools

Two LLM-callable tools registered in `profiles/r3_mn1/tools.txt`:

| Tool | File | Purpose |
|------|------|---------|
| `store_memory` | `tools/store_memory.py` | Explicitly store a fact the user asks the robot to remember |
| `recall_memory` | `tools/recall_memory.py` | Explicitly search memories by query |

These complement the automatic memory system (ParallelEnricher + AutoMemoryTap) — the LLM can also proactively store/recall via tool calls.

#### Memory Injection (in ParallelEnricher)

- Runs concurrently with vision on each `LLMContextFrame`
- Searches Mem0 with the last user message as query
- Injects matching memories as a system message: `[memories about user: likes jazz, name is Alex]`
- 5-second cache TTL to avoid redundant API calls

#### Auto Memory Tap

- Monitors assistant text output after `AssistantAgg`
- Periodically sends conversation text to Mem0 with `infer=True`
- Mem0 auto-extracts facts — no manual heuristics needed
- Rate-limited: max 1 store per 10 seconds

#### Web UI

Browse and manage stored memories at `http://192.168.178.155:3001` (OpenMemory dashboard).

---

### Proactive Engagement

**Method:** `PipecatProvider.send_idle_signal(idle_duration)`

The robot initiates interaction when idle — subtle at first, more engaging over time.

#### Tiered Idle Behavior

| Duration | Behavior | Implementation |
|----------|----------|----------------|
| 15–45s | Subtle reaction | Micro-expression or small head movement |
| 45s+ | Comment on scene | Describe what it sees, suggest a dance, ask a question |
| 120s+ | Reduced frequency | One prompt every 60s to avoid being annoying |

**How it works:**
1. The idle timer (in `main.py`) calls `send_idle_signal(idle_duration_seconds)`
2. The method builds a context-aware prompt: `"[system: idle for Ns. Scene: {description}. Do something.]"`
3. Injects a `TranscriptionFrame` into the pipeline, triggering the LLM to respond
4. The LLM uses its tools (micro_expression, play_emotion, dance, camera) based on the prompt instructions

#### Vision-Triggered Greetings

When the ParallelEnricher's vision component detects a scene change (e.g., person appears), the idle signal includes the scene description. The profile instructions tell the LLM to greet newcomers naturally.

**Profile section** (`profiles/r3_mn1/instructions.txt`):
```
## IDLE BEHAVIOR
When you receive a [system: idle for ...] message, you are being nudged to
do something because nobody has spoken for a while. ...
```

---

### LLM Router

**Files:** `src/reachy_mini_conversation_app/llm/intent_classifier.py`, `llm_router.py`

Routes simple messages to a fast/small LLM and complex messages to the full model. Falls back to the default model if no fast model is configured.

#### Intent Classifier

Pattern-based classifier (no ML overhead) with 4 categories:

| Intent | Examples | Confidence |
|--------|----------|------------|
| `CASUAL` | "hi", "thanks", "ok", "how are you?" | 0.7–0.9 |
| `COMMAND` | "dance for me", "take a photo", "move head left" | 0.7–0.9 |
| `REASONING` | "explain quantum computing", "why is the sky blue?" | 0.6–0.8 |
| `CREATIVE` | "tell me a joke", "write a poem about robots" | 0.8 |

**Classification logic:**
1. Check casual patterns (exact match on short messages) → CASUAL
2. Short messages (≤3 words): check command → creative → reasoning → CASUAL fallback
3. Longer messages: command hits → COMMAND; creative patterns → CREATIVE; reasoning patterns → REASONING
4. Long messages (>10 words) with no pattern match → REASONING
5. Medium-length, no signal → CASUAL

#### LLM Router

```python
class LLMRouter:
    def route(self, intent: Intent) -> LLMConfig:
        if self.fast and intent == Intent.CASUAL:
            return self.fast  # fast model for greetings/acknowledgments
        return self.default   # full model for everything else
```

**Env vars:** `LLM_FAST_BASE_URL`, `LLM_FAST_MODEL` (both optional — if not set, all intents use the default model)

#### IntentRouter FrameProcessor

Sits in the pipeline between ParallelEnricher and LLM. On each `LLMContextFrame`:
1. Extracts last user message text
2. Runs `classify()` to get intent + confidence
3. Logs the routing decision: `IntentRouter: 'hello' → casual (90%) → small-model`

---

### Parallel Enrichment

**Class:** `ParallelEnricher` (inner class in `pipecat_provider.py`)

Replaces the sequential `MemoryInjector → VisionInjector` pair with a single processor that runs both operations concurrently via `asyncio.gather()`.

#### Why Not ParallelPipeline?

Pipecat's `ParallelPipeline` sends the same frame to multiple branches, but both memory and vision modify the same `LLMContextFrame` (memory adds a system message, vision modifies the user message). `ParallelPipeline`'s frame deduplication would drop one modification. The `asyncio.gather()` approach runs both fetch operations concurrently, then applies results sequentially to the frame — safe and correct.

#### Execution Flow

```
LLMContextFrame arrives
    │
    ├── asyncio.gather()
    │   ├── _fetch_memories(user_text) → search Mem0 API (network I/O)
    │   └── _fetch_vision(frame)       → camera capture + encode (CPU)
    │
    ├── Apply memory result → insert system message at index 1
    ├── Apply vision result → modify last user message (image or description)
    │
    └── push enriched frame downstream
```

**Latency saving:** ~latency-of-slower-operation per turn. If memory takes 50ms and vision takes 100ms, the sequential approach took 150ms; parallel takes ~100ms.

#### TTS Streaming Optimization

`TTSTextChunker.max_chars` reduced from 150 → 80 for faster time-to-first-audio:
- 80 chars ≈ 13 words ≈ one short sentence
- TTS starts producing audio sooner since it doesn't wait for longer text chunks
- Sentence boundary awareness preserved (waterfall split: sentence → clause → phrase → word)

---

### WebRTC Voice Chat

**File:** `src/reachy_mini_conversation_app/transports/webrtc_transport.py`

Browser-based voice conversation via WebRTC — talk to the robot from any phone or laptop without Gradio.

#### Architecture

```
Browser                               Server (port 7860)
┌─────────────────┐                  ┌──────────────────────────────┐
│ getUserMedia()  │                  │ GET /webrtc → HTML UI        │
│   ↓             │                  │                              │
│ RTCPeerConnection ── POST ──────→  │ POST /webrtc/offer           │
│   ↓ SDP offer   │   /webrtc/offer │   → SmallWebRTCRequestHandler│
│                  │ ← SDP answer ── │   → SmallWebRTCConnection    │
│   ↓             │                  │   → SmallWebRTCTransport     │
│ Audio track  ←→ │ ← WebRTC ────→  │   → Pipeline(VAD→STT→LLM→   │
│ (mic+speaker)   │                  │              TTS→transport)  │
│                  │                  │                              │
│ PATCH ──────────────────────────→  │ PATCH /webrtc/offer          │
│ (ICE candidates)│                  │   → trickle ICE              │
└─────────────────┘                  └──────────────────────────────┘
```

#### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/webrtc` | Serve the voice chat web UI |
| POST | `/webrtc/offer` | Handle WebRTC SDP offer, create pipeline |
| PATCH | `/webrtc/offer` | Handle ICE candidate trickle |
| GET | `/webrtc/sessions` | Return active/max session counts |

#### Web UI

Custom HTML/JS page at `static/webrtc.html`:
- One-button connect/disconnect
- Visual status ring (idle → connecting → connected)
- Live session count display
- Microphone permission request via `getUserMedia()`
- Remote audio playback via `ontrack` handler
- STUN server: `stun:stun.l.google.com:19302`

#### Session Management

- Max concurrent sessions: 2 (configurable via `WEBRTC_MAX_CLIENTS`)
- Each WebRTC connection gets its own pipeline (VAD → STT → LLM → TTS)
- Pipelines share the same service endpoints (STT, LLM, TTS URLs) but create independent service instances
- Sessions are tracked in `_active_sessions` dict with async lock
- Automatic cleanup when connection drops or pipeline ends
- Returns HTTP 429 when session limit is reached

#### Env Vars

| Variable | Default | Purpose |
|----------|---------|---------|
| `WEBRTC_MAX_CLIENTS` | `2` | Max concurrent WebRTC sessions |
| `WEBRTC_ICE_SERVERS` | `stun:stun.l.google.com:19302` | Comma-separated STUN/TURN URLs |

#### Usage

Navigate to `http://<robot-ip>:7860/webrtc` in any modern browser. Click "Connect", grant microphone access, and start talking.

---

## Micro-Expressions

Non-verbal sounds the robot plays as quick emotional reactions — faster and more natural than generating full TTS for simple acknowledgments.

### Sound Library (`audio/sound_library.py`)

Procedurally generates 8 expression sounds using sine waves, frequency sweeps, harmonics, vibrato, and smooth amplitude envelopes. All output at 24 kHz int16 mono.

| Expression | Duration | Character |
|------------|----------|-----------|
| `acknowledge` | 260ms | Rising two-note — "mm-hmm" |
| `think` | 500ms | Sustained hum with vibrato |
| `surprise` | 200ms | Quick upward frequency sweep |
| `happy` | 225ms | Bright ascending double chirp |
| `sad` | 350ms | Slow descending tone |
| `curious` | 300ms | Rising inflection (question tone) |
| `laugh` | 360ms | Bouncing alternating pitches |
| `concerned` | 400ms | Low wavering pulse |

**Custom overrides:** Place a WAV file named `<expression>.wav` in `src/reachy_mini_conversation_app/sounds/` to override any procedural default. Extra WAVs with new names are also loaded automatically.

### Micro-Expression Tool (`tools/micro_expression.py`)

LLM-callable tool that:
1. Loads the sound from `SoundLibrary`
2. Feeds the audio to `HeadWobbler` (base64 PCM) for synchronized head movement
3. Pushes 20ms audio chunks to `output_queue` for the robot speaker
4. Returns immediately — non-blocking, interruptible via barge-in

The tool accesses `output_queue` and `head_wobbler` through `ToolDependencies`, which are wired by the pipecat provider after pipeline creation.

### Usage in Profile

The `r3_mn1` profile instructs the LLM to:
- Use `micro_expression` **instead of speech** for simple reactions (e.g., user says "thanks" → play `acknowledge` instead of "You're welcome")
- Use `micro_expression` **before speech** for immediate emotion (e.g., surprising news → play `surprise`, then comment)
- Use `play_emotion` naturally during conversation to express physical emotions (happiness, curiosity, etc.)
- Vary response length: sometimes just a sound, sometimes a short sentence, sometimes a full answer

### Audio Flow for Micro-Expressions

```
LLM tool call → micro_expression("surprise")
    → SoundLibrary.get("surprise") → int16 PCM (200ms)
    → HeadWobbler.feed(base64) → synchronized head movement
    → output_queue.put(20ms chunks) → robot speaker
```

This bypasses the entire TTS pipeline. A micro-expression plays in ~10ms from tool invocation, compared to ~2.4s for the full STT→LLM→TTS path.

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
| `instructions.txt` | Robot identity, personality, hardware awareness, conversational rhythm, emotional awareness, memory instructions, idle behavior |
| `tools.txt` | Enabled tools: dance, stop_dance, play_emotion, stop_emotion, move_head, camera, do_nothing, store_memory, recall_memory |
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
| `MEM0_BASE_URL` | `http://192.168.178.155:8765` | OpenMemory (Mem0) REST API endpoint |
| `MEM0_USER_ID` | `default` | User ID for memory storage/retrieval |
| `MEM0_APP_NAME` | `r3mn1` | App name registered with OpenMemory |
| `LLM_MULTIMODAL` | auto-detected | Force multimodal vision on/off (`1`/`true`/`yes` or `0`/`false`/`no`). Auto-detected from model name if unset. |
| `LLM_FAST_BASE_URL` | — | Optional fast LLM endpoint for casual intents |
| `LLM_FAST_MODEL` | — | Optional fast LLM model name |
| `MIN_INTERRUPT_WORDS` | `3` | Minimum words to trigger speech interruption |
| `WEBRTC_MAX_CLIENTS` | `2` | Max concurrent WebRTC voice sessions |
| `WEBRTC_ICE_SERVERS` | `stun:stun.l.google.com:19302` | STUN/TURN servers for WebRTC |
| `ENABLE_DOA_TRACKING` | `0` | Direction of Arrival speaker tracking (requires ReSpeaker mic array) |
| `REACHY_ROBOT_NAME` | auto-detect | Robot hostname/IP for SDK discovery (also settable via `--robot-name` CLI flag) |
| `HA_URL` | `http://192.168.178.77:8123` | Home Assistant URL |
| `HA_TOKEN` | — | HA long-lived access token |

### pyproject.toml

Optional dependency group for the local pipeline:

```toml
[project.optional-dependencies]
local_pipeline = [
    "pipecat-ai[silero,whisper,ollama,kokoro]>=0.0.49",
    "fastmcp>=2.0.0",
    "noisereduce>=3.0.0",
]
```

`noisereduce` is optional — if not installed, the noise filter gracefully skips.

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
| 18 | `ed31c2b` | Comprehensive R3-MN1 documentation |
| 19 | `913b4cc` | Remote robot connection, `no_media` for pipecat, network mode override |
| 20 | `00a87f9` | Register 9 robot tools with pipecat LLM, fix STT/TTS settings deprecation |
| 21 | `f484494` | Console provider support for non-OpenAI providers, remove no_media for pipecat |
| 22 | `344cbe9` | Fix float32 audio from robot mic (LocalStream compatibility) |
| 23 | `6c55020` | Save deployment status for continuity |
| 24 | `4469734` | Subprocess-based camera frame capture fallback |
| 25 | `a4bc945` | Smart-turn VAD gate and TTS text chunker |
| 26 | `fbc3821` | Per-turn vision injection, built-in smart-turn, fix barge-in |
| 27 | `632f376` | Make VisionInjector model-aware (multimodal vs text-only) |
| 28 | `54d742e` | Audit: clean up dead code, stale TODOs, update docs |
| 29 | `9e8e2be` | Non-verbal micro-expression sounds for natural reactions |
| 30 | `b38639c` | Micro-expressions docs, pipeline docs update |
| 31 | `2343ed1` | Fix PROVIDER env var for daemon-launched mode |
| 32 | `a0facbf` | Load .env from project root when daemon launches from its own CWD |
| 33 | `0f8064c` | Ensure .env is loaded before parse_args reads PROVIDER |
| 34 | `270123f` | ASR noise filter, context trimming, health probes, robustness |
| 35 | `2fa2943` | systemd service for crash recovery and auto-restart |
| 36 | `2231400` | Dropped frames, regex gaps, tool robustness from stack audit |
| 37 | `7c5c766` | Test suite, refactoring, and robustness improvements |
| 38 | `ef96ed4` | DoA speaker tracking and GStreamer backpressure |
| 39 | `1bb5dd1` | Document local pipeline provider, deployment, and new env vars |
| 40 | `b399137` | Vision pipeline hardening — frame copy, head tracker guard, GST env |
| 41 | `2c8b311` | Repair 7 broken tests across vision, console, camera, and config |
| 42 | `d0514cf` | Enable multimodal vision for Qwen3.5 and other VLMs |
| 43 | `2d29569` | Web dashboard with status, controls, logs, and config tabs |
| 44 | `25cf484` | Make dashboard work with ReachyMiniApp base class |
| 45 | `b73c9f7` | Allow LLM_MULTIMODAL=0 to explicitly disable vision injection |
| 46 | `d8cbc14` | Flatten 2-D audio array before resample and auto wake/sleep robot |
| 47 | `5b3b2f9` | Harden pipeline resilience with error boundaries, retry, and timeouts |
| 48 | `9e698ad` | Use gather(return_exceptions=True) instead of asyncio.wait |
| 49 | `e2dea15` | Self-review: full pipeline rebuild, error escalation, non-blocking I/O |
| 50 | `f9b7f3e` | Bounded queues, barge-in generation counter, pipeline health monitoring |
| 51 | `0eea6d7` | Robot operations scripts (logs, services, shell) |
| 52 | `f9b9f63` | Pipecat audit — tool call safety, wobbler guard, full service rebuild |
| 53 | `4758483` | Pipeline augmentations — 8 phases of companion features |

---

## Known Issues & TODOs

### Active Issues

- **`<asr_text>` prefix**: Qwen3-ASR prepends `<asr_text>` to all transcriptions. The `ASRTextCleaner` processor strips it in the provider pipeline, but the standalone `test_pipeline.py` still shows it (cosmetic — doesn't affect functionality).
- **GStreamer required on robot**: System packages needed for mic/speaker/camera. Already installed on the RPi. For other hosts: `sudo apt-get install -y gir1.2-gstreamer-1.0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-pulseaudio gir1.2-gst-plugins-base-1.0 gstreamer1.0-nice`
- **v4l2h264enc broken on RPi 5**: Hardware H264 encoder fails. Must apply `scripts/patch-daemon-camera.sh` on the robot to switch to openh264enc. See [Camera Fix](#camera-fix-v4l2h264enc--openh264enc).
- **Smart Turn timeout**: Pipecat's built-in Smart Turn v3.2 analyzer (bundled ONNX model, 8MB int8) adds ~5s delay after the last VAD stop before committing the user turn. This is by design (waits for multi-sentence input) but may feel slow for single-sentence queries. The analyzer uses Whisper feature extraction on the last 8s of audio to predict turn completion.

### What Works Now

- Full STT→LLM→TTS pipeline (Qwen-ASR + Qwen3.5-35B + Qwen3-TTS) tested end-to-end
- 12 robot tools registered: dance, stop_dance, play_emotion, stop_emotion, move_head, camera, do_nothing, micro_expression, task_cancel, task_status, store_memory, recall_memory
- Robot connected at 192.168.178.127 (movements, emotions, head control)
- Gradio UI at http://0.0.0.0:7860 for browser-based audio interaction
- WebRTC voice chat at http://0.0.0.0:7860/webrtc (no Gradio needed, up to 2 concurrent sessions)
- LLM TTFB: 0.3s (thinking disabled via chat_template_kwargs)
- Barge-in support (allow_interruptions + output queue drain), VAD-driven listening mode, head wobble from TTS audio
- Smart interruption: MinWordsUserTurnStartStrategy prevents single-word backchannel from interrupting (min 3 words)
- Noise suppression via noisereduce (stationary, prop_decrease=0.75) before VAD
- Parallel memory + vision enrichment via asyncio.gather() before each LLM turn
- Conversation memory via OpenMemory (Mem0) — auto-extract facts, semantic search, inject as context
- Intent-based LLM routing (casual/command/reasoning/creative) with optional fast model for casual intents
- Per-turn vision injection (multimodal image for VLMs, text description for text-only LLMs)
- TTS text chunking (waterfall split, max 80 chars for faster TTFB)
- Smart Turn v3.2 turn-completion detection (pipecat built-in ONNX model)
- Non-verbal micro-expressions (8 procedural sounds) for quick emotional reactions, bypassing TTS
- Camera capture via unixfdsrc socket (1280×720 YUY2 from daemon) and subprocess fallback
- Pipeline observability: TTFB tracking, rolling averages, token usage, pipeline events via /api/metrics
- Proactive idle engagement: tiered idle signals (subtle → medium → reduced frequency)
- Prompt-based conversational rhythm and emotional awareness instructions

### Camera Fix (v4l2h264enc → openh264enc)

The daemon's WebRTC pipeline (`webrtc_daemon.py`) uses a GStreamer `tee` to split the camera feed:

```
libcamerasrc → capsfilter → tee ─→ queue → unixfdsink (camera socket)
                                └→ queue → h264enc → webrtcsink (WebRTC)
```

On this RPi 5, `v4l2h264enc` (hardware H264 encoder) fails with "not enough memory or failing driver". This stalls the `tee`, blocking **both** branches — the camera socket gets caps but zero buffers, and WebRTC video never streams.

**Fix:** The patch adds runtime detection — it probes `v4l2h264enc` at startup and falls back to `openh264enc` (software encoder) if the hardware encoder is broken. Apply with:

```bash
ssh pollen@192.168.178.127 'bash -s' < scripts/patch-daemon-camera.sh
```

Then kill the daemon process (API restart is not sufficient — the GStreamer pipeline is created once at init):

```bash
ssh pollen@192.168.178.127 'kill $(pgrep -f "reachy_mini.daemon.app.main")'
# systemd auto-restarts it
```

The patch script (`scripts/patch-daemon-camera.sh`) supports both v1.5.x (`webrtc_daemon.py`) and v1.6.x (`media_server.py`). **Must be re-applied after any `pip install reachy-mini` upgrade.**

### TODOs

| Area | Task | Priority |
|------|------|----------|
| **MCP Integration** | Wire `MCPManager.start()` into `main.py` alongside `MovementManager` | Medium |
| **Home Assistant** | Add HA token, wire state events from robot_server to HA | Medium |
| **Vision MCP** | Wire SmolVLM2 into vision_server MCP tool, connect camera | Medium |
| **TTS Voices** | Query TTS endpoint for available voices in `get_available_voices()` | Low |
| **Profile hot-reload** | Implement `apply_personality()` to rebuild LLMContext without pipeline restart | Low |
| **WebRTC transcript** | Wire pipecat data channel to send live transcripts to web UI | Low |
| **LLM Router** | Actually switch LLM service settings when fast model is configured (currently logs only) | Low |
| **Memory dedup** | Add deduplication to AutoMemoryTap to avoid storing near-identical facts | Low |

### Completed (from Companion Roadmap)

| Area | What | Phase |
|------|------|-------|
| **Prompt enrichment** | Conversational rhythm, emotional awareness, memory, idle behavior instructions | Phase 0 |
| **Observability** | MetricsObserver, TTFB tracking, /api/metrics endpoint | Phase 1 |
| **Audio quality** | Noise suppression (noisereduce), MinWordsUserTurnStartStrategy | Phase 2 |
| **Memory** | Mem0 client, store/recall tools, MemoryInjector, AutoMemoryTap | Phase 3 |
| **Proactive engagement** | Tiered idle signals, vision-triggered greetings | Phase 4 |
| **LLM Router** | Intent classifier, model routing, IntentRouter processor | Phase 5 |
| **Parallel pipeline** | ParallelEnricher (asyncio.gather), TTS chunk size optimization | Phase 6 |
| **Web interface** | SmallWebRTCTransport, /webrtc UI, session management | Phase 7 |

### Files NOT Modified

Per project rules, these files remain untouched:

- `moves.py` — MovementManager behavior unchanged
- `openai_realtime.py` — OpenAI provider path intact as stable baseline
- `tools/core_tools.py` — Tool registry read-only reference
- `config.py` — Env loading unchanged
- `audio/head_wobbler.py` — Speech-to-head-movement unchanged
- `audio/speech_tapper.py` — Audio signal processing unchanged
