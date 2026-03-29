# R3-MN1 Project Instructions

## Identity

This is a fork of `reachy_mini_conversation_app` (Pollen Robotics) reworked
into R3-MN1: a local-first conversational robot stack.  The OpenAI
Realtime path must remain intact as default/fallback.

## Hard Rules

- **Do not modify `moves.py`** unless explicitly asked.  MovementManager
  behaviour must stay unchanged.
- **Do not modify `openai_realtime.py`** unless explicitly asked.  The
  OpenAI provider path is the stable baseline.
- **Do not hardcode URLs or model names.**  All endpoints and model IDs
  must be read from environment variables or config.  The `.env.r3mn1` file
  is the single source of truth for the local stack.
- **Keep the OpenAI path as default.**  `--provider openai` (or no flag)
  must work exactly as upstream.
- **MCP manager wiring only when safe.**  Do not integrate MCPManager into
  `main.py` until all three servers are verified importable and the
  lifecycle is tested.

## Environment

- Python 3.12, managed via `uv`
- Virtual env at `.venv/`
- Local inference endpoints defined in `.env.r3mn1`
- The local VM IP is set via `LOCAL_VM_IP` env var (default: 192.168.178.155)

## Infrastructure

| Service | IP / URL | Notes |
|---|---|---|
| Inference VM | 192.168.178.155 | Hosts all AI Docker containers |
| LLM — ik-llama.cpp | :3443/v1 | Qwen3.5-35B-A3B-IQ4_XS |
| ASR — qwen-asr (vLLM) | :8015/v1 | Qwen3-ASR |
| TTS — Qwen3-TTS-Openai-Fastapi | :7034/v1 | qwen3-tts / tts-1 |
| OpenMemory (Mem0) | :8765 (API), :3001 (UI) | Conversation memory |
| Reachy Mini | 192.168.178.127 | Robot daemon |
| Home Assistant | 192.168.178.77:8123 | HA instance |

## Architecture

```
providers/
  base.py              — ConversationProvider ABC (no __init__)
  openai_provider.py   — OpenAI Realtime (MRO bridge, empty body)
  pipecat_provider.py  — Local pipeline: STT → LLM → TTS via pipecat-ai

llm/
  intent_classifier.py — Pattern-based intent classification (casual/command/reasoning/creative)
  llm_router.py        — Routes intents to fast or default LLM config

memory/
  mem0_client.py       — Async HTTP client for OpenMemory (Mem0) REST API

transports/
  webrtc_transport.py  — SmallWebRTCTransport routes + session management

tools/
  store_memory.py      — LLM tool: store facts to Mem0
  recall_memory.py     — LLM tool: search memories from Mem0
  (+ core tools: dance, camera, move_head, play_emotion, micro_expression, etc.)

mcp_servers/
  memory_server.py     — MCP memory stubs (not yet wired)
  vision_server.py     — MCP vision stubs (not yet wired)
  robot_server.py      — MCP robot stubs (not yet wired)
```

## Pipeline Order

```
VAD → STT → ASRCleaner → UserAgg(+SmartTurn+MinWords) → ContextTrimmer
  → ParallelEnricher(memory+vision) → IntentRouter → LLM
  → TextTap → TTSChunker(80 chars) → TTS → Sink → AssistantAgg → AutoMemoryTap
```

## Key Files

| File | Role | Touch? |
|---|---|---|
| `main.py` | Entrypoint, `_build_handler()` factory | Minimal edits only |
| `utils.py` | CLI args (`--provider`) | Minimal edits only |
| `moves.py` | MovementManager | **Do not touch** |
| `openai_realtime.py` | OpenAI provider | **Do not touch** |
| `tools/core_tools.py` | Tool registry | Read-only reference |
| `config.py` | Env loading, Config class | Read-only reference |
| `audio/head_wobbler.py` | Speech → head movement | Read-only reference |
| `audio/speech_tapper.py` | Audio signal processing | Read-only reference |

## Audio Contract

- fastrtc operates at **24 kHz** int16 mono (both directions)
- Pipecat pipeline operates at **16 kHz** internally (Silero VAD requirement)
- Resample at the boundary: 24→16 on receive, 16→24 on emit
- HeadWobbler expects **24 kHz** base64-encoded PCM (resample before feeding)

## Testing

- Unit tests: `pytest tests/` (217 tests, all passing)
- Import checks: `python -c "from reachy_mini_conversation_app.providers.pipecat_provider import PipecatProvider"`
- Lint: `ruff check src/`
- `--provider openai` must remain default and fully functional

## Documentation

- `README.md` — User-facing overview and setup
- `docs/R3-MN1.md` — Full technical reference (pipeline, augmentations, configuration)
- `docs/COMPANION_ROADMAP.md` — Implementation tracker for 8 augmentation phases
- `.env.r3mn1` — Reference environment config for local stack
