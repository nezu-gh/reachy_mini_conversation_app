# R3-MN1 Project Instructions

## Identity

This is a fork of `reachy_mini_conversation_app` (Pollen Robotics) being
reworked into R3-MN1: a local-first conversational robot stack.  The OpenAI
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
- **German-aware profile.**  The `r3_mn1` profile should default to German
  when the user speaks German, but respond in the language the user uses.

## Environment

- Python 3.12, managed via `uv`
- Virtual env at `.venv/`
- Local inference endpoints defined in `.env.r3mn1`
- The local VM IP is set via `LOCAL_VM_IP` env var (default: 192.168.178.155)

## Architecture

```
providers/
  base.py              — ConversationProvider ABC (no __init__)
  openai_provider.py   — OpenAI Realtime (MRO bridge, empty body)
  pipecat_provider.py  — Local pipeline: STT → LLM → TTS via pipecat-ai

mcp_servers/
  memory_server.py     — JSON-backed memory (TODO: vector store)
  vision_server.py     — Camera + VLM stubs
  robot_server.py      — Tool registry bridge stubs

mcp_manager.py         — Lifecycle coordinator for MCP servers
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

- Import checks: `python -c "from reachy_mini_conversation_app.providers.pipecat_provider import PipecatProvider"`
- Lint: `ruff check src/`
- Unit tests: `pytest tests/` (requires robot SDK mocks)
- `--provider openai` must remain default and fully functional
