# Companion Roadmap: Natural Personal Robot

Goal: make R3-MN1 feel like a natural, alive personal companion — not a voice assistant bolted onto a robot.

**Status:** All 8 phases implemented and tested. See [R3-MN1.md](R3-MN1.md) for full technical documentation.

---

## Implementation Summary

| Phase | Feature | Status | Files |
|-------|---------|--------|-------|
| **0** | Prompt enrichment (rhythm, emotions, memory, idle) | Done | `profiles/r3_mn1/instructions.txt` |
| **1** | Observability (MetricsObserver, TTFB, /api/metrics) | Done | `pipecat_provider.py`, `dashboard_api.py` |
| **2** | Audio quality (noise filter, smart interruptions) | Done | `pipecat_provider.py`, `pyproject.toml` |
| **3** | Conversation memory (OpenMemory/Mem0) | Done | `memory/mem0_client.py`, `tools/store_memory.py`, `tools/recall_memory.py`, `pipecat_provider.py` |
| **4** | Proactive engagement (idle signals, presence) | Done | `pipecat_provider.py`, `profiles/r3_mn1/instructions.txt` |
| **5** | LLM router (intent classifier, model routing) | Done | `llm/intent_classifier.py`, `llm/llm_router.py`, `pipecat_provider.py` |
| **6** | Parallel pipeline (memory+vision, TTS optimization) | Done | `pipecat_provider.py` |
| **7** | Web interface (SmallWebRTCTransport) | Done | `transports/webrtc_transport.py`, `static/webrtc.html`, `console.py` |

All paths relative to `src/reachy_mini_conversation_app/` unless noted.

---

## Pipeline (Current)

```
VAD → STT → ASRCleaner → UserAgg(+SmartTurn+MinWords) → ContextTrimmer
  → ParallelEnricher(memory+vision) → IntentRouter → LLM
  → TextTap → TTSChunker(80 chars) → TTS → Sink → AssistantAgg → AutoMemoryTap
```

---

## Phase Details

### Phase 0: Prompt Enrichment
Added 4 sections to `profiles/r3_mn1/instructions.txt`:
- **CONVERSATIONAL RHYTHM** — vary pause length, use silence, match energy
- **EMOTIONAL AWARENESS** — read emotional state, use micro-expressions proactively
- **MEMORY** — use store_memory/recall_memory tools naturally
- **IDLE BEHAVIOR** — respond to idle signals with appropriate actions

### Phase 1: Observability
- `_PipelineMetricsObserver` consumes pipecat's MetricsFrame (TTFB, processing, token usage)
- Rolling 20-sample TTFB averages per service
- `GET /api/metrics` endpoint for detailed pipeline metrics
- Pipeline lifecycle events with timestamps

### Phase 2: Audio Quality
- **Noise suppression**: `noisereduce` (stationary, prop_decrease=0.75) in PipelineSource before VAD
- **Smart interruptions**: `MinWordsUserTurnStartStrategy(min_words=3)` prevents backchannel interrupts
- Both gracefully degrade if dependencies missing

### Phase 3: Conversation Memory
- **Mem0 client** (`memory/mem0_client.py`): async HTTP client for OpenMemory REST API
- **Memory tools**: `store_memory` and `recall_memory` for explicit LLM-triggered storage/recall
- **MemoryInjector** (in ParallelEnricher): searches Mem0 per turn, injects as system message, 5s cache
- **AutoMemoryTap**: fire-and-forget store with `infer=True` for automatic fact extraction, 10s rate limit
- Backend: OpenMemory at `192.168.178.155:8765` (API) / `:3001` (UI)

### Phase 4: Proactive Engagement
- `send_idle_signal()` implemented with tiered behavior (15s subtle → 45s medium → 120s reduced)
- Injects `TranscriptionFrame` with context-aware prompts including current scene description
- Profile instructions guide LLM on when/how to act during idle periods

### Phase 5: LLM Router
- **Intent classifier** (`llm/intent_classifier.py`): pattern-based, 4 categories (casual/command/reasoning/creative)
- **LLM router** (`llm/llm_router.py`): routes casual to optional fast model, everything else to default
- **IntentRouter** processor: classifies + logs routing decision per turn
- Env vars: `LLM_FAST_BASE_URL`, `LLM_FAST_MODEL`

### Phase 6: Parallel Enrichment
- **ParallelEnricher**: runs memory search + vision capture via `asyncio.gather()`
- Replaces sequential MemoryInjector → VisionInjector (saves ~latency-of-slower-op per turn)
- **TTS chunk size**: 150 → 80 chars for faster time-to-first-audio

### Phase 7: Web Interface
- **SmallWebRTCTransport**: browser voice chat at `/webrtc`
- Endpoints: `POST /webrtc/offer`, `PATCH /webrtc/offer`, `GET /webrtc/sessions`
- Custom web UI (`static/webrtc.html`): one-button connect, session count, dark theme
- Max 2 concurrent sessions (configurable via `WEBRTC_MAX_CLIENTS`)
- Each session gets its own pipeline (VAD → STT → LLM → TTS)

---

## Tests

| Phase | Test File | Count |
|-------|-----------|-------|
| 1 | `tests/test_pipecat_provider.py` (extended) | 3 |
| 2 | `tests/test_pipecat_provider.py` (extended) | 2 |
| 3 | `tests/test_memory.py` | 8 |
| 4 | `tests/test_pipecat_provider.py` (extended) | 2 |
| 5 | `tests/test_llm_router.py` | 25 |
| 6 | `tests/test_parallel_enricher.py` | 7 |
| 7 | `tests/test_webrtc_transport.py` | 6 |

Total: **217 tests passing** (full suite).

---

## Remaining Work

| Area | Task | Notes |
|------|------|-------|
| LLM Router | Actually switch LLM service settings for fast model | Currently logs only |
| WebRTC | Wire data channel for live transcripts | Web UI shows no transcript yet |
| Memory | Deduplication in AutoMemoryTap | May store near-identical facts |
| Backchanneling | Code-based "mmhmm" during user speech | Complex pipeline changes needed |
| Emotion classifier | Parallel sentiment detection | Beyond prompt-based approach |
