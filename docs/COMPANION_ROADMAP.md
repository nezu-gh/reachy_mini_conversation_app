# Companion Roadmap: Natural Personal Robot

Goal: make R3-MN1 feel like a natural, alive personal companion — not a voice assistant bolted onto a robot.

---

## Table of Contents

1. [Web Interface (SmallWebRTC)](#1-web-interface-smallwebrtc)
2. [LLM Router](#2-llm-router)
3. [Memory System](#3-memory-system)
4. [Proactive Engagement](#4-proactive-engagement)
5. [Emotion Expression](#5-emotion-expression)
6. [Conversational Rhythm](#6-conversational-rhythm)

---

## 1. Web Interface (SmallWebRTC)

### What

Replace the Gradio + fastrtc audio bridge with Pipecat's `SmallWebRTCTransport` — a self-hosted WebRTC transport using `aiortc`. Serve a custom web dashboard that combines voice interaction, conversation transcript, personality settings, and robot status in a single page.

### Why

- Gradio adds latency (audio goes through its abstraction layer before reaching pipecat)
- SmallWebRTC gives native WebRTC with direct browser↔pipeline audio, no intermediary
- We already have a polished design system in `static/style.css` (dark theme, glassmorphism panels)
- We already have personality REST endpoints in `headless_personality_ui.py`
- The existing `PipelineSource`/`PipelineSink` bridge (fastrtc queues → pipecat frames) can be replaced by pipecat's native transport input/output

### Architecture

```
Browser (WebRTC)                    Server (FastAPI + Pipecat)
┌──────────────────┐               ┌─────────────────────────────────┐
│  Mic → MediaTrack ──── SDP ────→ │ POST /api/offer                 │
│  Speaker ← Track  ←── answer ──← │   → SmallWebRTCConnection       │
│                      │           │   → SmallWebRTCTransport         │
│  WebSocket ─────────────────────→│ WS /ws/events                   │
│    ← transcript                  │   ← user/assistant transcripts  │
│    ← robot status                │   ← emotion/tool events         │
│    ← emotion events              │   ← idle/listening state        │
│                      │           │                                 │
│  REST ──────────────────────────→│ /personalities/* (existing)     │
│                                  │ /status (existing)              │
└──────────────────────┘           └─────────────────────────────────┘
```

Two modes coexist:
- **Browser mode**: SmallWebRTCTransport handles audio I/O directly with the browser
- **Robot mode**: LocalStream handles audio I/O with the robot's mic/speaker (unchanged)

Both can run simultaneously — the web interface is for monitoring, personality management, and optional browser-based voice interaction (useful for testing without the physical robot).

### Implementation

**Dependencies:**

```toml
# pyproject.toml additions
"pipecat-ai[webrtc]>=0.0.108",      # SmallWebRTCTransport + aiortc
"pipecat-ai-small-webrtc-prebuilt",  # optional: prebuilt React UI as fallback
```

**Server-side (`web_transport.py`, new file):**

```python
from fastapi import FastAPI, Request
from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport, SmallWebRTCParams

ice_servers = [IceServer(urls="stun:stun.l.google.com:19302")]

def mount_webrtc_routes(app: FastAPI, build_pipeline_fn):
    """Mount WebRTC offer/answer endpoint on the FastAPI app."""

    @app.post("/api/offer")
    async def offer(request: Request):
        body = await request.json()
        conn = SmallWebRTCConnection(ice_servers)
        await conn.initialize(sdp=body["sdp"], type=body["type"])

        # Build pipeline with this connection's transport
        transport = SmallWebRTCTransport(
            webrtc_connection=conn,
            params=SmallWebRTCParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                audio_in_sample_rate=16000,
                audio_out_sample_rate=24000,
            ),
        )
        # build_pipeline_fn creates the same VAD→STT→...→TTS pipeline
        # but uses transport.input()/transport.output() instead of
        # PipelineSource/PipelineSink
        asyncio.create_task(build_pipeline_fn(transport))

        return conn.get_answer()
```

**Client-side (`static/index.html`, extend existing):**

Add a voice panel to the existing settings page. The WebRTC client is ~80 lines of vanilla JS:

```javascript
// static/voice.js
async function startVoice() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const pc = new RTCPeerConnection({
    iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
  });

  // Add mic track
  stream.getTracks().forEach(track => pc.addTrack(track, stream));

  // Receive speaker track
  pc.ontrack = (event) => {
    const audio = document.getElementById("remote-audio");
    audio.srcObject = event.streams[0];
    audio.play();
  };

  // SDP exchange
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  // Wait for ICE gathering
  await new Promise(resolve => {
    if (pc.iceGatheringState === "complete") resolve();
    else pc.onicegatheringstatechange = () => {
      if (pc.iceGatheringState === "complete") resolve();
    };
  });

  const resp = await fetch("/api/offer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      sdp: pc.localDescription.sdp,
      type: pc.localDescription.type,
    }),
  });
  const answer = await resp.json();
  await pc.setRemoteDescription(answer);
}
```

**Event stream (WebSocket or SSE):**

Add a `/ws/events` WebSocket endpoint that pushes:
- User/assistant transcripts (from the TextTap and TranscriptionFrame processors)
- Robot state changes (listening, speaking, idle, emotion playing)
- Tool invocations (dance started, emotion triggered)

```python
# In pipecat_provider.py, the TextTap and Sink processors already
# capture transcripts and audio events. Add a broadcast to connected
# WebSocket clients alongside the existing AdditionalOutputs mechanism.
```

**Dashboard panels (extend `static/index.html`):**

| Panel | Content |
|-------|---------|
| **Voice** | Start/stop button, audio visualizer (Web Audio API), connection status |
| **Transcript** | Live scrolling conversation (user + assistant messages via WebSocket) |
| **Robot Status** | Listening/speaking/idle indicator, current emotion, head pose |
| **Personality** | Already exists — profile selector, instructions editor, tool toggles |
| **Memory** | Future — show stored memories, search, delete (once memory system exists) |

### Integration with main.py

```python
# In run(), after building the handler:
if args.gradio:
    # Existing Gradio path (keep as fallback)
    ...
else:
    # Headless mode: LocalStream for robot audio + WebRTC for browser
    stream_manager = LocalStream(handler, robot, ...)

    # Mount WebRTC routes on the same FastAPI app
    from reachy_mini_conversation_app.web_transport import mount_webrtc_routes
    mount_webrtc_routes(settings_app, handler.build_browser_pipeline)
```

### Open questions

- Do we want browser voice and robot voice simultaneously? (Probably not — one active audio source at a time, with a toggle)
- Should the web interface be accessible from the local network or just localhost?
- Do we want video from the robot's camera streamed to the browser dashboard?

---

## 2. LLM Router

### What

A lightweight intent classifier that routes each user turn to different LLM configurations, optimizing for speed on simple exchanges and full capability on complex ones.

### Why

Brevdev's three-way router (chitchat / vision / agent) is their most interesting pattern. For simple greetings or small talk, we don't need tools, vision processing, or long generation. Routing lets us:
- Cut latency in half for ~60% of interactions (the casual ones)
- Reserve full tool+vision capability for when it's actually needed
- Make the robot feel more responsive in natural conversation

### Architecture

Unlike brevdev (which routes to separate models via NVIDIA API), we route to the **same Qwen3.5 endpoint with different generation configs**:

```
UserAgg → RouterProcessor → VisionInjector → LLM
              │
              ├─ chitchat:  max_tokens=80, no tools, skip vision
              ├─ vision:    max_tokens=200, no tools, inject camera frame
              └─ complex:   max_tokens=500, tools enabled, inject camera if VLM
```

### Implementation

**RouterProcessor (FrameProcessor inside `start_up()`):**

```python
class RouterProcessor(FrameProcessor):
    """Classify user intent and set LLM params per route."""

    # Simple keyword heuristics first, LLM classification as upgrade path
    _CHITCHAT_PATTERNS = re.compile(
        r"^(hi|hello|hey|how are you|what's up|good morning|good night|"
        r"thanks|thank you|bye|goodbye|see you|ok|okay|sure|yeah|yes|no|"
        r"haha|lol|nice|cool|wow|hmm|hm+|oh|ah)\b",
        re.IGNORECASE
    )
    _VISION_PATTERNS = re.compile(
        r"(look|see|watch|show|what do you see|what's in front|"
        r"describe|camera|photo|picture|image|screen|face|person|"
        r"what am i|how do i look|what's this|what is this)",
        re.IGNORECASE
    )

    async def process_frame(self, frame, direction):
        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return

        # Get last user message text
        messages = frame.context.messages
        last_user = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    last_user = content
                elif isinstance(content, list):
                    last_user = " ".join(
                        p.get("text", "") for p in content
                        if p.get("type") == "text"
                    )
                break

        # Classify
        if self._CHITCHAT_PATTERNS.match(last_user.strip()):
            route = "chitchat"
        elif self._VISION_PATTERNS.search(last_user):
            route = "vision"
        else:
            route = "complex"

        # Tag the frame with route metadata for downstream processors
        frame.context._route = route
        await self.push_frame(frame, direction)
```

**VisionInjector respects route:**

```python
# In VisionInjector.process_frame():
route = getattr(frame.context, "_route", "complex")
if route == "chitchat":
    # Skip vision entirely for small talk
    await self.push_frame(frame, direction)
    return
# Otherwise inject camera frame as before
```

**LLM service picks up route:**

The tricky part — pipecat's `OpenAILLMService` doesn't natively support per-request params. Options:
1. **Override `_get_chat_completions()`** in a subclass to read route from context
2. **Swap `extra_body` on the fly** before each LLM call
3. **Use two LLM service instances** and route frames to different pipeline branches

Option 1 is cleanest. Subclass `OpenAILLMService`:

```python
class RoutedLLMService(OpenAILLMService):
    _ROUTE_PARAMS = {
        "chitchat": {"max_tokens": 80, "temperature": 0.8},
        "vision":   {"max_tokens": 200, "temperature": 0.6},
        "complex":  {"max_tokens": 500, "temperature": 0.6},
    }

    async def _get_chat_completions(self, context, messages):
        route = getattr(context, "_route", "complex")
        params = self._ROUTE_PARAMS.get(route, {})
        # Temporarily override settings
        original_max_tokens = self._max_tokens
        self._max_tokens = params.get("max_tokens", original_max_tokens)
        try:
            return await super()._get_chat_completions(context, messages)
        finally:
            self._max_tokens = original_max_tokens
```

### Future upgrade: LLM-based classification

Replace regex with a single-shot LLM call using a small/fast model:

```
Classify this user message into one of: chitchat, vision, complex
User: "{message}"
Route:
```

This could even be done with the same Qwen3.5 endpoint using `max_tokens=5` and a constrained grammar. But regex is good enough to start — fast and zero-latency.

---

## 3. Memory System

### What

Persistent memory that lets the robot remember past conversations, user preferences, personal facts, and build a relationship over time.

### Why

A companion that forgets everything between conversations isn't a companion. Memory enables:
- "Remember I told you about my dog?" → "Yes, Luna the golden retriever"
- Greeting by name, referencing past conversations
- Building personality consistency over time
- Knowing preferences (music taste, communication style, schedule)

### Architecture

```
┌─────────────────────────────────────────┐
│              Memory Layer                │
│                                         │
│  ┌─────────┐  ┌──────────┐  ┌────────┐ │
│  │ SQLite   │  │ Embeddings│  │ Summary│ │
│  │ Store    │←─│ (local)   │  │ Agent  │ │
│  │          │  │           │  │        │ │
│  └────┬─────┘  └──────────┘  └───┬────┘ │
│       │                          │      │
│  memories table              end-of-conv│
│  - key, value, category     summarizer  │
│  - embedding vector                     │
│  - timestamp, access_count              │
└───────────┬─────────────────────────────┘
            │
   Injected into LLM context
   as system message prefix
```

### Implementation

**Storage (`memory/store.py`, new):**

```python
import sqlite3
import json
from datetime import datetime
from pathlib import Path

class MemoryStore:
    def __init__(self, db_path: Path):
        self.db = sqlite3.connect(str(db_path))
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                embedding BLOB,
                created_at TEXT,
                accessed_at TEXT,
                access_count INTEGER DEFAULT 0
            )
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY,
                started_at TEXT,
                ended_at TEXT,
                summary TEXT,
                message_count INTEGER
            )
        """)

    def store(self, key: str, value: str, category: str = "general"):
        now = datetime.utcnow().isoformat()
        self.db.execute(
            "INSERT INTO memories (key, value, category, created_at, accessed_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (key, value, category, now, now),
        )
        self.db.commit()

    def recall(self, query: str, limit: int = 5) -> list[dict]:
        """Substring search for now; upgrade to embedding similarity later."""
        rows = self.db.execute(
            "SELECT key, value, category FROM memories "
            "WHERE key LIKE ? OR value LIKE ? "
            "ORDER BY accessed_at DESC LIMIT ?",
            (f"%{query}%", f"%{query}%", limit),
        ).fetchall()
        # Update access tracking
        for row in rows:
            self.db.execute(
                "UPDATE memories SET access_count = access_count + 1, "
                "accessed_at = ? WHERE key = ?",
                (datetime.utcnow().isoformat(), row[0]),
            )
        self.db.commit()
        return [{"key": r[0], "value": r[1], "category": r[2]} for r in rows]

    def get_recent(self, limit: int = 10) -> list[dict]:
        rows = self.db.execute(
            "SELECT key, value, category FROM memories "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [{"key": r[0], "value": r[1], "category": r[2]} for r in rows]
```

**Integration with pipecat pipeline:**

Two approaches:
1. **Tool-based** (LLM decides when to remember/recall): Add `store_memory` and `recall_memory` as LLM tools alongside the existing robot tools. The LLM calls them when it recognizes something worth remembering.
2. **Automatic injection**: Before each LLM turn, inject recent/relevant memories into the system prompt. Use keyword overlap between the current user message and stored memories to select relevant ones.

Start with both — tools for explicit storage, automatic injection for recall:

```python
# In the system prompt builder:
relevant = memory_store.recall(last_user_message, limit=3)
if relevant:
    memory_context = "You remember: " + "; ".join(
        f"{m['key']}: {m['value']}" for m in relevant
    )
    # Prepend to system message
```

**Conversation summarization:**

At the end of each conversation (on shutdown or after extended idle), summarize the conversation and store it:

```python
async def summarize_conversation(messages: list[dict]) -> str:
    """Ask the LLM to summarize the conversation into key facts."""
    prompt = (
        "Summarize this conversation into key facts worth remembering "
        "about the user. Focus on: name, preferences, interests, "
        "personal details, requests, and anything they asked you to "
        "remember. Return as bullet points.\n\n"
        + "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    )
    # Single LLM call with low max_tokens
    ...
```

### Future: Embeddings

Replace substring search with semantic similarity using a small local embedding model:
- `sentence-transformers/all-MiniLM-L6-v2` (22MB, runs on CPU)
- Store embeddings in SQLite as BLOBs
- Cosine similarity search at query time
- Could also use ChromaDB or Qdrant for the vector store

---

## 4. Proactive Engagement

### What

The robot initiates conversation when idle — comments on what it sees, asks how the user's day is going, suggests activities, or just makes ambient observations.

### Why

A companion that only speaks when spoken to feels like a tool. Proactive engagement makes the robot feel present and alive. The key is doing it tastefully — not annoying, not too frequent, contextually appropriate.

### Implementation

**Idle timer (extend existing `send_idle_signal`):**

The stub already exists in `pipecat_provider.py`. Implementation:

```python
async def send_idle_signal(self, idle_duration: float) -> None:
    """Inject an idle nudge into the LLM context."""
    if not self._pipeline_task or not self._pipeline_running:
        return

    # Tiered idle behavior
    if idle_duration < 30:
        return  # Too soon, don't be annoying
    elif idle_duration < 120:
        # Short idle: visual observation or ambient comment
        nudge = (
            "[System: The user has been quiet for "
            f"{int(idle_duration)}s. You can see them nearby. "
            "Make a brief, natural observation or ask a gentle "
            "question. Keep it to one sentence. Don't be pushy.]"
        )
    elif idle_duration < 300:
        # Medium idle: suggest an activity
        nudge = (
            "[System: It's been a few minutes of silence. "
            "Suggest something fun — a dance, a joke, or ask "
            "about their day. Be casual and brief.]"
        )
    else:
        # Long idle: ambient mode
        nudge = (
            "[System: Extended idle period. If the user is still "
            "nearby, make one gentle comment. If they seem busy, "
            "stay quiet. You can also do a small idle animation.]"
        )

    # Inject into pipeline
    context = LLMContext(messages=[
        {"role": "system", "content": self._system_prompt},
        {"role": "user", "content": nudge},
    ])
    frame = LLMContextFrame(context=context)
    await self._pipeline_task.queue_frame(frame)
```

**Idle detection in LocalStream:**

```python
# In console.py, track when the last user audio was received
self._last_user_audio = time.monotonic()

# In record_loop():
if audio_frame is not None:
    self._last_user_audio = time.monotonic()

# Separate idle check task:
async def idle_check_loop(self):
    while not self._stop_event.is_set():
        await asyncio.sleep(10)
        idle = time.monotonic() - self._last_user_audio
        if idle > 30:
            await self.handler.send_idle_signal(idle)
```

**Vision-triggered engagement:**

If the camera sees something interesting (person entering room, new object), trigger a comment:

```python
# Periodic camera check (every 10s during idle):
frame = camera_worker.get_latest_frame()
if frame is not None and vision_processor is not None:
    description = await vision_processor.process_image(
        frame, "Briefly describe any notable changes in the scene."
    )
    if description and "nothing" not in description.lower():
        # Inject as context for proactive comment
        ...
```

### Guardrails

- **Cooldown**: Minimum 60s between proactive utterances
- **Back-off**: If user doesn't respond to 2 proactive attempts, increase cooldown to 5 minutes
- **Time awareness**: Reduce proactivity at night (if time-of-day is available)
- **Presence detection**: Only engage if camera confirms someone is present
- **Interruptible**: All proactive speech should be easily interrupted by user

---

## 5. Emotion Expression

### What

The LLM triggers robot emotions (antenna patterns, head movements) based on conversation content — laughing at jokes, showing curiosity when learning something, expressing concern when the user is upset.

### Why

The robot has a full emotion repertoire (happy, sad, surprised, confused, etc.) via `play_emotion` but it's only triggered by explicit tool calls. A natural companion should express emotions continuously as part of conversation flow, not just when explicitly asked to dance or emote.

### Implementation

**Approach 1: LLM-driven (simple, immediate)**

Add emotion hints to the system prompt:

```
When responding, you may express emotions naturally by calling play_emotion.
Don't overdo it — use emotions sparingly, like a real person would.
- Express happiness when the user shares good news
- Show curiosity (head tilt) when learning something new
- React with surprise to unexpected information
- Show empathy when the user is upset
- Laugh (happy emotion) at genuinely funny moments
Do NOT announce your emotions in text. Just call the tool silently.
```

The LLM already has `play_emotion` as a tool. The system prompt just needs to encourage using it naturally instead of only on explicit request.

**Approach 2: Parallel emotion classifier (more natural, no latency cost)**

Run a lightweight sentiment/emotion classifier on the conversation in parallel with the main LLM response:

```python
class EmotionDetector(FrameProcessor):
    """Detect emotion cues and trigger robot expressions."""

    _EMOTION_MAP = {
        "joy": "happy",
        "surprise": "surprised",
        "sadness": "sad",
        "curiosity": "curious",  # head tilt
        "amusement": "laughing",
        "gratitude": "happy",
    }

    async def process_frame(self, frame, direction):
        if isinstance(frame, TranscriptionFrame):
            # Classify user emotion from their speech
            emotion = self._classify(frame.text)
            if emotion and emotion in self._EMOTION_MAP:
                robot_emotion = self._EMOTION_MAP[emotion]
                # Trigger responsive emotion
                await movement_manager.queue_emotion(robot_emotion)

        elif isinstance(frame, TTSTextFrame):
            # Classify what the robot is about to say
            emotion = self._classify(frame.text)
            if emotion:
                # Express emotion while speaking
                ...

        await self.push_frame(frame, direction)

    def _classify(self, text: str) -> str | None:
        """Keyword-based emotion detection. Upgrade to model later."""
        text_lower = text.lower()
        if any(w in text_lower for w in ("haha", "lol", "funny", "hilarious")):
            return "amusement"
        if any(w in text_lower for w in ("wow", "really?", "no way", "amazing")):
            return "surprise"
        if any(w in text_lower for w in ("sorry", "sad", "unfortunately", "miss")):
            return "sadness"
        if "?" in text and len(text) > 20:
            return "curiosity"
        return None
```

**Start with Approach 1** (zero code change — just update the r3_mn1 profile instructions). Add Approach 2 later for more nuanced, low-latency emotion expression.

---

## 6. Conversational Rhythm

### What

Natural conversational behaviors: backchanneling ("mmhmm", "right"), variable response length, thinking pauses, and natural turn-taking signals.

### Why

Human conversation has rhythm. A robot that always responds with full sentences after a fixed pause feels mechanical. Natural companions:
- Acknowledge mid-sentence ("mmhmm", "yeah", "right")
- Sometimes respond briefly, sometimes at length
- Pause before complex answers (thinking signal)
- Don't always need the last word

### Implementation

**Backchanneling (medium complexity):**

A `BackchannelProcessor` that listens to interim transcriptions and occasionally emits quick TTS responses without interrupting the user's turn:

```python
class BackchannelProcessor(FrameProcessor):
    """Emit brief acknowledgments during user speech."""

    _RESPONSES = ["mmhmm", "right", "yeah", "I see", "okay"]
    _MIN_INTERVAL = 8.0  # seconds between backchannels
    _MIN_WORDS = 15  # minimum user words before backchanneling

    def __init__(self):
        super().__init__()
        self._last_backchannel = 0
        self._word_count = 0

    async def process_frame(self, frame, direction):
        if isinstance(frame, InterimTranscriptionFrame):
            self._word_count += len(frame.text.split())
            now = time.monotonic()

            if (self._word_count >= self._MIN_WORDS
                and now - self._last_backchannel > self._MIN_INTERVAL):
                # Emit a backchannel via a small side-channel TTS
                response = random.choice(self._RESPONSES)
                # Queue a low-priority TTS frame that doesn't
                # interrupt the main conversation flow
                self._last_backchannel = now
                self._word_count = 0

        elif isinstance(frame, TranscriptionFrame):
            # Full transcript = turn ended, reset
            self._word_count = 0

        await self.push_frame(frame, direction)
```

> **Note:** Backchanneling is hard to get right with current pipecat architecture because it would interrupt the VAD/turn-taking flow. This is a stretch goal — start with the system prompt approach.

**System prompt approach (simple, immediate):**

Add to the r3_mn1 profile instructions:

```
Vary your response length naturally:
- Simple acknowledgments: "Yeah." / "Makes sense." / "Hmm, interesting."
- Casual exchanges: 1-2 short sentences
- Substantive topics: As long as needed, but still concise
- Don't pad short answers. "Yeah" is a complete response.

When you need a moment to think about something complex, say so naturally:
"Hmm, let me think about that..." then continue.
```

**Thinking pause (medium complexity):**

When the LLM router classifies a message as "complex", inject a brief filler before the full response:

```python
# In RouterProcessor, for complex route:
if route == "complex":
    filler = random.choice([
        "Hmm, let me think about that...",
        "That's a good question...",
        "Let me consider that...",
    ])
    # Emit filler TTS immediately, then let the LLM response follow
    await self.push_frame(TTSTextFrame(text=filler), direction)
```

This gives the user audio feedback instantly while the LLM generates the full response.

---

## Priority Order

| # | Feature | Effort | Impact | Dependencies |
|---|---------|--------|--------|-------------|
| 1 | **Emotion expression** (prompt-based) | Low | High | Just update r3_mn1 profile |
| 2 | **Conversational rhythm** (prompt-based) | Low | High | Just update r3_mn1 profile |
| 3 | **LLM Router** | Medium | High | New FrameProcessor |
| 4 | **Memory system** | Medium | Very High | New module + tools |
| 5 | **Web interface** | High | Medium | New transport + frontend |
| 6 | **Proactive engagement** | Medium | High | Memory + idle timer |
| 7 | **Backchanneling** (code-based) | High | Medium | Complex pipeline changes |
| 8 | **Emotion classifier** (code-based) | Medium | Medium | New FrameProcessor |

Items 1-2 can be done immediately by updating the profile instructions.
Items 3-4 are the core companion features.
Item 5 (web interface) is independent and can be built in parallel.
Items 6-8 build on top of 3-4.
