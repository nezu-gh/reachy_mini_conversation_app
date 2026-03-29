# R3-MN1 Current Status (2026-03-29 03:25)

## What Works

### On the Robot (192.168.178.127)
- **App deployed** at `/home/pollen/r3-mn1` using `/venvs/apps_venv`
- **Audio (ears)**: Robot mic captures 16kHz float32 stereo via ALSA dsnoop (`reachymini_audio_src`)
- **Audio (mouth)**: Robot speaker plays via ALSA dmix (`reachymini_audio_sink`)
- **Pipecat pipeline**: Full STT→LLM→TTS chain works end-to-end
  - STT: Qwen-ASR 0.6B @ 192.168.178.155:8015
  - LLM: Qwen3.5-35B-A3B @ 192.168.178.155:3443 (thinking disabled, TTFB ~0.3s)
  - TTS: Qwen3-TTS @ 192.168.178.155:7034
  - VAD: Silero (local on robot)
  - Smart Turn: v3.2 (local on robot)
- **9 tools registered**: dance, stop_dance, play_emotion, stop_emotion, move_head, camera, do_nothing, task_cancel, task_status
- **Robot movements**: Connected via WebSocket to daemon at localhost:8000
- **Head wobbler**: Running, feeds TTS audio energy into head movements
- **Profile**: r3_mn1 loaded with custom instructions

### On the VM (192.168.178.155 / development machine)
- All inference services running (LLM, ASR, TTS)
- Pipeline smoke test passes (`test_pipeline.py`)
- Git repo up-to-date, all changes pushed

## What Doesn't Work Yet

### Camera (eyes)
- **Problem**: GStreamer `unixfdsrc` → Python `appsink` pipeline stuck in PAUSED state
- **Root cause**: The GLib main context doesn't propagate properly in Python threads with `unixfdsrc`
- `gst-launch-1.0` CLI works fine (1280x720 YUY2 30fps from IPC socket)
- SDK 1.5.1's `camera_gstreamer.py` uses `v4l2convert` which can't do YUY2→BGR
- Patched to `videoconvert` but still stuck at PAUSED→PLAYING transition
- **Workaround needed**: Use subprocess-based frame capture, or investigate GLib context threading

### Audio device warnings
- "No Reachy Mini Audio Source/Sink card found" — `_get_audio_device()` fails but falls back to ALSA config via `.asoundrc` (dsnoop/dmix). Audio works despite warnings.

## SDK Version Note
- **Daemon**: reachy-mini 1.5.1 (in `/venvs/mini_daemon`)
- **App**: Downgraded to reachy-mini 1.5.1 (in `/venvs/apps_venv`) to match daemon
- pyproject.toml requires `>=1.6.0` but 1.5.1 works (just a pip warning)

## Key Files Modified
- `src/reachy_mini_conversation_app/providers/pipecat_provider.py` — Full pipecat pipeline with tools, float32 audio fix
- `src/reachy_mini_conversation_app/console.py` — Skip API key check for pipecat, accept any handler type
- `src/reachy_mini_conversation_app/main.py` — Removed `no_media` override for pipecat
- Robot patch: `/venvs/apps_venv/lib/.../camera_gstreamer.py` — Changed v4l2convert→videoconvert

## How to Run

```bash
# On the robot:
ssh pollen@192.168.178.127  # password: root
cd /home/pollen/r3-mn1
/venvs/apps_venv/bin/python -m reachy_mini_conversation_app.main --provider pipecat --no-camera

# With camera (currently broken):
/venvs/apps_venv/bin/python -m reachy_mini_conversation_app.main --provider pipecat

# Update from git:
cd /home/pollen/r3-mn1 && git pull
```

## Next Steps
1. **Fix camera**: Investigate GLib context issue or implement subprocess-based frame capture
2. **Test speech interaction**: Speak to the robot and verify full loop (mic→STT→LLM→TTS→speaker)
3. **Test tool calls**: Ask robot to dance, move head, show emotions
4. **Commit and push** remaining changes
5. **Update docs**
