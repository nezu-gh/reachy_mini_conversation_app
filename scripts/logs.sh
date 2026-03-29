#!/usr/bin/env bash
# Tail logs from the conversation app running on the robot.
#
# Usage:
#   ./scripts/logs.sh              # follow app stdout (default)
#   ./scripts/logs.sh --daemon     # follow daemon logs
#   ./scripts/logs.sh --all        # both app + daemon interleaved
#   ./scripts/logs.sh --sse        # live SSE log stream from dashboard API
#   ./scripts/logs.sh -n 50        # last 50 lines then follow

source "$(dirname "$0")/_common.sh"

LINES=30
MODE="app"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --daemon|-d) MODE="daemon"; shift ;;
        --all|-a)    MODE="all"; shift ;;
        --sse|-s)    MODE="sse"; shift ;;
        -n)          LINES="$2"; shift 2 ;;
        --help|-h)   head -8 "$0" | tail -6; exit 0 ;;
        *)           echo "Unknown option: $1"; exit 1 ;;
    esac
done

case "$MODE" in
    app)
        echo "==> Following app logs on $ROBOT_HOST (Ctrl+C to stop)..."
        # The daemon runs the app as a subprocess. Its stdout/stderr go
        # through the daemon's process tree. Find the app PID and tail
        # /proc/<pid>/fd/1 for stdout, or fall back to process output.
        _ssh "
            APP_PID=\$(pgrep -f 'reachy_mini_conversation_app.main' | tail -1)
            if [[ -z \"\$APP_PID\" ]]; then
                echo 'No conversation app process found.'
                exit 1
            fi
            echo \"Tailing PID \$APP_PID ...\"
            tail -f /proc/\$APP_PID/fd/1 /proc/\$APP_PID/fd/2 2>/dev/null || \
                strace -e write -p \$APP_PID -s 1000 2>&1 | grep -oP '\"[^\"]+\"'
        "
        ;;
    daemon)
        echo "==> Following daemon logs on $ROBOT_HOST (Ctrl+C to stop)..."
        _ssh "
            DAEMON_PID=\$(pgrep -f 'reachy_mini.daemon.app.main' | head -1)
            if [[ -z \"\$DAEMON_PID\" ]]; then
                echo 'No daemon process found.'
                exit 1
            fi
            echo \"Tailing daemon PID \$DAEMON_PID ...\"
            tail -f /proc/\$DAEMON_PID/fd/1 /proc/\$DAEMON_PID/fd/2 2>/dev/null
        "
        ;;
    all)
        echo "==> Following all reachy logs on $ROBOT_HOST (Ctrl+C to stop)..."
        _ssh "
            PIDS=\$(pgrep -f 'reachy_mini' | tr '\n' ' ')
            if [[ -z \"\$PIDS\" ]]; then
                echo 'No reachy processes found.'
                exit 1
            fi
            echo \"Tailing PIDs: \$PIDS\"
            FDS=''
            for pid in \$PIDS; do
                FDS=\"\$FDS /proc/\$pid/fd/1 /proc/\$pid/fd/2\"
            done
            tail -f \$FDS 2>/dev/null
        "
        ;;
    sse)
        echo "==> Streaming dashboard logs from $ROBOT_HOST:7860 (Ctrl+C to stop)..."
        curl -sN "http://$ROBOT_HOST:7860/api/logs" 2>/dev/null | while read -r line; do
            # SSE format: "data: {...}"
            if [[ "$line" == data:* ]]; then
                echo "$line" | sed 's/^data: //' | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(f\"[{d.get('level','?'):>7}] {d.get('name','')}: {d.get('msg','')}\")
except: pass
" 2>/dev/null
            fi
        done
        ;;
esac
