#!/usr/bin/env bash
# Deploy the conversation app to the Reachy Mini robot.
#
# Usage:
#   ./scripts/deploy.sh              # deploy + restart
#   ./scripts/deploy.sh --no-restart # deploy only (skip app restart)
#   ./scripts/deploy.sh --status     # just check app status
#
# Reads ROBOT_HOST and ROBOT_PASS from .env.r3mn1 or environment.
# Defaults: pollen@192.168.178.127, password: root

set -euo pipefail
source "$(dirname "$0")/_common.sh"

status() {
    echo "==> Checking app status on $ROBOT_HOST..."
    _daemon_api "/api/apps/current-app-status"
    echo
    # Try the settings/dashboard health endpoint (port 7860)
    echo "==> Pipeline health:"
    _ssh "curl -sf http://localhost:7860/api/health 2>/dev/null" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    p = d.get('pipeline', {})
    print(json.dumps(p, indent=2))
except: print('  (not available)')
" 2>/dev/null || echo "  (not available)"
}

deploy() {
    echo "==> Deploying source to $ROBOT_USER@$ROBOT_HOST:$ROBOT_APP_DIR ..."

    # Sync source
    _rsync "$PROJECT_DIR/src/reachy_mini_conversation_app/" \
           "$ROBOT_USER@$ROBOT_HOST:$ROBOT_APP_DIR/src/reachy_mini_conversation_app/"

    # Sync tests
    _rsync "$PROJECT_DIR/tests/" \
           "$ROBOT_USER@$ROBOT_HOST:$ROBOT_APP_DIR/tests/"

    # Sync .env (runtime config)
    if [[ -f "$PROJECT_DIR/.env" ]]; then
        sshpass -p "$ROBOT_PASS" scp $SSH_OPTS \
            "$PROJECT_DIR/.env" \
            "$ROBOT_USER@$ROBOT_HOST:$ROBOT_APP_DIR/.env" 2>/dev/null
    fi

    echo "    Done."
}

restart() {
    echo "==> Restarting app via daemon API..."
    result=$(_ssh "curl -s --max-time 30 -X POST http://localhost:8000/api/apps/restart-current-app 2>/dev/null" || echo '{"error": "restart request failed"}')
    echo "    $result"

    # Wait for it to come up
    echo -n "==> Waiting for app to start"
    for i in $(seq 1 20); do
        sleep 2
        state=$(_ssh "curl -sf http://localhost:8000/api/apps/current-app-status 2>/dev/null" | python3 -c "
import sys, json
try: print(json.load(sys.stdin).get('state','unknown'))
except: print('unknown')
" 2>/dev/null || echo "unknown")
        echo -n "."
        if [[ "$state" == "running" ]]; then
            echo " running!"
            return 0
        fi
    done
    echo " timeout (last state: $state)"
    return 1
}

# --- Main ---
case "${1:-deploy}" in
    --status|-s)
        status
        ;;
    --no-restart)
        deploy
        ;;
    --help|-h)
        head -8 "$0" | tail -6
        ;;
    *)
        deploy
        restart
        echo
        status
        ;;
esac
