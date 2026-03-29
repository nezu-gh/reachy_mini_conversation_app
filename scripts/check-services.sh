#!/usr/bin/env bash
# Check if all inference services (STT, LLM, TTS) on the VM are healthy.
#
# Usage:
#   ./scripts/check-services.sh          # check all services
#   ./scripts/check-services.sh --watch  # poll every 10s until all healthy
#   ./scripts/check-services.sh --robot  # also check robot daemon + app

source "$(dirname "$0")/_common.sh"

# Service endpoints (read from .env or defaults)
if [[ -f "$PROJECT_DIR/.env.r3mn1" ]]; then
    ASR_BASE_URL="${ASR_BASE_URL:-$(grep -oP '(?<=ASR_BASE_URL=).*' "$PROJECT_DIR/.env.r3mn1" 2>/dev/null || echo "http://$LOCAL_VM_IP:8015/v1")}"
    LLM_BASE_URL="${LLM_BASE_URL:-$(grep -oP '(?<=LLM_BASE_URL=).*' "$PROJECT_DIR/.env.r3mn1" 2>/dev/null || echo "http://$LOCAL_VM_IP:3443/v1")}"
    TTS_BASE_URL="${TTS_BASE_URL:-$(grep -oP '(?<=TTS_BASE_URL=).*' "$PROJECT_DIR/.env.r3mn1" 2>/dev/null || echo "http://$LOCAL_VM_IP:7034/v1")}"
fi
ASR_BASE_URL="${ASR_BASE_URL:-http://$LOCAL_VM_IP:8015/v1}"
LLM_BASE_URL="${LLM_BASE_URL:-http://$LOCAL_VM_IP:3443/v1}"
TTS_BASE_URL="${TTS_BASE_URL:-http://$LOCAL_VM_IP:7034/v1}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

check_service() {
    local name="$1" url="$2"
    if curl -sf --max-time 3 "$url/models" >/dev/null 2>&1; then
        echo -e "  ${GREEN}OK${NC}  $name  ($url)"
        return 0
    else
        echo -e "  ${RED}FAIL${NC}  $name  ($url)"
        return 1
    fi
}

check_robot() {
    echo
    echo "Robot ($ROBOT_HOST):"

    # Daemon
    if curl -sf --max-time 3 "http://$ROBOT_HOST:8000/api/apps/current-app-status" >/dev/null 2>&1; then
        echo -e "  ${GREEN}OK${NC}  Daemon  (port 8000)"
    else
        echo -e "  ${RED}FAIL${NC}  Daemon  (port 8000)"
    fi

    # App (settings/dashboard on 7860)
    health=$(curl -sf --max-time 3 "http://$ROBOT_HOST:7860/api/health" 2>/dev/null)
    if [[ -n "$health" ]]; then
        pipeline_alive=$(echo "$health" | python3 -c "import sys,json; print(json.load(sys.stdin).get('pipeline',{}).get('pipeline_alive', False))" 2>/dev/null)
        drops=$(echo "$health" | python3 -c "import sys,json; print(json.load(sys.stdin).get('pipeline',{}).get('frames_dropped', '?'))" 2>/dev/null)
        if [[ "$pipeline_alive" == "True" ]]; then
            echo -e "  ${GREEN}OK${NC}  App pipeline  (drops: $drops)"
        else
            echo -e "  ${YELLOW}WAIT${NC}  App running but pipeline not alive yet"
        fi
    else
        echo -e "  ${RED}FAIL${NC}  App  (port 7860)"
    fi
}

run_checks() {
    echo "Inference services ($LOCAL_VM_IP):"
    local failures=0
    check_service "ASR (Qwen-ASR)" "$ASR_BASE_URL" || ((failures++))
    check_service "LLM (Qwen3.5)"  "$LLM_BASE_URL" || ((failures++))
    check_service "TTS (Qwen3-TTS)" "$TTS_BASE_URL" || ((failures++))
    return $failures
}

# --- Main ---
case "${1:-}" in
    --watch|-w)
        echo "Polling services every 10s (Ctrl+C to stop)..."
        while true; do
            echo
            echo "--- $(date '+%H:%M:%S') ---"
            if run_checks; then
                echo -e "\n${GREEN}All services healthy.${NC}"
                exit 0
            fi
            sleep 10
        done
        ;;
    --robot|-r)
        run_checks
        check_robot
        ;;
    --help|-h)
        head -6 "$0" | tail -4
        ;;
    *)
        run_checks
        ;;
esac
