#!/usr/bin/env bash
# Orchestrate stress clients across Alpine nodes from macOS
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLIENT_LOCAL_PATH="${SCRIPT_DIR}/alpine_stress_client.sh"
REMOTE_CLIENT_PATH="/tmp/organic_stress_client.sh"
REMOTE_PID_FILE="/tmp/organic_stress_client.pid"
REMOTE_LOG_FILE="/tmp/organic_stress_client.log"

DEFAULT_HOSTS=("alpine-1" "alpine-2" "alpine-3" "alpine-4")

SSH_BASE_OPTS=(
    -o BatchMode=yes
    -o ConnectTimeout=5
)

usage() {
    cat <<'EOF'
Usage: swarm_stress.sh <start|stop|status> [options]

Commands:
  start                Push client script, start generators on each host
  stop                 Stop remote clients and clean temporary files
  status               Show whether clients are still running

Options (apply to start/status/stop):
  --hosts "list"       Space/comma separated host list (default: alpine-1..4)
  --target URL         Base URL of stress service (default: http://localhost:7777)
  --mode MODE          full (cpu+mem+network) or compute (cpu+mem) [default: full]
  --duration SEC       How long each client runs (default: 300)
  --parallel N         Parallel curl calls per cycle per host (default: 4)
  --sleep SEC          Pause between cycles (default: 0)
  --pattern MODE       burst (default) or steady for constant concurrency
  --cpu-duration SEC   CPU load duration per request (default: 2)
  --cpu-workers N      Number of CPU processes spawned server-side (default: 4)
  --cpu-spin N         Inner spin factor per process (default: 200000)
  --memory-mb MB       Memory to request per call (default: 256)
  --memory-hold SEC    How long to hold memory (default: 1.5)
  --memory-workers N   Parallel memory allocators (default: 1)
  --network-mb MB      Network MB to stream per call (default: 25)
  --network-chunk-kb K Chunk size for network streaming (default: 256)

Examples:
  ./scripts/swarm_stress.sh start --target http://10.0.0.10:7777 --mode full
  ./scripts/swarm_stress.sh stop  --hosts "alpine-1 alpine-3"
  ./scripts/swarm_stress.sh status
EOF
}

ensure_client_exists() {
    if [ ! -x "$CLIENT_LOCAL_PATH" ]; then
        echo "Missing executable client script at $CLIENT_LOCAL_PATH" >&2
        exit 2
    fi
}

parse_hosts() {
    local host_string="$1"
    host_string="${host_string//,/ }"
    HOSTS=()
    for host in $host_string; do
        HOSTS+=("$host")
    done
}

command="${1:-}"
if [ -z "$command" ]; then
    usage
    exit 1
fi
shift

HOSTS=("${DEFAULT_HOSTS[@]}")
TARGET="http://localhost:7777"
MODE="full"
DURATION=300
PARALLEL=4
REST_INTERVAL=0
PATTERN="burst"
CPU_DURATION=2
CPU_WORKERS=4
CPU_SPIN=200000
MEMORY_MB=256
MEMORY_HOLD=1.5
MEMORY_WORKERS=1
NETWORK_MB=25
NETWORK_CHUNK_KB=256

while [ $# -gt 0 ]; do
    case "$1" in
        --hosts)
            [ $# -ge 2 ] || { echo "Missing value for --hosts" >&2; exit 1; }
            parse_hosts "$2"
            shift 2
            ;;
        --target)
            [ $# -ge 2 ] || { echo "Missing value for --target" >&2; exit 1; }
            TARGET="$2"
            shift 2
            ;;
        --mode)
            [ $# -ge 2 ] || { echo "Missing value for --mode" >&2; exit 1; }
            MODE="$2"
            shift 2
            ;;
        --duration)
            [ $# -ge 2 ] || { echo "Missing value for --duration" >&2; exit 1; }
            DURATION="$2"
            shift 2
            ;;
        --parallel)
            [ $# -ge 2 ] || { echo "Missing value for --parallel" >&2; exit 1; }
            PARALLEL="$2"
            shift 2
            ;;
        --sleep)
            [ $# -ge 2 ] || { echo "Missing value for --sleep" >&2; exit 1; }
            REST_INTERVAL="$2"
            shift 2
            ;;
        --pattern)
            [ $# -ge 2 ] || { echo "Missing value for --pattern" >&2; exit 1; }
            PATTERN="$2"
            shift 2
            ;;
        --cpu-duration)
            [ $# -ge 2 ] || { echo "Missing value for --cpu-duration" >&2; exit 1; }
            CPU_DURATION="$2"
            shift 2
            ;;
        --cpu-workers)
            [ $# -ge 2 ] || { echo "Missing value for --cpu-workers" >&2; exit 1; }
            CPU_WORKERS="$2"
            shift 2
            ;;
        --cpu-spin)
            [ $# -ge 2 ] || { echo "Missing value for --cpu-spin" >&2; exit 1; }
            CPU_SPIN="$2"
            shift 2
            ;;
        --memory-mb)
            [ $# -ge 2 ] || { echo "Missing value for --memory-mb" >&2; exit 1; }
            MEMORY_MB="$2"
            shift 2
            ;;
        --memory-hold)
            [ $# -ge 2 ] || { echo "Missing value for --memory-hold" >&2; exit 1; }
            MEMORY_HOLD="$2"
            shift 2
            ;;
        --memory-workers)
            [ $# -ge 2 ] || { echo "Missing value for --memory-workers" >&2; exit 1; }
            MEMORY_WORKERS="$2"
            shift 2
            ;;
        --network-mb)
            [ $# -ge 2 ] || { echo "Missing value for --network-mb" >&2; exit 1; }
            NETWORK_MB="$2"
            shift 2
            ;;
        --network-chunk-kb)
            [ $# -ge 2 ] || { echo "Missing value for --network-chunk-kb" >&2; exit 1; }
            NETWORK_CHUNK_KB="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [ "${#HOSTS[@]}" -eq 0 ]; then
    echo "No hosts provided" >&2
    exit 1
fi

build_query_string() {
    local base="cpu=true&memory=true&cpu_duration=${CPU_DURATION}&cpu_workers=${CPU_WORKERS}&cpu_spin=${CPU_SPIN}&memory_mb=${MEMORY_MB}&memory_hold=${MEMORY_HOLD}&memory_workers=${MEMORY_WORKERS}"
    case "$MODE" in
        full)
            echo "${base}&network=true&network_mb=${NETWORK_MB}&network_chunk_kb=${NETWORK_CHUNK_KB}"
            ;;
        compute)
            echo "${base}&network=false"
            ;;
        *)
            echo "Unsupported mode: $MODE" >&2
            exit 1
            ;;
    esac
}

push_client() {
    local host="$1"
    scp "${SSH_BASE_OPTS[@]}" -q "$CLIENT_LOCAL_PATH" "${host}:${REMOTE_CLIENT_PATH}"
    ssh "${SSH_BASE_OPTS[@]}" "$host" "chmod +x ${REMOTE_CLIENT_PATH}"
}

start_host() {
    local host="$1"
    local query="$2"
    echo "[${host}] starting stress client -> ${TARGET}/stress?${query}"
    push_client "$host"
    ssh "${SSH_BASE_OPTS[@]}" "$host" "nohup ${REMOTE_CLIENT_PATH} '${TARGET}' '${query}' '${DURATION}' '${PARALLEL}' '${REST_INTERVAL}' '${PATTERN}' > ${REMOTE_LOG_FILE} 2>&1 & echo \$! > ${REMOTE_PID_FILE}"
}

stop_host() {
    local host="$1"
    ssh "${SSH_BASE_OPTS[@]}" "$host" "if [ -f ${REMOTE_PID_FILE} ]; then pid=\$(cat ${REMOTE_PID_FILE}); if [ -n \"\$pid\" ]; then kill \$pid 2>/dev/null || true; fi; rm -f ${REMOTE_PID_FILE}; fi; rm -f ${REMOTE_CLIENT_PATH} 2>/dev/null || true"
}

status_host() {
    local host="$1"
    ssh "${SSH_BASE_OPTS[@]}" "$host" "if [ -f ${REMOTE_PID_FILE} ]; then pid=\$(cat ${REMOTE_PID_FILE}); if [ -n \"\$pid\" ] && kill -0 \$pid 2>/dev/null; then echo '[${host}] running pid='\$pid; else echo '[${host}] pid file present but process not running'; fi; else echo '[${host}] idle'; fi"
}

case "$command" in
    start)
        ensure_client_exists
        query_string="$(build_query_string)"
        for host in "${HOSTS[@]}"; do
            start_host "$host" "$query_string"
        done
        ;;
    stop)
        for host in "${HOSTS[@]}"; do
            echo "[${host}] stopping stress client"
            stop_host "$host"
        done
        ;;
    status)
        for host in "${HOSTS[@]}"; do
            status_host "$host"
        done
        ;;
    *)
        echo "Unknown command: $command" >&2
        usage
        exit 1
        ;;
esac
