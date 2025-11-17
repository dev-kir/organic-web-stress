#!/bin/sh
# Lightweight stress client executed on Alpine nodes
set -eu

TARGET_BASE="${1:-}"
QUERY_STRING="${2:-}"
DURATION_SECONDS="${3:-300}"
PARALLEL_REQUESTS="${4:-2}"
REST_INTERVAL="${5:-0}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-120}"

if [ -z "$TARGET_BASE" ] || [ -z "$QUERY_STRING" ]; then
    echo "Usage: alpine_stress_client.sh <target_base_url> <query_string> [duration] [parallel_requests] [sleep_between_loops]" >&2
    exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required on the Alpine node" >&2
    exit 2
fi

end_time=$(( $(date +%s) + ${DURATION_SECONDS%.*} ))
cycle=0

while [ "$(date +%s)" -lt "$end_time" ]; do
    cycle=$((cycle + 1))
    echo "$(date -Iseconds) [organic-stress] cycle=$cycle target=${TARGET_BASE}/stress?${QUERY_STRING} parallel=$PARALLEL_REQUESTS"

    i=0
    while [ "$i" -lt "$PARALLEL_REQUESTS" ]; do
        (
            curl -sf --connect-timeout 5 --max-time "$REQUEST_TIMEOUT" \
                "${TARGET_BASE}/stress?${QUERY_STRING}" >/dev/null || \
                echo "$(date -Iseconds) [organic-stress] request failed"
        ) &
        i=$((i + 1))
    done
    wait

    if [ "$REST_INTERVAL" -gt 0 ]; then
        sleep "$REST_INTERVAL"
    fi
done

echo "$(date -Iseconds) [organic-stress] completed duration=${DURATION_SECONDS}s cycles=$cycle"
