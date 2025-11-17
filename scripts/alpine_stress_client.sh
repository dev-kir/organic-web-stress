#!/bin/sh
# Lightweight stress client executed on Alpine nodes
set -eu

TARGET_BASE="${1:-}"
QUERY_STRING="${2:-}"
DURATION_SECONDS="${3:-300}"
PARALLEL_REQUESTS="${4:-2}"
REST_INTERVAL="${5:-0}"
PATTERN="$(printf "%s" "${6:-burst}" | tr '[:upper:]' '[:lower:]')"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-120}"

if [ -z "$TARGET_BASE" ] || [ -z "$QUERY_STRING" ]; then
    echo "Usage: alpine_stress_client.sh <target_base_url> <query_string> [duration] [parallel_requests] [sleep_between_loops] [pattern]" >&2
    exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required on the Alpine node" >&2
    exit 2
fi

case "$PATTERN" in
    burst|steady)
        ;;
    *)
        echo "Unsupported pattern '$PATTERN' (use 'burst' or 'steady')" >&2
        exit 3
        ;;
esac

end_time=$(( $(date +%s) + ${DURATION_SECONDS%.*} ))
cycle=0

send_request() {
    if ! curl -sf --connect-timeout 5 --max-time "$REQUEST_TIMEOUT" \
        "${TARGET_BASE}/stress?${QUERY_STRING}" >/dev/null; then
        echo "$(date -Iseconds) [organic-stress] request failed"
    fi
}

if [ "$PATTERN" = "burst" ]; then
    while [ "$(date +%s)" -lt "$end_time" ]; do
        cycle=$((cycle + 1))
        echo "$(date -Iseconds) [organic-stress] pattern=burst cycle=$cycle target=${TARGET_BASE}/stress?${QUERY_STRING} parallel=$PARALLEL_REQUESTS"

        i=0
        while [ "$i" -lt "$PARALLEL_REQUESTS" ]; do
            send_request &
            i=$((i + 1))
        done
        wait

        if [ "$REST_INTERVAL" -gt 0 ]; then
            sleep "$REST_INTERVAL"
        fi
    done
else
    echo "$(date -Iseconds) [organic-stress] pattern=steady workers=$PARALLEL_REQUESTS target=${TARGET_BASE}/stress?${QUERY_STRING}"
    worker() {
        while [ "$(date +%s)" -lt "$end_time" ]; do
            send_request
            if [ "$REST_INTERVAL" -gt 0 ]; then
                sleep "$REST_INTERVAL"
            fi
        done
    }

    i=0
    while [ "$i" -lt "$PARALLEL_REQUESTS" ]; do
        worker &
        i=$((i + 1))
    done
    wait
fi

final_cycles="$cycle"
if [ "$PATTERN" = "steady" ]; then
    final_cycles="steady"
fi

echo "$(date -Iseconds) [organic-stress] completed duration=${DURATION_SECONDS}s pattern=${PATTERN} cycles=${final_cycles}"
