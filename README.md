# Organic Web Stress Harness

This repo provides a FastAPI app (`organic_web_stress.py`) plus helper scripts to hammer a Swarm service from multiple Raspberry Pi/Alpine nodes.

## Service Endpoints

Run the server locally (`uvicorn organic_web_stress:app --host 0.0.0.0 --port 7777`) or via the provided Dockerfile. The flexible `/stress` endpoint now triggers CPU and memory workloads **concurrently** and can optionally stream network payloads.

Examples:

```bash
# CPU + memory + network in one request
curl "http://SERVER:7777/stress?cpu=true&memory=true&network=true&cpu_duration=2&cpu_workers=4&memory_mb=256&memory_hold=1.5&network_mb=25"

# CPU + memory only (no network)
curl "http://SERVER:7777/stress?cpu=true&memory=true&network=false&cpu_duration=3&cpu_workers=6&memory_mb=512&memory_hold=2"
```

Responses contain the per-resource stats so you can correlate with Grafana dashboards.

## Swarm Stress From macOS

Use `scripts/swarm_stress.sh` to start/stop coordinated stress loops on your remote Alpine nodes (`alpine-1`, `alpine-2`, `alpine-3`, `alpine-4` by default).

### Prerequisites

- Passwordless SSH from macOS to each node (e.g., `ssh alpine-1` works and resolves to the Raspberry Pi).
- `curl` installed on the Alpine nodes (`apk add curl`).
- `ssh`/`scp` available on macOS (standard).

### Start a test

```bash
./scripts/swarm_stress.sh start \
  --target http://your-service:7777 \
  --mode full \
  --duration 600 \
  --parallel 6 \
  --hosts "alpine-1 alpine-2 alpine-3 alpine-4"
```

`--mode full` hits CPU + memory + network, while `--mode compute` emits only CPU + memory requests. The script automatically copies `scripts/alpine_stress_client.sh` to `/tmp/organic_stress_client.sh` on each node and starts it in the background (log file `/tmp/organic_stress_client.log`).

### Stop or inspect

```bash
./scripts/swarm_stress.sh status --hosts "alpine-1 alpine-3"
./scripts/swarm_stress.sh stop   --hosts "alpine-1 alpine-3"
```

Status checks whether the remote PID stored at `/tmp/organic_stress_client.pid` is still active. `stop` cleans up the background process and temporary script.

### Advanced knobs

- `--cpu-duration`, `--cpu-workers`, `--memory-mb`, `--memory-hold`, `--network-mb` tune the payload of each `/stress` call.
- `--parallel` controls how many concurrent requests each node fires per cycle.
- `--sleep` adds a pause (seconds) between request bursts if you need a saw-tooth profile instead of constant pressure.

If you want to run the client script manually on one node you can `scp scripts/alpine_stress_client.sh alpine-1:/tmp` and launch it directly:

```bash
ssh alpine-1 "/tmp/alpine_stress_client.sh http://your-service:7777 'cpu=true&memory=true&network=true&cpu_duration=2&cpu_workers=4&memory_mb=256&memory_hold=1.5&network_mb=10' 900 4 0"
```

This setup lets you reproduce failover + redeploy timings (one node vs multiple nodes) while collecting metrics from Grafana/Prometheus.
