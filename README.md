# Organic Web Stress Harness

This repo provides a FastAPI app (`organic_web_stress.py`) plus helper scripts to hammer a Swarm service from multiple Raspberry Pi/Alpine nodes.

## Service Endpoints

Run the server locally (`uvicorn organic_web_stress:app --host 0.0.0.0 --port 7777`) or via the provided Dockerfile. The flexible `/stress` endpoint now triggers CPU and memory workloads **concurrently** and can optionally stream network payloads. CPU load is handled by a multi-process pool so a single request can saturate all cores, and memory pressure can be fanned out across several workers.

Examples:

```bash
# CPU + memory + network in one request
curl "http://SERVER:7777/stress?cpu=true&memory=true&network=true&cpu_duration=3&cpu_workers=6&cpu_spin=300000&memory_mb=512&memory_hold=2&memory_workers=2&network_mb=50&network_chunk_kb=512"

# CPU + memory only (no network)
curl "http://SERVER:7777/stress?cpu=true&memory=true&network=false&cpu_duration=4&cpu_workers=4&cpu_spin=400000&memory_mb=1024&memory_hold=3&memory_workers=4"
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

- `--cpu-duration`, `--cpu-workers`, `--cpu-spin` cooperate to determine how long and how hard each request burns CPU (each worker is a dedicated server-side process).
- `--memory-mb`, `--memory-hold`, `--memory-workers` configure how much RAM each request grabs and how many parallel allocators run.
- `--network-mb`, `--network-chunk-kb` control the total bytes streamed back per request and the chunk size (larger chunks push network interfaces harder).
- `--parallel` controls how many concurrent requests each node fires per cycle.
- `--sleep` adds a pause (seconds) between request bursts if you need a saw-tooth profile instead of constant pressure.

### Heavy-load recipe (50 seconds)

If you want all Alpine nodes to smash CPU, memory, and network simultaneously for ~50 s (so you can watch Swarm reschedule on failover), run this from the repo root:

```bash
./scripts/swarm_stress.sh stop || true
./scripts/swarm_stress.sh start \
  --target http://192.168.2.50:7777 \
  --mode full \
  --duration 50 \
  --parallel 30 \
  --cpu-duration 5 \
  --cpu-workers 8 \
  --cpu-spin 400000 \
  --memory-mb 1024 \
  --memory-hold 3 \
  --memory-workers 3 \
  --network-mb 200 \
  --network-chunk-kb 512
```

Each Raspberry Pi fires 30 concurrent `/stress` calls. Every call forces the container to spawn 8 CPU processes spinning 400k operations per loop for 5 s, allocates ~1 GB of RAM across three workers for 3 s, and streams ~200 MB back in large chunks—enough to peg CPU/memory/network and keep requests in flight during container restarts. Adjust the knobs upward/downward depending on how aggressive you want the test to be.

If you want to run the client script manually on one node you can `scp scripts/alpine_stress_client.sh alpine-1:/tmp` and launch it directly:

```bash
ssh alpine-1 "/tmp/alpine_stress_client.sh http://your-service:7777 'cpu=true&memory=true&network=true&cpu_duration=2&cpu_workers=4&memory_mb=256&memory_hold=1.5&network_mb=10' 900 4 0"
```

This setup lets you reproduce failover + redeploy timings (one node vs multiple nodes) while collecting metrics from Grafana/Prometheus.
