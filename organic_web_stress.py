#!/usr/bin/env python3
"""
Organic Web Stress - Complete Edition
Realistic traffic simulation + flexible stress testing
For Docker Swarm Load Balancing
"""
import asyncio
import json
import math
import random
import time
import uuid
import socket
import os
from datetime import datetime
from typing import Any, Dict, Optional
from concurrent.futures import ProcessPoolExecutor
from threading import Lock

from fastapi import FastAPI, Query, Response
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

app = FastAPI(title="Organic Web Stress", version="2.0")

SERVER_ID = socket.gethostname()
request_counter = {"total": 0, "by_endpoint": {}}
MAX_CPU_PROCESSES = max(4, (os.cpu_count() or 2) * 2)
_cpu_executor: Optional[ProcessPoolExecutor] = None
_cpu_executor_lock = Lock()

# ==============================
# HELPER FUNCTIONS
# ==============================

def simulate_database_query(complexity: str = "simple") -> float:
    """Simulate database query delay"""
    delays = {
        "simple": (0.01, 0.05),
        "medium": (0.05, 0.15),
        "complex": (0.15, 0.40),
    }
    min_delay, max_delay = delays.get(complexity, (0.01, 0.05))
    delay = random.uniform(min_delay, max_delay)
    time.sleep(delay)
    return delay

def simulate_cpu_work(intensity: str = "light") -> dict:
    """Simulate CPU-intensive work"""
    intensities = {
        "light": 100_000,
        "medium": 500_000,
        "heavy": 2_000_000,
    }
    iterations = intensities.get(intensity, 100_000)
    
    start = time.time()
    result = 0
    for i in range(iterations):
        result += math.sqrt(random.random() * 999)
    elapsed = time.time() - start
    
    return {"iterations": iterations, "elapsed_ms": round(elapsed * 1000, 2)}

def add_tracking_headers(response: Response, endpoint: str, start_time: float):
    """Add tracking headers"""
    elapsed = time.time() - start_time
    response.headers["X-Server-ID"] = SERVER_ID
    response.headers["X-Response-Time-Ms"] = str(round(elapsed * 1000, 2))
    response.headers["X-Request-ID"] = str(uuid.uuid4())
    response.headers["X-Endpoint"] = endpoint
    
    request_counter["total"] += 1
    request_counter["by_endpoint"][endpoint] = request_counter["by_endpoint"].get(endpoint, 0) + 1


def get_cpu_executor() -> ProcessPoolExecutor:
    """Lazily create/reuse the process pool for CPU stress"""
    global _cpu_executor
    if _cpu_executor is None:
        with _cpu_executor_lock:
            if _cpu_executor is None:
                _cpu_executor = ProcessPoolExecutor(max_workers=MAX_CPU_PROCESSES)
    return _cpu_executor


def _cpu_worker_process(duration: float, spin_factor: int, worker_id: int) -> Dict[str, Any]:
    """Run CPU operations in a dedicated process to saturate cores"""
    cpu_start = time.time()
    deadline = cpu_start + max(0.1, duration)
    spin = max(10_000, spin_factor)
    iterations = 0

    while time.time() < deadline:
        for _ in range(spin):
            math.sqrt(random.random() * 999_999)
            iterations += 1

    return {
        "worker_id": worker_id,
        "elapsed": round(time.time() - cpu_start, 3),
        "iterations": iterations,
    }


async def run_cpu_stress(cpu_duration: float, cpu_workers: int, cpu_spin: int) -> Dict[str, Any]:
    """Run CPU stress using multiple processes"""
    worker_count = min(max(1, cpu_workers), MAX_CPU_PROCESSES)
    loop = asyncio.get_running_loop()
    executor = get_cpu_executor()
    tasks = [
        loop.run_in_executor(
            executor,
            _cpu_worker_process,
            cpu_duration,
            cpu_spin,
            worker_id,
        )
        for worker_id in range(worker_count)
    ]

    worker_stats = await asyncio.gather(*tasks)
    total_iterations = sum(w["iterations"] for w in worker_stats)
    longest = max((w["elapsed"] for w in worker_stats), default=0.0)

    return {
        "workers": worker_count,
        "spin_factor": cpu_spin,
        "iterations": total_iterations,
        "elapsed": longest,
        "per_worker": worker_stats,
    }


def _memory_worker(mem_mb: int, hold: float, worker_id: int) -> Dict[str, Any]:
    mem_start = time.time()
    chunks = []
    allocated = 0
    target_bytes = max(1, mem_mb) * 1024 * 1024
    chunk_size = 64 * 1024 * 1024

    try:
        while allocated < target_bytes:
            block = bytearray(min(chunk_size, target_bytes - allocated))
            block[0] = 1
            block[-1] = 1
            chunks.append(block)
            allocated += len(block)

        if hold > 0:
            time.sleep(hold)
    except MemoryError:
        allocated = sum(len(c) for c in chunks)
    finally:
        chunks.clear()

    return {
        "worker_id": worker_id,
        "requested_mb": mem_mb,
        "allocated_mb": round(allocated / (1024 * 1024), 2),
        "hold_seconds": hold,
        "elapsed": round(time.time() - mem_start, 3),
    }


async def run_memory_stress(memory_mb: int, memory_hold: float, memory_workers: int) -> Dict[str, Any]:
    """Allocate/hold memory asynchronously across multiple workers"""
    total_request = max(1, memory_mb)
    worker_count = min(max(1, memory_workers), total_request)
    per_worker = total_request // worker_count
    extra = total_request % worker_count

    loop = asyncio.get_running_loop()
    tasks = []
    for worker_id in range(worker_count):
        quota = per_worker + (1 if worker_id < extra else 0)
        tasks.append(asyncio.to_thread(_memory_worker, quota, memory_hold, worker_id))

    worker_stats = await asyncio.gather(*tasks)
    total_allocated = sum(w["allocated_mb"] for w in worker_stats)
    longest = max((w["elapsed"] for w in worker_stats), default=0.0)

    return {
        "workers": worker_count,
        "requested_mb": memory_mb,
        "allocated_mb": round(total_allocated, 2),
        "hold_seconds": memory_hold,
        "elapsed": longest,
        "per_worker": worker_stats,
    }


async def gather_task_results(tasks: Dict[str, asyncio.Task]) -> Dict[str, Any]:
    """Await all async tasks and bucket the results"""
    results: Dict[str, Any] = {}
    for name, task in tasks.items():
        results[name] = await task
    return results

# ==============================
# BASIC ENDPOINTS
# ==============================

@app.get("/")
async def homepage(response: Response):
    """Homepage - Light load"""
    start = time.time()
    db_time = simulate_database_query("simple")
    cpu_stats = simulate_cpu_work("light")
    
    data = {
        "page": "homepage",
        "message": "Welcome to Organic Web Stress v2",
        "server_id": SERVER_ID,
        "processing": {
            "db_ms": round(db_time * 1000, 2),
            "cpu_ms": cpu_stats["elapsed_ms"],
        }
    }
    
    add_tracking_headers(response, "homepage", start)
    return JSONResponse(data)

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "server_id": SERVER_ID}

@app.get("/api/data")
async def api_data(response: Response):
    """API endpoint - Medium load"""
    start = time.time()
    db_time = simulate_database_query("medium")
    cpu_stats = simulate_cpu_work("medium")
    
    items = [
        {"id": i, "value": random.randint(100, 999)}
        for i in range(50)
    ]
    
    data = {
        "endpoint": "api_data",
        "items": items,
        "count": len(items),
        "processing": {
            "db_ms": round(db_time * 1000, 2),
            "cpu_ms": cpu_stats["elapsed_ms"],
        }
    }
    
    add_tracking_headers(response, "api_data", start)
    return JSONResponse(data)

@app.get("/dashboard")
async def dashboard(response: Response):
    """Dashboard - Heavy load"""
    start = time.time()
    
    db_time1 = simulate_database_query("complex")
    db_time2 = simulate_database_query("medium")
    cpu_stats = simulate_cpu_work("heavy")
    
    # Allocate some memory
    tmp = bytearray(20 * 1024 * 1024)  # 20MB
    tmp[0] = 1
    tmp[-1] = 1
    del tmp
    
    data = {
        "page": "dashboard",
        "metrics": {
            "users": random.randint(100, 500),
            "requests": random.randint(50, 200),
        },
        "processing": {
            "db_ms": round((db_time1 + db_time2) * 1000, 2),
            "cpu_ms": cpu_stats["elapsed_ms"],
        }
    }
    
    add_tracking_headers(response, "dashboard", start)
    return JSONResponse(data)

@app.get("/search")
async def search(q: str = "default", response: Response = None):
    """Search endpoint"""
    start = time.time()
    complexity = "simple" if len(q) < 5 else "medium" if len(q) < 15 else "complex"
    
    db_time = simulate_database_query(complexity)
    cpu_stats = simulate_cpu_work("medium")
    
    results = [
        {"id": i, "title": f"Result {i} for '{q}'"}
        for i in range(random.randint(5, 20))
    ]
    
    data = {
        "query": q,
        "results": results,
        "count": len(results),
        "processing": {
            "db_ms": round(db_time * 1000, 2),
            "cpu_ms": cpu_stats["elapsed_ms"],
        }
    }
    
    add_tracking_headers(response, "search", start)
    return JSONResponse(data)

@app.get("/product/{product_id}")
async def product(product_id: str, response: Response):
    """Product page"""
    start = time.time()
    db_time = simulate_database_query("medium")
    cpu_stats = simulate_cpu_work("medium")
    
    data = {
        "product": {
            "id": product_id,
            "name": f"Product {product_id}",
            "price": round(random.uniform(10.0, 1000.0), 2),
        },
        "processing": {
            "db_ms": round(db_time * 1000, 2),
            "cpu_ms": cpu_stats["elapsed_ms"],
        }
    }
    
    add_tracking_headers(response, "product", start)
    return JSONResponse(data)

# ==============================
# FLEXIBLE STRESS ENDPOINT
# ==============================

@app.get("/stress")
async def stress(
    response: Response,
    cpu: bool = Query(False),
    memory: bool = Query(False),
    network: bool = Query(False),
    cpu_duration: float = Query(1.0),
    cpu_workers: int = Query(1),
    cpu_spin: int = Query(200_000),
    memory_mb: int = Query(128),
    memory_hold: float = Query(1.0),
    memory_workers: int = Query(1),
    network_mb: int = Query(5),
    network_chunk_kb: int = Query(256),
    fail_after: float = Query(0.0),  # NEW: Simulate container failure after N seconds
    slow_response: float = Query(0.0),  # NEW: Simulate slow response times
):
    """Flexible stress endpoint - mix CPU, memory, network, with realistic failure modes"""
    start = time.time()

    # Simulate slow response time (e.g., database lag, network latency)
    if slow_response > 0:
        await asyncio.sleep(slow_response)

    stats = {
        "requested": {"cpu": cpu, "memory": memory, "network": network},
        "server_id": SERVER_ID,
        "slow_response_delay": slow_response,
    }

    # Simulate container failure (crash after N seconds of stress)
    if fail_after > 0:
        async def delayed_crash():
            await asyncio.sleep(fail_after)
            # Simulate container crash by raising SystemExit
            import os
            os._exit(1)

        asyncio.create_task(delayed_crash())

    # CPU and memory load run concurrently using background tasks
    pending_tasks: Dict[str, asyncio.Task] = {}
    if cpu:
        pending_tasks["cpu"] = asyncio.create_task(run_cpu_stress(cpu_duration, cpu_workers, cpu_spin))

    if memory:
        pending_tasks["memory"] = asyncio.create_task(run_memory_stress(memory_mb, memory_hold, memory_workers))

    # Network transfer
    if network:
        total_bytes = network_mb * 1024 * 1024
        chunk_size = max(8, network_chunk_kb) * 1024
        
        async def stream():
            sent = 0
            while sent < total_bytes:
                chunk = b"X" * min(chunk_size, total_bytes - sent)
                yield chunk
                sent += len(chunk)
                await asyncio.sleep(0)
            
            # Send final stats
            stats["network"] = {
                "requested_mb": network_mb,
                "sent_bytes": sent,
                "sent_mb": round(sent / (1024 * 1024), 2),
                "chunk_kb": round(chunk_size / 1024, 2),
            }
            if pending_tasks:
                stats.update(await gather_task_results(pending_tasks))
            summary = json.dumps({"stats": stats})
            yield b"\n" + summary.encode()
        
        add_tracking_headers(response, "stress", start)
        return StreamingResponse(stream(), media_type="application/octet-stream")
    
    # No network - return JSON
    if pending_tasks:
        stats.update(await gather_task_results(pending_tasks))
    add_tracking_headers(response, "stress", start)
    return JSONResponse(stats)

# ==============================
# MONITORING ENDPOINTS
# ==============================

@app.get("/metrics")
async def metrics():
    """Metrics endpoint"""
    return {
        "server_id": SERVER_ID,
        "total_requests": request_counter["total"],
        "requests_by_endpoint": request_counter["by_endpoint"],
        "timestamp": datetime.now().isoformat(),
    }

@app.get("/request-stats")
async def request_stats():
    """Request distribution statistics"""
    return {
        "server_id": SERVER_ID,
        "hostname": socket.gethostname(),
        "total_requests": request_counter["total"],
        "by_endpoint": request_counter["by_endpoint"],
        "timestamp": datetime.now().isoformat(),
    }

@app.get("/degrade")
async def gradual_degradation(
    response: Response,
    duration: int = Query(60, description="Duration of degradation in seconds"),
    cpu_ramp: bool = Query(True, description="Gradually increase CPU usage"),
    memory_leak: bool = Query(True, description="Simulate memory leak"),
    slow_down: bool = Query(True, description="Gradually slow down responses"),
    network_ramp: bool = Query(False, description="Gradually increase network traffic"),
    network_mb_target: int = Query(100, description="Target network MB to send during ramp"),
):
    """
    Simulate gradual service degradation (realistic failure scenario)

    This endpoint simulates a container that slowly becomes unhealthy over time:
    - CPU usage ramps up gradually (like a runaway process)
    - Memory usage increases (like a memory leak)
    - Response times slow down (like database connection pool exhaustion)
    - Network traffic increases (like sending large responses)

    This is more realistic than instant crashes and gives proactive recovery
    time to detect and respond before complete failure.
    """

    # If network ramp is enabled, use streaming response to send data during degradation
    if network_ramp:
        async def degradation_stream():
            start = time.time()
            network_bytes_sent = 0
            memory_chunks = []

            # Ramp up gradually over the duration
            steps = min(duration, 20)  # Max 20 steps
            step_duration = duration / steps

            for step in range(steps):
                step_start = time.time()
                elapsed = step_start - start
                progress = (step + 1) / steps  # 0.0 to 1.0

                # CPU ramp: Start light, get heavier (ULTRA AGGRESSIVE!)
                if cpu_ramp:
                    base_spin = 1_000_000
                    max_spin = 10_000_000
                    spin = int(base_spin + (progress ** 2.5) * (max_spin - base_spin))
                    result = 0
                    for _ in range(spin):
                        result += math.sqrt(random.random() * 999)

                # Memory leak: Allocate MASSIVE amounts to trigger OOM!
                if memory_leak:
                    leak_mb = int(50 + (progress ** 1.5) * 450)
                    try:
                        chunk = bytearray(leak_mb * 1024 * 1024)
                        chunk[0] = 1
                        chunk[-1] = 1
                        chunk[len(chunk) // 2] = 1
                        memory_chunks.append(chunk)
                    except MemoryError:
                        pass

                # Network ramp: Send data NOW during this step (not after!)
                # Send progressively larger chunks (5MB to target MB)
                chunk_mb = int(5 + (progress ** 2) * (network_mb_target - 5))
                chunk_size = chunk_mb * 1024 * 1024
                network_bytes_sent += chunk_size

                # CRITICAL: Stream network data DURING the ramp step
                # Send in smaller chunks spread evenly across the step duration
                sent_this_step = 0
                mini_chunk_size = 64 * 1024  # 64KB mini-chunks (smaller for smoother streaming)

                # Calculate delay to spread chunk_size across step_duration
                num_chunks = max(1, chunk_size // mini_chunk_size)
                delay_per_chunk = step_duration / num_chunks if num_chunks > 1 else 0.01

                while sent_this_step < chunk_size:
                    mini_chunk = b"X" * min(mini_chunk_size, chunk_size - sent_this_step)
                    yield mini_chunk
                    sent_this_step += len(mini_chunk)
                    await asyncio.sleep(delay_per_chunk)  # Evenly distributed

                # Slow down: Add artificial delay
                if slow_down:
                    delay = 0.1 + (progress * 1.9)
                    await asyncio.sleep(delay)

                # Wait for next step (if not last)
                if step < steps - 1:
                    step_elapsed = time.time() - step_start
                    remaining = step_duration - step_elapsed
                    if remaining > 0:
                        await asyncio.sleep(remaining)

            # SUSTAIN peak load
            sustain_duration = max(60, duration)
            sustain_steps = int(sustain_duration / 3)
            max_spin = 10_000_000

            for sustain_step in range(sustain_steps):
                # Keep CPU at MAXIMUM
                if cpu_ramp:
                    result = 0
                    for _ in range(max_spin):
                        result += math.sqrt(random.random() * 999)

                # Keep adding memory during sustain
                if memory_leak and sustain_step % 3 == 0:
                    try:
                        extra_chunk = bytearray(200 * 1024 * 1024)
                        extra_chunk[0] = 1
                        extra_chunk[-1] = 1
                        memory_chunks.append(extra_chunk)
                    except MemoryError:
                        pass

                # CRITICAL: Continue streaming network during sustain!
                # Send 50MB spread evenly across 3 seconds (~17 MB/s = ~133 Mbps sustained rate)
                if network_ramp:
                    sustain_chunk_size = 50 * 1024 * 1024  # 50MB per 3s step
                    network_bytes_sent += sustain_chunk_size

                    # Stream in small chunks with timing to achieve ~17 MB/s sustained
                    sent_this_sustain_step = 0
                    mini_chunk_size = 64 * 1024  # 64KB mini-chunks (smaller for smoother streaming)

                    # Calculate delay to spread 50MB across 3 seconds
                    # 50MB / 3s = ~17 MB/s = ~133 Mbps (well above 40 Mbps threshold)
                    # 64KB chunks = 50MB / 64KB = ~800 chunks
                    # 3 seconds / 800 chunks = ~0.00375s per chunk
                    num_chunks = sustain_chunk_size // mini_chunk_size
                    delay_per_chunk = 3.0 / num_chunks  # Spread evenly across 3 seconds

                    while sent_this_sustain_step < sustain_chunk_size:
                        mini_chunk = b"X" * min(mini_chunk_size, sustain_chunk_size - sent_this_sustain_step)
                        yield mini_chunk
                        sent_this_sustain_step += len(mini_chunk)
                        await asyncio.sleep(delay_per_chunk)  # Evenly distributed timing
                else:
                    # If no network ramp, just sleep for the CPU/MEM stress
                    await asyncio.sleep(3)

            # Send final stats as JSON
            stats = {
                "server_id": SERVER_ID,
                "total_elapsed_s": round(time.time() - start, 2),
                "network_sent_mb": round(network_bytes_sent / (1024 * 1024), 2),
                "memory_leaked_mb": sum(len(c) for c in memory_chunks) // (1024 * 1024),
                "completed": True
            }
            yield f"\n{json.dumps(stats)}\n".encode()

            # Cleanup
            memory_chunks.clear()

        from starlette.responses import StreamingResponse
        return StreamingResponse(degradation_stream(), media_type="application/octet-stream")

    # Original non-streaming path (when network_ramp=false)
    start = time.time()
    stats = {
        "server_id": SERVER_ID,
        "degradation_mode": {
            "cpu_ramp": cpu_ramp,
            "memory_leak": memory_leak,
            "slow_down": slow_down,
            "network_ramp": network_ramp,
        },
        "duration": duration,
        "timeline": []
    }

    # Track network data sent
    network_bytes_sent = 0

    # Memory leak simulation - keep allocating chunks
    memory_chunks = []

    # Ramp up gradually over the duration
    steps = min(duration, 20)  # Max 20 steps
    step_duration = duration / steps

    for step in range(steps):
        step_start = time.time()
        elapsed = step_start - start
        progress = (step + 1) / steps  # 0.0 to 1.0

        # CPU ramp: Start light, get heavier (ULTRA AGGRESSIVE!)
        if cpu_ramp:
            # Spin factor increases EXPONENTIALLY from 1M to 10M (even more aggressive!)
            # This creates VERY high CPU load
            base_spin = 1_000_000
            max_spin = 10_000_000
            spin = int(base_spin + (progress ** 2.5) * (max_spin - base_spin))  # Power 2.5 for faster growth!
            result = 0
            for _ in range(spin):
                result += math.sqrt(random.random() * 999)

        # Memory leak: Allocate MASSIVE amounts to trigger OOM!
        if memory_leak:
            # Allocate 50MB to 500MB per step (10x more aggressive!)
            # This will consume 5-10GB over 20 steps to trigger OOM
            leak_mb = int(50 + (progress ** 1.5) * 450)
            try:
                chunk = bytearray(leak_mb * 1024 * 1024)
                # Touch the memory to force actual allocation
                chunk[0] = 1
                chunk[-1] = 1
                chunk[len(chunk) // 2] = 1  # Touch middle too
                memory_chunks.append(chunk)
            except MemoryError:
                # If we hit memory limit, that's actually what we want!
                # Just stop allocating more
                pass

        # Slow down: Add artificial delay
        if slow_down:
            # Delay increases from 0.1s to 2s
            delay = 0.1 + (progress * 1.9)
            await asyncio.sleep(delay)

        # Record this step
        stats["timeline"].append({
            "step": step + 1,
            "elapsed_s": round(elapsed, 1),
            "progress": f"{int(progress * 100)}%",
            "cpu_spin": spin if cpu_ramp else 0,
            "memory_leaked_mb": sum(len(c) for c in memory_chunks) // (1024 * 1024) if memory_leak else 0,
        })

        # Wait for next step (if not last)
        if step < steps - 1:
            step_elapsed = time.time() - step_start
            remaining = step_duration - step_elapsed
            if remaining > 0:
                await asyncio.sleep(remaining)

    # Final stats after ramp-up
    total_elapsed = time.time() - start
    stats["ramp_up_duration_s"] = round(total_elapsed, 2)
    stats["final_memory_leaked_mb"] = sum(len(c) for c in memory_chunks) // (1024 * 1024) if memory_leak else 0

    # CRITICAL: SUSTAIN the peak load for 60s MINIMUM to guarantee OOM and health check failures!
    # This ensures container stays at 100% CPU and peak memory long enough to fail
    sustain_duration = max(60, duration)  # SUSTAIN FOR 60s MINIMUM (or duration if longer)
    stats["sustaining_peak_for_s"] = sustain_duration
    stats["sustain_timeline"] = []

    await asyncio.sleep(1)  # Brief pause

    # Keep PEAK load sustained (this is what triggers actual failures!)
    sustain_steps = int(sustain_duration / 3)  # Every 3 seconds
    for sustain_step in range(sustain_steps):
        sustain_start = time.time()

        # Keep CPU at MAXIMUM (95-100%)
        if cpu_ramp:
            result = 0
            # Use maximum spin factor to keep CPU at 100%
            for _ in range(max_spin):
                result += math.sqrt(random.random() * 999)

        # Keep ADDING more memory during sustain to force OOM!
        if memory_leak and sustain_step % 3 == 0:  # Every 3rd step (every 9s)
            try:
                # Add another 200MB chunk during sustain
                extra_chunk = bytearray(200 * 1024 * 1024)
                extra_chunk[0] = 1
                extra_chunk[-1] = 1
                memory_chunks.append(extra_chunk)
            except MemoryError:
                # Perfect! We hit the limit!
                pass

        # Log every sustain step
        current_mem_mb = sum(len(c) for c in memory_chunks) // (1024 * 1024) if memory_leak else 0
        stats["sustain_timeline"].append({
            "sustain_step": sustain_step + 1,
            "elapsed_s": round(time.time() - start, 1),
            "memory_held_mb": current_mem_mb,
            "cpu_load": "100%" if cpu_ramp else "0%",
        })

        await asyncio.sleep(3)  # Sustain for 3 seconds per step

    # Record final time
    stats["total_elapsed_s"] = round(time.time() - start, 2)
    stats["total_network_sent_mb"] = round(network_bytes_sent / (1024 * 1024), 2) if network_ramp else 0

    # Cleanup
    memory_chunks.clear()

    # If network ramp enabled, return as streaming response to generate actual network traffic
    if network_ramp and network_bytes_sent > 0:
        async def stream_with_stats():
            # Send data in chunks to generate network traffic
            sent = 0
            chunk_size = 256 * 1024  # 256KB chunks
            while sent < network_bytes_sent:
                chunk = b"X" * min(chunk_size, network_bytes_sent - sent)
                yield chunk
                sent += len(chunk)
                await asyncio.sleep(0.001)  # Small delay to prevent blocking

            # Send final stats as JSON
            summary = json.dumps({"stats": stats})
            yield b"\n" + summary.encode()

        add_tracking_headers(response, "degrade", start)
        return StreamingResponse(stream_with_stats(), media_type="application/octet-stream")

    add_tracking_headers(response, "degrade", start)
    return JSONResponse(stats)

@app.get("/traffic-spike")
async def traffic_spike(
    response: Response,
    duration: int = Query(30, description="Duration of traffic spike in seconds"),
    cpu_load: int = Query(80, description="Target CPU percentage"),
    memory_mb: int = Query(512, description="Memory to allocate in MB"),
    network_mb: int = Query(50, description="Network traffic to generate in MB"),
):
    """
    Simulate a realistic traffic spike with CPU + Memory + Network load

    This endpoint is designed for MTTR_3 testing (high traffic scenario).
    It stresses CPU, Memory, AND Network simultaneously to trigger
    predictive scaling in the recovery manager.

    Use with concurrent requests to test load balancing and auto-scaling:
    ```
    for i in {1..10}; do
      curl "http://...:7777/traffic-spike?duration=10&cpu_load=70&memory_mb=256&network_mb=20" &
    done
    ```
    """
    start = time.time()

    stats = {
        "server_id": SERVER_ID,
        "scenario": "high_traffic_spike",
        "config": {
            "duration": duration,
            "cpu_target": cpu_load,
            "memory_mb": memory_mb,
            "network_mb": network_mb,
        }
    }

    # Calculate CPU spin to reach target CPU percentage
    # Rough estimation: 1M spins ≈ 10% CPU per worker
    workers = min(4, os.cpu_count() or 2)
    target_spin = int((cpu_load / 100) * 10_000_000)

    # Start all three loads concurrently
    pending_tasks: Dict[str, asyncio.Task] = {}

    # CPU load
    pending_tasks["cpu"] = asyncio.create_task(
        run_cpu_stress(duration, workers, target_spin)
    )

    # Memory load
    pending_tasks["memory"] = asyncio.create_task(
        run_memory_stress(memory_mb, duration, 1)
    )

    # Network load - stream data
    total_bytes = network_mb * 1024 * 1024
    chunk_size = 256 * 1024  # 256KB chunks

    async def stream_with_stats():
        sent = 0
        stream_start = time.time()

        while sent < total_bytes:
            chunk = b"X" * min(chunk_size, total_bytes - sent)
            yield chunk
            sent += len(chunk)
            await asyncio.sleep(0.001)  # Small delay to prevent blocking

        # Collect CPU and memory results
        if pending_tasks:
            results = await gather_task_results(pending_tasks)
            stats.update(results)

        stats["network"] = {
            "sent_mb": round(sent / (1024 * 1024), 2),
            "duration_s": round(time.time() - stream_start, 2),
        }
        stats["total_elapsed_s"] = round(time.time() - start, 2)

        # Send final stats as JSON at the end
        summary = json.dumps({"stats": stats})
        yield b"\n" + summary.encode()

    add_tracking_headers(response, "traffic-spike", start)
    return StreamingResponse(stream_with_stats(), media_type="application/octet-stream")

@app.get("/stress")
async def constant_stress(
    response: Response,
    size: int = Query(100000000, description="Size of data to stream in bytes (default 100MB)"),
):
    """
    Stream constant data for network stress testing.

    This endpoint is designed to be called repeatedly with curl --limit-rate
    to generate constant sustained network traffic at a precise rate.

    Example:
      # Generate 25 Mbps constant traffic:
      while true; do curl --limit-rate 3200K -o /dev/null "http://...:7777/stress?size=100000000"; done
    """
    async def stream_constant_data():
        sent = 0
        chunk_size = 64 * 1024  # 64KB chunks

        while sent < size:
            chunk = b"X" * min(chunk_size, size - sent)
            yield chunk
            sent += len(chunk)
            # No sleep - let curl's --limit-rate control the bandwidth

    add_tracking_headers(response, "stress", time.time())
    return StreamingResponse(stream_constant_data(), media_type="application/octet-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7777)
