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
):
    """Flexible stress endpoint - mix CPU, memory, network"""
    start = time.time()
    
    stats = {
        "requested": {"cpu": cpu, "memory": memory, "network": network},
        "server_id": SERVER_ID,
    }

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7777)
