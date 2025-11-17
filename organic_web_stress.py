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
from datetime import datetime
from typing import Any, Optional

from fastapi import FastAPI, Query, Response
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

app = FastAPI(title="Organic Web Stress", version="2.0")

SERVER_ID = socket.gethostname()
request_counter = {"total": 0, "by_endpoint": {}}

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
    memory_mb: int = Query(128),
    memory_hold: float = Query(1.0),
    network_mb: int = Query(5),
):
    """Flexible stress endpoint - mix CPU, memory, network"""
    start = time.time()
    
    stats = {
        "requested": {"cpu": cpu, "memory": memory, "network": network},
        "server_id": SERVER_ID,
    }
    
    # CPU work
    if cpu:
        cpu_start = time.time()
        iterations = 0
        deadline = cpu_start + cpu_duration
        
        # Simulate multi-worker by doing more iterations
        multiplier = max(1, cpu_workers)
        
        while time.time() < deadline:
            for _ in range(multiplier * 100_000):
                math.sqrt(random.random() * 9999)
                iterations += 1
        
        stats["cpu"] = {
            "duration": round(time.time() - cpu_start, 3),
            "iterations": iterations,
            "workers": cpu_workers,
        }
    
    # Memory allocation
    if memory:
        mem_start = time.time()
        chunks = []
        
        try:
            # Allocate in 64MB chunks
            for _ in range(max(1, memory_mb // 64)):
                chunk = bytearray(64 * 1024 * 1024)
                chunk[0] = 1
                chunk[-1] = 1
                chunks.append(chunk)
            
            # Hold memory
            if memory_hold > 0:
                time.sleep(memory_hold)
            
            allocated = sum(len(c) for c in chunks)
        except MemoryError:
            allocated = sum(len(c) for c in chunks)
        finally:
            chunks.clear()
        
        stats["memory"] = {
            "requested_mb": memory_mb,
            "allocated_mb": round(allocated / (1024 * 1024), 2),
            "hold_seconds": memory_hold,
            "elapsed": round(time.time() - mem_start, 3),
        }
    
    # Network transfer
    if network:
        total_bytes = network_mb * 1024 * 1024
        chunk_size = 256 * 1024  # 256KB chunks
        
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
            }
            summary = json.dumps({"stats": stats})
            yield b"\n" + summary.encode()
        
        add_tracking_headers(response, "stress", start)
        return StreamingResponse(stream(), media_type="application/octet-stream")
    
    # No network - return JSON
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