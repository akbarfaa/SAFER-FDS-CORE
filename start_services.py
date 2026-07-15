"""
SAFER FDS — Unified Production Startup Script

Designed for Render.com deployment: runs all microservices within a single
process tree. The gateway listens on the PORT env var (Render provides this).
Internal services use fixed high ports for localhost-only communication.
"""

import os
import sys
import subprocess
import time
import signal
import threading

# ─── Port Configuration ────────────────────────────────────────────────────
# Render.com provides PORT env var for the public-facing service
GATEWAY_PORT = int(os.getenv("PORT", "8000"))

# Internal microservice ports (localhost only, not exposed to internet)
SCORING_PORT = int(os.getenv("SCORING_PORT", "8001"))
GRAPH_PORT = int(os.getenv("GRAPH_PORT", "8002"))
DATA_PORT = int(os.getenv("DATA_PORT", "8003"))

# Set environment variables for the gateway to discover internal services
os.environ["GATEWAY_PORT"] = str(GATEWAY_PORT)
os.environ["SCORING_PORT"] = str(SCORING_PORT)
os.environ["GRAPH_PORT"] = str(GRAPH_PORT)
os.environ["DATA_PORT"] = str(DATA_PORT)
os.environ["SCORING_SERVICE_URL"] = f"http://127.0.0.1:{SCORING_PORT}"
os.environ["GRAPH_SERVICE_URL"] = f"http://127.0.0.1:{GRAPH_PORT}"
os.environ["DATA_SERVICE_URL"] = f"http://127.0.0.1:{DATA_PORT}"

# Detect the correct python executable
PYTHON = sys.executable
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Service Definitions ───────────────────────────────────────────────────
INTERNAL_SERVICES = [
    {
        "name": "Data Service",
        "module": os.path.join(BACKEND_DIR, "data_service", "main.py"),
        "port": DATA_PORT,
    },
    {
        "name": "Scoring Service",
        "module": os.path.join(BACKEND_DIR, "scoring_service", "main.py"),
        "port": SCORING_PORT,
    },
    {
        "name": "Graph Service",
        "module": os.path.join(BACKEND_DIR, "graph_service", "main.py"),
        "port": GRAPH_PORT,
    },
]

processes = []


def start_service(service):
    """Start a microservice as a subprocess."""
    print(f"[Launcher] Starting {service['name']} on port {service['port']}...")
    proc = subprocess.Popen(
        [PYTHON, service["module"]],
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=os.environ.copy(),
    )
    processes.append((proc, service["name"]))
    return proc


def shutdown_all():
    """Gracefully shutdown all services."""
    print("\n[Launcher] Shutting down all services...")
    for proc, name in processes:
        try:
            proc.terminate()
            proc.wait(timeout=3)
            print(f"  ✓ {name} stopped")
        except Exception:
            try:
                proc.kill()
                print(f"  ✗ {name} killed")
            except Exception:
                pass


def signal_handler(sig, frame):
    shutdown_all()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def main():
    print("=" * 60)
    print("  SAFER FDS — Production Launcher")
    print(f"  Gateway Port: {GATEWAY_PORT}")
    print(f"  Environment: {'Production (Render)' if os.getenv('RENDER') else 'Local'}")
    print("=" * 60)

    # 1. Start internal microservices
    for svc in INTERNAL_SERVICES:
        start_service(svc)

    # Wait for internal services to initialize
    print("[Launcher] Waiting for internal services to initialize...")
    time.sleep(3)

    # 2. Start the API Gateway as the main process
    print(f"[Launcher] Starting API Gateway on port {GATEWAY_PORT}...")
    import uvicorn

    # Add backend directory to path for imports
    sys.path.insert(0, BACKEND_DIR)

    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=GATEWAY_PORT,
            workers=1,
            log_level="info",
        )
    except KeyboardInterrupt:
        pass
    finally:
        shutdown_all()


if __name__ == "__main__":
    main()
