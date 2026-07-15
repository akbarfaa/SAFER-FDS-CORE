import os
import sys
import subprocess
import time
import signal

# Hugging Face sets PORT env var automatically to 7860
GATEWAY_PORT = int(os.getenv("PORT", "7860"))

SCORING_PORT = 8001
GRAPH_PORT = 8002
DATA_PORT = 8003

os.environ["GATEWAY_PORT"] = str(GATEWAY_PORT)
os.environ["SCORING_PORT"] = str(SCORING_PORT)
os.environ["GRAPH_PORT"] = str(GRAPH_PORT)
os.environ["SCORING_SERVICE_URL"] = f"http://127.0.0.1:{SCORING_PORT}"
os.environ["GRAPH_SERVICE_URL"] = f"http://127.0.0.1:{GRAPH_PORT}"
os.environ["DATA_SERVICE_URL"] = f"http://127.0.0.1:{DATA_PORT}"

# Override config environment
os.environ["IS_PRODUCTION"] = "true"
os.environ["RENDER"] = "true"  # keeps configs in production mode

PYTHON = sys.executable
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

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


def shutdown_all():
    print("\n[Launcher] Shutting down all services...")
    for proc, name in processes:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def signal_handler(sig, frame):
    shutdown_all()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def main():
    print("=" * 60)
    print("  SAFER FDS — Hugging Face Production Launcher (Gradio Wrapper)")
    print(f"  Gateway Port: {GATEWAY_PORT}")
    print("=" * 60)

    # 1. Start internal microservices
    for svc in INTERNAL_SERVICES:
        print(f"[Launcher] Starting {svc['name']} on port {svc['port']}...")
        proc = subprocess.Popen(
            [PYTHON, svc["module"]],
            stdout=sys.stdout,
            stderr=sys.stderr,
            env=os.environ.copy(),
        )
        processes.append((proc, svc["name"]))

    print("[Launcher] Waiting for internal services to initialize...")
    time.sleep(4)

    # 2. Run Gateway (main.py FastAPI) on PORT 7860
    print(f"[Launcher] Starting API Gateway on port {GATEWAY_PORT}...")
    import uvicorn

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
