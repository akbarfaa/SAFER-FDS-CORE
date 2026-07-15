"""
SAFER Backend Unified API Gateway

Combines and reverse-proxies the SAFER microservices under the unified "/api" namespace.
  - Port 8000: API Gateway (Unified Entry Point)
  - Port 8001: Scoring Service (Ensemble ML & SHAP Explainer)
  - Port 8002: Graph Service (NetworkX Analytics)
  - Port 8003: Data Service (Database Engine & CRUD)
"""

import sys
import os
import httpx
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.dirname(__file__))

from shared.config import (
    CORS_ORIGINS,
    GATEWAY_PORT,
    SCORING_SERVICE_URL,
    GRAPH_SERVICE_URL,
    DATA_SERVICE_URL,
    IS_PRODUCTION,
)

app = FastAPI(
    title="SAFER Unified API Gateway",
    description="Unified entry point for SAFER microservices. Sandbox Mode active.",
    version="2.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiter in production/sandbox mode
if IS_PRODUCTION:
    from middleware.rate_limiter import RateLimiterMiddleware
    app.add_middleware(RateLimiterMiddleware)

# Async HTTP Client for routing/proxying
client = httpx.AsyncClient()


@app.on_event("shutdown")
async def shutdown_event():
    await client.aclose()


# Helper function to proxy requests to internal microservices
async def proxy_request(service_url: str, path: str, request: Request) -> Response:
    url = f"{service_url}{path}"
    headers = dict(request.headers)
    
    # Remove host header to avoid routing loops/errors
    headers.pop("host", None)
    headers.pop("content-length", None)

    method = request.method
    content = await request.body()
    params = request.query_params

    try:
        resp = await client.request(
            method,
            url,
            headers=headers,
            params=params,
            content=content,
            timeout=30.0
        )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=dict(resp.headers)
        )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Service at {service_url} is currently unavailable. Error: {exc}"
        )


# ─── API Routes ─────────────────────────────────────────────────────────────

# 1. Simulator: score manual input without saving to database
@app.post("/api/transactions/simulate")
async def simulate_transaction(request: Request):
    """Proxies directly to ML Scoring Service to score manual inputs without saving."""
    return await proxy_request(SCORING_SERVICE_URL, "/score", request)


# 2. Batch generate synthetic transactions
@app.post("/api/transactions/batch")
async def batch_generate_proxy(request: Request):
    """Proxies batch generation to Data Service (with internal ML scoring)."""
    return await proxy_request(DATA_SERVICE_URL, "/transactions/batch-generate", request)


# 3. Transaction routes mapped to Data Service
@app.post("/api/transactions")
@app.get("/api/transactions")
async def transactions_crud_proxy(request: Request):
    return await proxy_request(DATA_SERVICE_URL, "/transactions", request)


@app.get("/api/transactions/{tx_id}")
@app.patch("/api/transactions/{tx_id}/audit")
@app.get("/api/transactions/{tx_id}/audit-history")
async def transaction_detail_proxy(tx_id: str, request: Request):
    # Extract relative path (e.g. /transactions/TX-12345/audit)
    path = request.url.path.replace("/api", "")
    return await proxy_request(DATA_SERVICE_URL, path, request)


# 4. Analytics routes mapped to Data Service
@app.get("/api/analytics/dashboard")
@app.get("/api/analytics/risk-distribution")
@app.get("/api/analytics/rail-distribution")
@app.get("/api/analytics/trend")
async def analytics_proxy(request: Request):
    path = request.url.path.replace("/api", "")
    return await proxy_request(DATA_SERVICE_URL, path, request)


# 5. Graph scenarios mapped to Graph Service
@app.get("/api/graph/scenarios")
async def graph_scenarios_proxy(request: Request):
    return await proxy_request(GRAPH_SERVICE_URL, "/scenarios", request)


# 6. Graph dynamic analysis
@app.post("/api/graph/analyze")
async def graph_analyze_proxy(request: Request):
    return await proxy_request(GRAPH_SERVICE_URL, "/analyze", request)


# 7. Investigate cluster bulk update
@app.post("/api/graph/investigate")
async def graph_investigate_proxy(request: Request):
    """
    Saves investigation status of a cluster. Maps to the batch audit status update
    on the Data Service (port 8003).
    """
    try:
        payload = await request.json()
        tx_ids = payload.get("transaction_ids", [])
        notes = payload.get("notes", "Cluster investigation")
        changed_by = payload.get("changed_by", "Analyst Demo")
        
        # We can sequentially PATCH each transaction or update them in database.
        # Let's hit Data Service's update endpoint for each tx_id
        results = []
        for tx_id in tx_ids:
            patch_url = f"{DATA_SERVICE_URL}/transactions/{tx_id}/audit"
            await client.patch(
                patch_url,
                json={"status": "under_investigation", "notes": notes, "changed_by": changed_by}
            )
        return {"status": "success", "message": f"{len(tx_ids)} transactions marked as under_investigation."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {
        "gateway": "ok",
        "version": "2.0.0",
        "sandbox_mode": IS_PRODUCTION,
        "port": GATEWAY_PORT,
        "services": {
            "scoring": SCORING_SERVICE_URL,
            "graph": GRAPH_SERVICE_URL,
            "data": DATA_SERVICE_URL
        },
        "model": {
            "engine": "XGBoost + LightGBM Ensemble",
            "version": "v2",
            "fraud_patterns": 8,
            "dataset_size": "100K transactions",
        }
    }


# ─── HTML Root Route (for Hugging Face Space visibility and healthcheck) ────
from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def index():
    html_content = """
    <!DOCTYPE html>
    <html lang="id">
    <head>
        <title>SAFER FDS — AI Fraud Intelligence Core Engine</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                background-color: #0b0f19;
                color: #f3f4f6;
                margin: 0;
                padding: 0;
                display: flex;
                align-items: center;
                justify-content: center;
                min-height: 100vh;
            }
            .card {
                background: rgba(255, 255, 255, 0.02);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 20px;
                padding: 40px;
                max-width: 550px;
                width: 90%;
                box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
                backdrop-filter: blur(12px);
                text-align: center;
            }
            h1 {
                font-size: 32px;
                font-weight: 800;
                margin-top: 0;
                margin-bottom: 8px;
                background: linear-gradient(135deg, #ef4444 0%, #6366f1 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                letter-spacing: -0.5px;
            }
            .status {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                background: rgba(16, 185, 129, 0.08);
                color: #10b981;
                padding: 6px 16px;
                border-radius: 20px;
                font-weight: 600;
                font-size: 13px;
                margin-bottom: 24px;
                border: 1px solid rgba(16, 185, 129, 0.15);
                letter-spacing: 0.5px;
                text-transform: uppercase;
            }
            .status-dot {
                width: 8px;
                height: 8px;
                background-color: #10b981;
                border-radius: 50%;
                box-shadow: 0 0 10px #10b981;
                animation: pulse 2s infinite;
            }
            @keyframes pulse {
                0% { transform: scale(0.9); opacity: 0.6; }
                50% { transform: scale(1.1); opacity: 1; }
                100% { transform: scale(0.9); opacity: 0.6; }
            }
            p {
                color: #9ca3af;
                font-size: 15px;
                line-height: 1.6;
                margin-bottom: 32px;
            }
            .links {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 12px;
            }
            .btn {
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 12px;
                border-radius: 8px;
                text-decoration: none;
                font-weight: 600;
                font-size: 14px;
                transition: all 0.2s;
            }
            .btn-primary {
                background: #6366f1;
                color: #ffffff;
                box-shadow: 0 4px 14px rgba(99, 102, 241, 0.3);
            }
            .btn-primary:hover {
                background: #4f46e5;
                transform: translateY(-1px);
            }
            .btn-secondary {
                background: rgba(255, 255, 255, 0.03);
                color: #f3f4f6;
                border: 1px solid rgba(255, 255, 255, 0.08);
            }
            .btn-secondary:hover {
                background: rgba(255, 255, 255, 0.08);
                transform: translateY(-1px);
            }
            .footer {
                margin-top: 36px;
                font-size: 12px;
                color: #4b5563;
                border-top: 1px solid rgba(255, 255, 255, 0.05);
                padding-top: 20px;
            }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🛡️ SAFER FDS</h1>
            <div class="status">
                <span class="status-dot"></span>
                Running B2B Sandbox Mode
            </div>
            <p>
                Ini adalah API Core Engine untuk SAFER yang melayani skoring transaksi B2B secara real-time. 
                Portal antarmuka analis utama (Dashboard & Fraud Graph) di-deploy secara terpisah di 
                <a href="https://safer-fds.pages.dev" target="_blank" style="color: #6366f1; text-decoration: none; font-weight: 500;">safer-fds.pages.dev</a>.
            </p>
            <div class="links">
                <a href="/docs" class="btn btn-primary">Swagger API Docs</a>
                <a href="/health" class="btn btn-secondary">Health Status</a>
            </div>
            <div class="footer">
                Model: Ensemble XGBoost + LightGBM v2 | Dataset: 100K Indonesia
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=GATEWAY_PORT, reload=True)
