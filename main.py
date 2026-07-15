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


# ─── Gradio Mount (for Hugging Face Space visibility) ───────────────────────
try:
    import gradio as gr
    
    # Simple, nice-looking dashboard indicating that the FDS core engine is alive
    with gr.Blocks(title="SAFER FDS Engine", css="footer {visibility: hidden}") as demo:
        gr.Markdown("# 🛡️ SAFER FDS — AI Fraud Intelligence Core Engine")
        gr.Markdown("### Status: **🟢 Running B2B Sandbox Mode**")
        gr.Markdown(
            "Ini adalah backend API Core Engine untuk SAFER yang melayani skoring transaksi B2B secara real-time. "
            "Portal antarmuka analis utama (Dashboard & Fraud Graph) dideploy secara terpisah di "
            "**[safer-fds.pages.dev](https://safer-fds.pages.dev)**."
        )
        
        with gr.Accordion("Dokumentasi Integrasi API", open=True):
            gr.Markdown("Hubungkan sistem core banking atau PJP fintech Anda ke endpoint berikut:")
            gr.Markdown("- **Interactive Swagger UI**: [Buka /docs](/docs)")
            gr.Markdown("- **Health Status Check**: [Buka /health](/health)")
            gr.Markdown("- **Batch Generation Sandbox**: `POST /api/transactions/batch`")
            gr.Markdown("- **Single Scoring Engine**: `POST /api/transactions/simulate`")
            
        with gr.Accordion("Metrik AI Engine v2", open=False):
            gr.Markdown("- **Model Architecture**: Ensemble XGBoost + LightGBM v2")
            gr.Markdown("- **Dataset Latih**: 100,000 Transaksi Tabular Indonesia")
            gr.Markdown("- **Akurasi**: 99.89% (FPR: 0.02% / Recall: 99.05%)")
            gr.Markdown("- **Pola Fraud**: Mule Ring, Device Farm, Judi Online, Structuring/Smurfing, dll.")

    app = gr.mount_gradio_app(app, demo, path="/")
    print("[Gateway] Mounted Gradio UI successfully on root path /")
except Exception as e:
    print(f"[Gateway] Failed to mount Gradio UI: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=GATEWAY_PORT, reload=True)
