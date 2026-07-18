"""
SAFER Data Service — FastAPI Application

Manages database, transaction CRUD, and analytics aggregation.
Runs on port 8003.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from shared.config import CORS_ORIGINS, DATA_PORT
from data_service.database import init_db
from data_service.routers import transactions, analytics, leads

app = FastAPI(
    title="SAFER Data Service",
    description="Database CRUD, transaction management, and analytics aggregation.",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(transactions.router)
app.include_router(analytics.router)
app.include_router(leads.router)


@app.on_event("startup")
def on_startup():
    """Initialize database tables on service start."""
    init_db()
    print(f"[Data Service] Running on port {DATA_PORT}")


@app.get("/health")
def health():
    return {"service": "data", "status": "ok", "port": DATA_PORT}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("data_service.main:app", host="0.0.0.0", port=DATA_PORT, reload=True)
