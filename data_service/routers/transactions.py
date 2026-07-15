"""
SAFER Data Service — Transaction Router

CRUD operations for transactions, batch generation, and audit workflows.
"""

import json
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from data_service.database import get_db
from data_service.models import Transaction, AuditLog, Alert
from data_service.schemas import (
    TransactionCreate, TransactionResponse, TransactionListResponse,
    AuditStatusUpdate, AuditLogResponse, BatchGenerateRequest,
)
from data_service.tx_generator import generate_batch
from shared.config import SCORING_SERVICE_URL

router = APIRouter(prefix="/transactions", tags=["transactions"])


# ─── List Transactions ──────────────────────────────────────────────────────

@router.get("", response_model=TransactionListResponse)
def list_transactions(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    severity: Optional[str] = None,
    audit_status: Optional[str] = None,
    rail: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """List transactions with pagination and optional filters."""
    q = db.query(Transaction).order_by(Transaction.timestamp.desc())

    if severity:
        q = q.filter(Transaction.severity == severity)
    if audit_status:
        q = q.filter(Transaction.audit_status == audit_status)
    if rail:
        q = q.filter(Transaction.payment_rail == rail)

    total = q.count()
    items = q.offset((page - 1) * page_size).limit(page_size).all()

    return TransactionListResponse(
        items=[TransactionResponse.model_validate(tx) for tx in items],
        total=total,
        page=page,
        page_size=page_size,
    )


# ─── Get Single Transaction ────────────────────────────────────────────────

@router.get("/{tx_id}", response_model=TransactionResponse)
def get_transaction(tx_id: str, db: Session = Depends(get_db)):
    """Get a single transaction by ID."""
    tx = db.query(Transaction).filter(Transaction.id == tx_id).first()
    if not tx:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return TransactionResponse.model_validate(tx)


# ─── Create Transaction (pre-scored) ───────────────────────────────────────

@router.post("", response_model=TransactionResponse, status_code=201)
def create_transaction(data: TransactionCreate, db: Session = Depends(get_db)):
    """Store a pre-scored transaction in the database."""
    tx = Transaction(**data.model_dump())
    db.add(tx)
    db.commit()
    db.refresh(tx)

    # Auto-create alert for high/critical
    if tx.severity in ("high", "critical"):
        alert = Alert(
            transaction_id=tx.id,
            alert_type="high_risk",
            severity=tx.severity,
            message=f"Transaction {tx.id} flagged as {tx.severity} risk (score: {tx.risk_score})",
        )
        db.add(alert)
        db.commit()

    return TransactionResponse.model_validate(tx)


# ─── Purge All Transactions ────────────────────────────────────────────────

@router.delete("", status_code=204)
def purge_transactions(db: Session = Depends(get_db)):
    """Delete all transactions, audit logs, and alerts in the database."""
    try:
        db.query(Alert).delete()
        db.query(AuditLog).delete()
        db.query(Transaction).delete()
        db.commit()
        return None
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to purge database: {str(e)}")


# ─── Batch Generate (with ML scoring) ──────────────────────────────────────

@router.post("/batch-generate", response_model=list[TransactionResponse])
async def batch_generate(req: BatchGenerateRequest, db: Session = Depends(get_db)):
    """
    Generate synthetic transactions, score them via Scoring Service, save to DB.
    If Scoring Service is unavailable, transactions are saved without ML scores.
    """
    raw_txs = generate_batch(count=req.count, fraud_ratio=req.fraud_ratio)
    results = []

    for raw in raw_txs:
        # Try to score via Scoring Service
        scored = False
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{SCORING_SERVICE_URL}/score",
                    json=_serialize_for_scoring(raw),
                )
                if resp.status_code == 200:
                    score_data = resp.json()
                    raw["risk_score"] = score_data.get("risk_score", 0)
                    raw["severity"] = score_data.get("severity", "low")
                    raw["fraud_probability"] = score_data.get("fraud_probability", 0.0)
                    raw["xgb_probability"] = score_data.get("xgb_probability", 0.0)
                    raw["lgb_probability"] = score_data.get("lgb_probability", 0.0)
                    raw["ai_reasoning"] = score_data.get("ai_reasoning", "")
                    raw["shap_values"] = json.dumps(score_data.get("shap_values", {}))
                    raw["primary_risk_factors"] = json.dumps(score_data.get("primary_risk_factors", []))
                    raw["suggested_action"] = score_data.get("suggested_action", "")
                    scored = True
        except Exception:
            pass  # Scoring service unavailable — save without scores

        if not scored:
            raw["risk_score"] = 0
            raw["severity"] = "low"
            raw["fraud_probability"] = 0.0

        # Set audit status based on severity
        if raw.get("severity") == "critical":
            raw["audit_status"] = "blocked"
        elif raw.get("severity") == "high":
            raw["audit_status"] = "pending_review"
        else:
            raw["audit_status"] = "pending_review"

        # Save to DB
        tx = Transaction(**{k: v for k, v in raw.items() if hasattr(Transaction, k)})
        db.add(tx)
        db.commit()
        db.refresh(tx)

        # Auto-create alert
        if tx.severity in ("high", "critical"):
            alert = Alert(
                transaction_id=tx.id,
                alert_type="high_risk",
                severity=tx.severity,
                message=f"Transaction {tx.id} flagged as {tx.severity} risk (score: {tx.risk_score})",
            )
            db.add(alert)
            db.commit()

        results.append(TransactionResponse.model_validate(tx))

    return results


# ─── Update Audit Status ───────────────────────────────────────────────────

@router.patch("/{tx_id}/audit", response_model=TransactionResponse)
def update_audit_status(
    tx_id: str,
    data: AuditStatusUpdate,
    db: Session = Depends(get_db),
):
    """Update the audit status of a transaction."""
    tx = db.query(Transaction).filter(Transaction.id == tx_id).first()
    if not tx:
        raise HTTPException(status_code=404, detail="Transaction not found")

    old_status = tx.audit_status

    # Update transaction
    tx.audit_status = data.status
    tx.audit_notes = data.notes or tx.audit_notes
    tx.audited_by = data.changed_by
    tx.audited_at = datetime.now(timezone.utc)

    # Create audit log entry
    log = AuditLog(
        transaction_id=tx_id,
        from_status=old_status,
        to_status=data.status,
        changed_by=data.changed_by,
        notes=data.notes,
    )
    db.add(log)
    db.commit()
    db.refresh(tx)

    return TransactionResponse.model_validate(tx)


# ─── Get Audit History ──────────────────────────────────────────────────────

@router.get("/{tx_id}/audit-history", response_model=list[AuditLogResponse])
def get_audit_history(tx_id: str, db: Session = Depends(get_db)):
    """Get all audit log entries for a transaction."""
    logs = (
        db.query(AuditLog)
        .filter(AuditLog.transaction_id == tx_id)
        .order_by(AuditLog.created_at.desc())
        .all()
    )
    return [AuditLogResponse.model_validate(log) for log in logs]


# ─── Helpers ────────────────────────────────────────────────────────────────

def _serialize_for_scoring(tx: dict) -> dict:
    """Convert transaction dict to JSON-serializable format for Scoring Service."""
    result = {}
    for k, v in tx.items():
        if isinstance(v, datetime):
            result[k] = v.isoformat()
        elif isinstance(v, bool):
            result[k] = int(v)
        else:
            result[k] = v
    return result
