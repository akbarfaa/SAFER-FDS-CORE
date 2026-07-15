"""
SAFER Data Service — Analytics Router

Aggregated dashboard KPIs, risk distribution, rail distribution, and trend data.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, case

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from data_service.database import get_db
from data_service.models import Transaction
from data_service.schemas import DashboardStats

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/dashboard", response_model=DashboardStats)
def dashboard_stats(db: Session = Depends(get_db)):
    """Aggregated KPI stats for the monitoring dashboard."""
    txs = db.query(Transaction).all()

    if not txs:
        return DashboardStats()

    total = len(txs)
    total_amount = sum(t.amount for t in txs)
    flagged = [t for t in txs if t.severity != "low"]
    flagged_count = len(flagged)
    flagged_amount = sum(t.amount for t in flagged)
    blocked_count = sum(1 for t in txs if t.audit_status == "blocked")
    pending = sum(1 for t in txs if t.audit_status in ("pending_review", "under_investigation"))

    by_severity = {"low": 0, "medium": 0, "high": 0, "critical": 0}
    for t in txs:
        sev = t.severity or "low"
        by_severity[sev] = by_severity.get(sev, 0) + 1

    by_audit = {}
    for t in txs:
        status = t.audit_status or "pending_review"
        by_audit[status] = by_audit.get(status, 0) + 1

    by_rail = {}
    for t in txs:
        rail = t.payment_rail or "Unknown"
        by_rail[rail] = by_rail.get(rail, 0) + 1

    return DashboardStats(
        total_transactions=total,
        total_amount=total_amount,
        flagged_count=flagged_count,
        flagged_amount=flagged_amount,
        blocked_count=blocked_count,
        pending_review=pending,
        by_severity=by_severity,
        by_audit_status=by_audit,
        by_rail=by_rail,
    )


@router.get("/risk-distribution")
def risk_distribution(db: Session = Depends(get_db)):
    """Risk score distribution for histogram/chart."""
    txs = db.query(Transaction.risk_score).all()
    buckets = [0] * 10  # 0-9, 10-19, ..., 90-100
    for (score,) in txs:
        idx = min(9, (score or 0) // 10)
        buckets[idx] += 1

    return {
        "buckets": [
            {"range": f"{i*10}-{i*10+9}", "count": buckets[i]}
            for i in range(10)
        ]
    }


@router.get("/rail-distribution")
def rail_distribution(db: Session = Depends(get_db)):
    """Transaction count per payment rail."""
    results = (
        db.query(Transaction.payment_rail, func.count(Transaction.id))
        .group_by(Transaction.payment_rail)
        .all()
    )
    return {
        "rails": [{"rail": rail, "count": count} for rail, count in results]
    }


@router.get("/trend")
def trend_data(db: Session = Depends(get_db)):
    """Transaction trend over time (grouped by hour)."""
    txs = db.query(Transaction.timestamp, Transaction.severity).order_by(Transaction.timestamp).all()

    if not txs:
        return {"points": []}

    # Group by hour
    hourly: dict[str, dict] = {}
    for ts, sev in txs:
        if ts is None:
            continue
        hour_key = ts.strftime("%Y-%m-%d %H:00")
        if hour_key not in hourly:
            hourly[hour_key] = {"time": hour_key, "total": 0, "flagged": 0}
        hourly[hour_key]["total"] += 1
        if sev and sev != "low":
            hourly[hour_key]["flagged"] += 1

    return {"points": list(hourly.values())}
