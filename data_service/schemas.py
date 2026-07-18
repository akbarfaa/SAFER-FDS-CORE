"""
SAFER Data Service — Pydantic Schemas

Request / response validation models.
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ─── Transaction ────────────────────────────────────────────────────────────

class TransactionBase(BaseModel):
    """Fields shared between create and response."""
    sender_name: str = ""
    sender_account: str = ""
    sender_bank: str = ""
    sender_city: str = ""
    sender_province: str = ""
    sender_lat: float = 0.0
    sender_lng: float = 0.0
    receiver_name: str = ""
    receiver_account: str = ""
    receiver_bank: str = ""
    receiver_city: str = ""
    receiver_province: str = ""
    receiver_lat: float = 0.0
    receiver_lng: float = 0.0
    amount: float = 0.0
    payment_rail: str = ""
    ewallet_provider: str = "None"
    merchant: str = ""
    merchant_category: str = ""
    channel: str = ""
    device_type: str = ""
    device_brand: str = ""
    device_fingerprint: str = ""
    ip_address: str = ""
    is_new_device: bool = False
    account_age_days: int = 365
    # Fraud indicators
    is_velocity_anomaly: bool = False
    is_geo_mismatch: bool = False
    is_off_hours: bool = False
    is_high_value_for_rail: bool = False
    is_suspicious_ip: bool = False
    is_risky_merchant: bool = False
    is_new_account: bool = False
    has_failed_attempts: bool = False
    is_device_mismatch: bool = False
    is_sim_swap: bool = False
    is_unusual_beneficiary: bool = False
    velocity_count: int = 1
    geo_distance_km: float = 0.0


class TransactionCreate(TransactionBase):
    """Used when submitting a pre-scored transaction to the DB."""
    id: str
    timestamp: datetime
    risk_score: int = 0
    severity: str = "low"
    fraud_probability: float = 0.0
    xgb_probability: float = 0.0
    lgb_probability: float = 0.0
    ai_reasoning: str = ""
    shap_values: str = ""
    primary_risk_factors: str = ""
    suggested_action: str = ""
    audit_status: str = "pending_review"
    is_fraud: bool = False


class TransactionResponse(TransactionCreate):
    """Full transaction record returned by API."""
    audit_notes: str = ""
    audited_by: Optional[str] = None
    audited_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    model_version: str = "v1"

    class Config:
        from_attributes = True


class TransactionListResponse(BaseModel):
    """Paginated list of transactions."""
    items: list[TransactionResponse]
    total: int
    page: int
    page_size: int


# ─── Audit ──────────────────────────────────────────────────────────────────

class AuditStatusUpdate(BaseModel):
    """Request body for updating audit status."""
    status: str
    notes: str = ""
    changed_by: str = "Analyst Demo"


class AuditLogResponse(BaseModel):
    """Single audit log entry."""
    id: int
    transaction_id: str
    from_status: str
    to_status: str
    changed_by: str
    notes: str
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ─── Analytics ──────────────────────────────────────────────────────────────

class DashboardStats(BaseModel):
    """Aggregated KPI stats for dashboard."""
    total_transactions: int = 0
    total_amount: float = 0.0
    flagged_count: int = 0
    flagged_amount: float = 0.0
    blocked_count: int = 0
    pending_review: int = 0
    by_severity: dict[str, int] = {}
    by_audit_status: dict[str, int] = {}
    by_rail: dict[str, int] = {}


class RiskDistribution(BaseModel):
    """Risk score distribution data."""
    buckets: list[dict] = []


class TrendData(BaseModel):
    """Trend data over time."""
    points: list[dict] = []


# ─── Batch Generation ──────────────────────────────────────────────────────

class BatchGenerateRequest(BaseModel):
    """Request to generate batch of synthetic transactions."""
    count: int = Field(default=5, ge=1, le=100)
    fraud_ratio: float = Field(default=0.18, ge=0.0, le=1.0)


# ─── Lead Registration & API Keys ─────────────────────────────────────────

class LeadCreate(BaseModel):
    name: str
    email: str
    company: str
    position: str
    interest_model: str


class LeadResponse(BaseModel):
    id: int
    name: str
    email: str
    company: str
    position: str
    interest_model: str
    client_id: str
    client_secret: str
    created_at: datetime

    class Config:
        from_attributes = True
