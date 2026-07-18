import secrets
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from data_service.database import get_db
from data_service.models import LeadRegistration
from data_service.schemas import LeadCreate, LeadResponse
from typing import List

router = APIRouter(prefix="/leads", tags=["leads"])

@router.post("", response_model=LeadResponse, status_code=status.HTTP_201_CREATED)
def create_lead(lead_in: LeadCreate, db: Session = Depends(get_db)):
    """Create a new lead registration and automatically generate sandbox API credentials."""
    # Generate API key pair
    client_id = f"sfr_client_{secrets.token_hex(8)}"
    client_secret = f"sfr_secret_{secrets.token_hex(16)}"
    
    db_lead = LeadRegistration(
        name=lead_in.name,
        email=lead_in.email,
        company=lead_in.company,
        position=lead_in.position,
        interest_model=lead_in.interest_model,
        client_id=client_id,
        client_secret=client_secret
    )
    
    db.add(db_lead)
    db.commit()
    db.refresh(db_lead)
    return db_lead

@router.get("", response_model=List[LeadResponse])
def get_leads(db: Session = Depends(get_db)):
    """List all lead registrations (Admin only concept)."""
    return db.query(LeadRegistration).order_by(LeadRegistration.created_at.desc()).all()

@router.post("/validate")
def validate_lead_credentials(
    credentials: dict, 
    db: Session = Depends(get_db)
):
    """
    Validate Client ID & Client Secret. Called internally by the Gateway.
    Expects json payload: {"client_id": "...", "client_secret": "..."}
    """
    client_id = credentials.get("client_id")
    client_secret = credentials.get("client_secret")
    
    if not client_id or not client_secret:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing client_id or client_secret"
        )
        
    lead = db.query(LeadRegistration).filter(
        LeadRegistration.client_id == client_id,
        LeadRegistration.client_secret == client_secret
    ).first()
    
    if not lead:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
        
    return {
        "status": "authorized",
        "company": lead.company,
        "name": lead.name
    }
