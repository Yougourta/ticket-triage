import pytest

from pydantic import ValidationError
from src.models import OriginalTicket, ClassifiedTicket

def test_valid_original_ticket():
    ticket = OriginalTicket(
            id="PROJ-001",
            summary="Cannot login to customer portal",
            description="Since this morning I cannot access my account. Password reset email never received. I have a payment due tonight.",
            reporter="john.doe@company.com",
            created_at="2026-03-03T08:30:00Z",
            type="Bug"
    )
    assert ticket.id == "PROJ-001"

def test_invalid_email_in_original_ticket():
    with pytest.raises(ValidationError):
        OriginalTicket(
            id="PROJ-002",
            summary="Request for new feature",
            description="I would like to have a dark mode option in the settings.",
            reporter="invalid-email",
            created_at="2026-03-03T09:00:00Z",
            type="Feature"
        )

def test_valid_classified_ticket():
    classified_ticket_data = ClassifiedTicket(
        id="PROJ-004",
        summary="Invoice amount incorrect",
        description="My last invoice shows 150 EUR but I should have been charged 99 EUR according to my plan. Please correct this urgently.",
        reporter="billing.issue@company.com",
        created_at="2026-03-03T11:30:00Z",
        type="Support",
        ai_category="Billing",
        ai_priority="High",
        ai_summary="Invoice amount discrepancy: charged 150 EUR instead of contracted 99 EUR",
        ai_escalate=True,
        ai_confidence=0.92,
        ai_processed_at="2026-03-03T11:30:00Z"
    )
    assert classified_ticket_data.id == "PROJ-004"

def test_invalid_classified_ticket_confidence():
    with pytest.raises(ValidationError):
        ClassifiedTicket(
            id="PROJ-002",
            summary="Update delivery address",
            description="I would like to update my delivery address for my next order before it ships tomorrow.",
            reporter="jane.smith@company.com",
            created_at="2026-03-03T09:15:00Z",
            type="Support",
            ai_category="Access",
            ai_priority="High",
            ai_summary="User needs to update delivery address before order ships tomorrow",
            ai_escalate=True,
            ai_confidence=1.1,  # Invalid confidence value
            ai_processed_at="2026-03-03T09:15:00Z"
        )

def test_invalid_classified_ticket_category():
    with pytest.raises(ValidationError):
        ClassifiedTicket(
            id="PROJ-002",
            summary="Update delivery address",
            description="I would like to update my delivery address for my next order before it ships tomorrow.",
            reporter="jane.smith@company.com",
            created_at="2026-03-03T09:15:00Z",
            type="Support",
            ai_category="InvalidCategory",  # Invalid category
            ai_priority="High",
            ai_summary="User needs to update delivery address before order ships tomorrow",
            ai_escalate=True,
            ai_confidence=1,
            ai_processed_at="2026-03-03T09:15:00Z"
        )