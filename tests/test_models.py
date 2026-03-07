import pytest

from pydantic import ValidationError
from src.models import OriginalTicket

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