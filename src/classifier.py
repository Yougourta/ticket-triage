import json
import anthropic
import asyncio

from pydantic import ValidationError
from .config import MODEL, MAX_TOKENS
from .models import OriginalTicket, ClassifiedTicket
from .logger import logger

# Call the AI agent to classify the ticket
async def call_ai_agent(model, max_tokens, system_prompt, original_ticket):
    # Initialize the Anthropic client and send the classification request
    client = anthropic.AsyncAnthropic()
    # Call the AI model to classify the ticket
    try:
        message = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": original_ticket.model_dump_json()
                },
                {
                    "role": "assistant",
                    "content": "{"
                }
            ]
        )
        return "{" + message.content[0].text
    except Exception as e:   
        logger.error(f"Error calling AI agent: {e}")
        return "{}"
    
# Validate the JIRA tickets
async def classify_ticket(ticket) -> OriginalTicket:
    system_prompt = """
        You are a support ticket classification agent.
        Analyze the ticket and return ONLY a valid JSON object with no markdown.

        Format:
        {
            "ai_category": "Access|Billing|Technical|Other",
            "ai_priority": "High|Medium|Low",
            "ai_summary": "one sentence between 10 and 200 characters",
            "ai_escalate": true|false,
            "ai_confidence": 0.0 to 1.0,
            "ai_processed_at": "ISO 8601 timestamp"
        }

        Escalate if the user mentions urgency, deadline, financial loss, or threat.
        Set confidence based on how clearly the ticket matches a category.
    """

    try:
        # Validate and parse the tickets using Pydantic
        original_ticket = OriginalTicket(**ticket)

        # Call the AI model to classify the ticket
        classified_ticket = await call_ai_agent(MODEL, MAX_TOKENS, system_prompt, original_ticket)
        try:
            # Parse the AI response and combine it with the original ticket data
            classified_ticket = json.loads(classified_ticket)
            classified_ticket["id"] = original_ticket.id
            classified_ticket["summary"] = original_ticket.summary
            classified_ticket["description"] = original_ticket.description
            classified_ticket["reporter"] = original_ticket.reporter
            classified_ticket["created_at"] = original_ticket.created_at
            classified_ticket["type"] = original_ticket.type
            # Validate the classification result and return a structured ticket
            return ClassifiedTicket(**classified_ticket).model_dump(mode="json")
        except ValidationError as e:
            logger.error(f"Error validating classification: {e}")
            logger.error(f"Raw classification data: {classified_ticket}")
            return {
                "id": original_ticket.id,
                "summary": original_ticket.summary,
                "description": original_ticket.description,
                "reporter": original_ticket.reporter,
                "created_at": original_ticket.created_at,
                "type": "Other",
                "ai_category": "Other",
                "ai_priority": "Low",
                "ai_summary": "Unable to classify ticket",
                "ai_escalate": False,
                "ai_confidence": 0.0,
                "ai_processed_at": None
            }
    except ValidationError as e:
        logger.error(f"Error validating ticket: {e}")
        logger.error(f"Raw ticket data: {ticket}")
        return dict | None

