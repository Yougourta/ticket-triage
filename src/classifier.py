import json
import anthropic
import asyncio

from pydantic import ValidationError
from .config import MODEL, MAX_TOKENS, TEMPERATURE
from .models import OriginalTicket, ClassifiedTicket
from .logger import logger

# Call the AI agent to classify the ticket
async def call_ai_agent(model, max_tokens, temperature, system_prompt, original_ticket):
    # Initialize the Anthropic client and send the classification request
    client = anthropic.AsyncAnthropic()
    # Call the AI model to classify the ticket
    try:
        message = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
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
        You are a support ticket classification agent working for TotalEnergies Electricité & Gaz France as a CCaaS tools specialist (Odigo, Ring Central, SightCall, etc.). 
        Your task is to analyze incoming support tickets and classify them based on their content. 
        Each ticket comes in a JSON format with the following fields: id, summary, description, reporter, created_at, and type.
        Based on the information provided in the ticket, you need to determine the appropriate category (Access, Billing, Technical, or Other), assign a priority level (High, Medium, Low), and provide a one-sentence summary of the issue.
        Additionally, you should determine whether the ticket needs to be escalated to a human agent based on the urgency and potential impact of the issue. 
        Finally, you should provide a confidence score for your classification, indicating how certain you are about the assigned category and priority. Set ai_confidence based on how clearly the ticket matches a category:
            - 0.9+ : unambiguous, clear category and priority
            - 0.7-0.9 : likely classification but some ambiguity
            - below 0.7 : set "ai_escalate" to true regardless of content
        Your response should be a valid JSON object containing the following fields: ai_category, ai_priority, ai_summary, ai_escalate, ai_confidence, and ai_processed_at. 
        Ensure that your response strictly adheres to this format and does not include any additional text or markdown.
        
        Example of the input ticket:
        {
            "id": "PROJ-001",
            "summary": "Cannot login to customer portal",
            "description": "Since this morning I cannot access my account. Password reset email never received. I have a payment due tonight.",
            "reporter": "john.doe@company.com",
            "created_at": "2026-03-03T08:30:00Z",
            "type": "Bug"
        }
        Example of the expected output:
        {
            "ai_category": "Access",
            "ai_priority": "High",
            "ai_summary": "User unable to login; password reset email not received, payment due tonight",
            "ai_escalate": true,
            "ai_confidence": 0.87,
            "ai_processed_at": "2026-03-03T08:30:00Z"
        }

        Here is the required format for your response:

        Format:
        {
            "ai_category": "Access|Billing|Technical|Other",
            "ai_priority": "High|Medium|Low",
            "ai_summary": "one sentence between 10 and 200 characters",
            "ai_escalate": true|false,
            "ai_confidence": 0.0 to 1.0,
            "ai_processed_at": "ISO 8601 timestamp"
        }
    """

    try:
        # Validate and parse the tickets using Pydantic
        original_ticket = OriginalTicket(**ticket)

        # Call the AI model to classify the ticket
        classified_ticket = await call_ai_agent(MODEL, MAX_TOKENS, TEMPERATURE, system_prompt, original_ticket)
        try:
            # Parse the AI response and combine it with the original ticket data
            classified_ticket = json.loads(classified_ticket)
            classified_ticket["id"] = original_ticket.id
            classified_ticket["summary"] = original_ticket.summary
            classified_ticket["description"] = original_ticket.description
            classified_ticket["reporter"] = original_ticket.reporter
            classified_ticket["created_at"] = original_ticket.created_at
            classified_ticket["type"] = original_ticket.type
            # If confidence is below 0.7, set escalate to true
            if classified_ticket["ai_confidence"] < 0.7:
                classified_ticket["ai_escalate"] = True
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

