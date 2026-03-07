import tiktoken

from src.file_handler import load_tickets
from src.logger import logger

encoding = tiktoken.get_encoding("cl100k_base")

nb_tokens = 0
tickets = load_tickets("data/tickets.json")
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
for ticket in tickets:
    nb_ticket_tokens = len(encoding.encode(ticket["summary"] + " " + ticket["description"] + " " + system_prompt))
    nb_tokens += nb_ticket_tokens
    logger.info(f"Ticket ID: {ticket['id']}, NB Tokens: {nb_ticket_tokens}")
logger.info(f"Total NB Tokens: {nb_tokens}")
