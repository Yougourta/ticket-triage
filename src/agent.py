import json
import asyncio
from typing import TypedDict
from pydantic import ValidationError
from langgraph.graph import StateGraph, START, END
from src.models import OriginalTicket, ClassifiedTicket, ProcessedTicket
from src.classifier import call_ai_agent
from src.config import MODEL, MAX_TOKENS, TEMPERATURE
from src.logger import logger

class TicketState(TypedDict):
    ticket: dict
    original: OriginalTicket
    processed: ProcessedTicket
    result: dict | None
    error: str | None

def validate_ticket(state: TicketState) -> TicketState:
    try:
        original = OriginalTicket(**state['ticket'])
        return {"original": original, "error": None}
    except ValidationError as e:
        return {"error": str(e)}

def validate_processed_ticket(state: TicketState) -> TicketState:
    try:
        processed_ticket = ProcessedTicket(**state["result"])
        return {"processed": processed_ticket, "error": None}
    except ValidationError as e:
        return {"result": None, "error": str(e)}
    
async def classify_with_claude(state: TicketState) -> TicketState:
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
        Always explain the reasoning behind your classification and prioritization decisions.
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
            "ai_processed_at": "2026-03-03T08:30:00Z",
            "ai_reasoning": "The ticket describes a clear access issue with a high urgency due to the upcoming payment deadline, leading to a high confidence score. However, since the confidence is below 0.9, escalation is recommended to ensure timely resolution."
        }

        Here is the required format for your response:

        Format:
        {
            "ai_category": "Access|Billing|Technical|Other",
            "ai_priority": "High|Medium|Low",
            "ai_summary": "one sentence between 10 and 200 characters",
            "ai_escalate": true|false,
            "ai_confidence": 0.0 to 1.0,
            "ai_processed_at": "ISO 8601 timestamp",
            "ai_reasoning": "explanation of the reasoning behind the classification and prioritization decisions"
        }
    """
    classified_ticket = await call_ai_agent(MODEL, MAX_TOKENS, TEMPERATURE, system_prompt, state['ticket'])
    if classified_ticket:
        return {"result": json.loads(classified_ticket), "error": None}
    else:
        return {"result": None, "error": "Failed to classify ticket with AI agent."}
    
def check_confidence(state: TicketState) -> TicketState:
    classified_ticket = state["ticket"] | state["result"]
    if classified_ticket["ai_confidence"] < 0.7:
        classified_ticket["ai_escalate"] = True
    return {"result": classified_ticket, "error": None}

async def generate_response(state: TicketState) -> TicketState:
    system_prompt = """
    You are a support ticket classification agent working for TotalEnergies Electricité & Gaz France as a CCaaS tools specialist (Odigo, Ring Central, SightCall, etc.). 
    Your task is to analyze the ticket information and its classification results to generate a professional response that can be sent to the customer.
    Additionally, you should provide a recommended action for the human agent who will handle the ticket after escalation (if applicable).
    Example of the input ticket:
    {
        "id": "PROJ-001",
        "summary": "Cannot login to customer portal",
        "description": "Since this morning I cannot access my account. Password reset email never received. I have a payment due tonight.",
        "reporter": "john.doe@company.com",
        "created_at": "2026-03-03T08:30:00Z",
        "type": "Bug",
        "ai_category": "Access",
        "ai_priority": "High",
        "ai_summary": "User unable to login; password reset email not received, payment due tonight",
        "ai_escalate": true,
        "ai_confidence": 0.87,
        "ai_processed_at": "2026-03-03T08:30:00Z",
        "ai_reasoning": "The ticket describes a clear access issue with a high urgency due to the upcoming payment deadline, leading to a high confidence score. However, since the confidence is below 0.9, escalation is recommended to ensure timely resolution."
    }
    Example of the expected output:
    {
        "id": "PROJ-001",
        "summary": "Cannot login to customer portal",
        "description": "Since this morning I cannot access my account. Password reset email never received. I have a payment due tonight.",
        "reporter": "john.doe@company.com",
        "created_at": "2026-03-03T08:30:00Z",
        "type": "Bug",
        "ai_category": "Access",
        "ai_priority": "High",
        "ai_summary": "User unable to login; password reset email not received, payment due tonight",
        "ai_escalate": true,
        "ai_confidence": 0.87,
        "ai_processed_at": "2026-03-03T08:30:00Z",
        "ai_reasoning": "The ticket describes a clear access issue with a high urgency due to the upcoming payment deadline, leading to a high confidence score. However, since the confidence is below 0.9, escalation is recommended to ensure timely resolution.",
        "ai_suggested_response": "Dear John Doe, we understand that you are experiencing issues logging into the customer portal and have not received the password reset email. We apologize for the inconvenience. Our team is currently investigating the issue and will work to resolve it as quickly as possible. In the meantime, please ensure to check your spam folder for the password reset email. We will keep you updated on the progress and aim to have this resolved before your payment deadline tonight.",
        "ai_recommended_action": "Escalader vers le département logistique/gestion des commandes. Cette demande ne relève pas du domaine des outils CCaaS (Odigo, Ring Central, SightCall) mais plutôt de la gestion des commandes et de la logistique. L'agent humain doit vérifier le statut de la commande et mettre à jour l'adresse de livraison avant l'expédition prévue demain. Priorité : HAUTE en raison du délai critique (expédition demain)."
    }
    The generated response MUST be PROFESSIONAL and writtend ONLY in FRENCH. The response should address the customer's issue, acknowledge the urgency, and provide clear next steps or assurances. The response should also reflect the classification results, such as the identified category and priority level, to ensure it is tailored to the specific situation described in the ticket.
    """
    processed_ticket = await call_ai_agent(MODEL, MAX_TOKENS, TEMPERATURE, system_prompt, state["ticket"] | state["result"])
    try:
        return {"result": json.loads(processed_ticket), "error": None}
    except ValidationError as e:
            logger.error(f"Error validating classification: {e}")
            logger.error(f"Raw classification data: {processed_ticket}")
            return {"result": None, "error": str(e)}

def save_result(state: TicketState) -> TicketState:
    try:
        processed_ticket = state["processed"]
        return {"result": processed_ticket.model_dump(mode="json"), "error": None}
    except ValidationError as e:
        return {"result": None, "error": str(e)}
    
def build_graph():
    graph = StateGraph(TicketState)

    graph.add_node("validate_ticket", validate_ticket)
    graph.add_node("classify_with_claude", classify_with_claude)
    graph.add_node("check_confidence", check_confidence)
    graph.add_node("generate_response", generate_response)
    graph.add_node("validate_processed_ticket", validate_processed_ticket)
    graph.add_node("save_result", save_result)

    graph.add_edge(START, "validate_ticket")
    graph.add_edge("validate_ticket", "classify_with_claude")
    graph.add_edge("classify_with_claude", "check_confidence")
    graph.add_edge("check_confidence", "generate_response")
    graph.add_edge("generate_response", "validate_processed_ticket")
    graph.add_edge("validate_processed_ticket", "save_result")

    return graph.compile()

async def run_agent(ticket: dict) -> dict:
    graph = build_graph()
    result = await graph.ainvoke({"ticket": ticket, "original": None, "processed": None, "result": None, "error": None})
    return result["result"]