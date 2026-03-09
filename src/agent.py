import json
import asyncio
from typing import TypedDict
from pydantic import ValidationError
from langgraph.graph import StateGraph, START, END
from src.models import OriginalTicket, ClassifiedTicket
from src.classifier import call_ai_agent
from src.config import MODEL, MAX_TOKENS, TEMPERATURE
from src.logger import logger

class TicketState(TypedDict):
    ticket: dict
    original: OriginalTicket
    result: dict | None
    error: str | None

def validate_ticket(state: TicketState) -> TicketState:
    try:
        original = OriginalTicket(**state['ticket'])
        return {"original": original, "error": None}
    except ValidationError as e:
        return {"error": str(e)}
    
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
    classified_ticket = await call_ai_agent(MODEL, MAX_TOKENS, TEMPERATURE, system_prompt, state['original'])
    if classified_ticket:
        return {"result": json.loads(classified_ticket), "error": None}
    else:
        return {"result": None, "error": "Failed to classify ticket with AI agent."}
    
def check_confidence(state: TicketState) -> TicketState:
    classified_ticket = state["ticket"] | state["result"]
    if classified_ticket["ai_confidence"] < 0.7:
        classified_ticket["ai_escalate"] = True
    return {"result": classified_ticket, "error": None}

def save_result(state: TicketState) -> TicketState:
    try:
        result = ClassifiedTicket(**state["result"])
        return {"result": result.model_dump(mode="json"), "error": None}
    except ValidationError as e:
        return {"result": None, "error": str(e)}
    
def build_graph():
    graph = StateGraph(TicketState)

    graph.add_node("validate_ticket", validate_ticket)
    graph.add_node("classify_with_claude", classify_with_claude)
    graph.add_node("check_confidence", check_confidence)
    graph.add_node("save_result", save_result)

    graph.add_edge(START, "validate_ticket")
    graph.add_edge("validate_ticket", "classify_with_claude")
    graph.add_edge("classify_with_claude", "check_confidence")
    graph.add_edge("check_confidence", "save_result")

    return graph.compile()

async def run_agent(ticket: dict) -> dict:
    graph = build_graph()
    result = await graph.ainvoke({"ticket": ticket, "original": None, "result": None, "error": None})
    return result["result"]