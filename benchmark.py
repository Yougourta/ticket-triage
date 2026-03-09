from src.file_handler import load_tickets
from src.logger import logger

# Load Claude classified tickets
claude_tickets = load_tickets("output/classified_tickets.json")
# Load Mistral classified tickets
mistral_tickets = load_tickets("output/mistral_classified_tickets.json")

# Compare the classifications and calculate accuracy
def calculate_accuracy(claude_tickets, mistral_tickets):
    global_score = 0
    for claude_ticket, mistral_ticket in zip(claude_tickets, mistral_tickets):
        logger.info(f"{claude_ticket['id']}:")
        logger.info(f"    Claude → {claude_ticket['ai_category']} / {claude_ticket['ai_priority']} / {claude_ticket['ai_escalate']}")
        logger.info(f"    Mistral → {mistral_ticket['ai_category']} / {mistral_ticket['ai_priority']} / {mistral_ticket['ai_escalate']}")
        logger.info(f"    Match → " + ("✓ category " if claude_ticket['ai_category'] == mistral_ticket['ai_category'] else "✗ category") + " " + ("✓ priority " if claude_ticket['ai_priority'] == mistral_ticket['ai_priority'] else "✗ priority") + " " + ("✓ escalate " if claude_ticket['ai_escalate'] == mistral_ticket['ai_escalate'] else "✗ escalate") + "\n")
        global_score += 1 if claude_ticket['ai_category'] == mistral_ticket['ai_category'] and claude_ticket['ai_priority'] == mistral_ticket['ai_priority'] and claude_ticket['ai_escalate'] == mistral_ticket['ai_escalate'] else 0
    logger.info(f"Global Accuracy: {global_score / len(claude_tickets) * 100:.2f}%")
calculate_accuracy(claude_tickets, mistral_tickets)
