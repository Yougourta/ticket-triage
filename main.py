from src.file_handler import load_tickets, save_classified_tickets
from src.config import MODEL, MAX_TOKENS, INPUT_FILE, OUTPUT_FILE, LOG_LEVEL
from src.classifier import classify_ticket

# Read the original tickets from the input file, classify them, and save the results to the output file
tickets = load_tickets(INPUT_FILE)

# Classify each ticket and save the results
classified_tickets = []
for index, ticket in enumerate(tickets, start=1):
    classified_ticket = classify_ticket(ticket)
    if classified_ticket:
        classified_tickets.append(classified_ticket)

save_classified_tickets(OUTPUT_FILE, classified_tickets)
# Log the summary of the classification process
print(f"Processed {len(tickets)} tickets. Classified {len(classified_tickets)} tickets saved to '{OUTPUT_FILE}'.")
print(f"Classification completed with model '{MODEL}' and max tokens {MAX_TOKENS}.")
print(f"Escalate tickets: {sum(1 for t in classified_tickets if t.get('ai_escalate'))}")