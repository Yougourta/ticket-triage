from src.file_handler import load_tickets, save_classified_tickets
from src.config import MODEL, MAX_TOKENS, INPUT_FILE, OUTPUT_FILE, LOG_LEVEL
from src.classifier import classify_ticket
from src.logger import logger

# Read the original tickets from the input file, classify them, and save the results to the output file
tickets = load_tickets(INPUT_FILE)

# Classify each ticket and save the results
classified_tickets = []
for index, ticket in enumerate(tickets, start=1):
    classified_ticket = classify_ticket(ticket)
    if classified_ticket:
        classified_tickets.append(classified_ticket)

save_classified_tickets(OUTPUT_FILE, classified_tickets)