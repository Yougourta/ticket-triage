import asyncio

from src.file_handler import load_tickets, save_classified_tickets
from src.config import MODEL, MAX_TOKENS, INPUT_FILE, OUTPUT_FILE, LOG_LEVEL
from src.classifier import classify_ticket
from src.logger import logger

async def main():
    # Read the original tickets from the input file, classify them, and save the results to the output file
    tickets = load_tickets(INPUT_FILE)

    # Classify each ticket and save the results
    classification_tasks = [classify_ticket(ticket) for ticket in tickets]
    classified_tickets = await asyncio.gather(*classification_tasks)
    # Save the classified tickets to the output file
    save_classified_tickets(OUTPUT_FILE, classified_tickets)

    for i, classified_ticket in enumerate(classified_tickets, 1):
        logger.info(f"Ticket {i} with ID {classified_ticket['id']} is classified.")
        
    # Log the summary of the classification process
    logger.info(f"Processed {len(tickets)} tickets. Classified {len(classified_tickets)} tickets saved to '{OUTPUT_FILE}'.")
    logger.info(f"Classification completed with model '{MODEL}' and max tokens {MAX_TOKENS}.")
    logger.info(f"Escalate tickets: {sum(1 for t in classified_tickets if t.get('ai_escalate'))}")

if __name__ == "__main__":
    asyncio.run(main())