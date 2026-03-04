import json

from typing import List, Dict, Any
from .logger import logger
from .models import ClassifiedTicket

def load_tickets(file_path: str) -> List[Dict[str, Any]]:
    """
    Read a JSON file and return a list of ticket objects.
    
    Args:
        file_path: Path to the JSON file (e.g., 'data/tickets.json')
    
    Returns:
        List of ticket dictionaries
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            tickets = json.load(file)
            logger.info(f"Loaded {len(tickets) if isinstance(tickets, list) else 1} tickets from '{file_path}'")
            return tickets if isinstance(tickets, list) else [tickets]
    except FileNotFoundError:
        logger.error(f"Error: File '{file_path}' not found.")
        return []
    except json.JSONDecodeError:
        logger.error(f"Error: Invalid JSON format in '{file_path}'.")
        return []
    
def save_classified_tickets(file_path: str, tickets: List[dict[str, Any]]) -> None:
    """
    Save a list of classified ticket objects to a JSON file.
    
    Args:
        file_path: Path to the output JSON file (e.g., 'data/classified_tickets.json')
        tickets: List of classified ticket dictionaries to save
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(tickets, file, indent=4)
            logger.info(f"Saved {len(tickets)} classified tickets to '{file_path}'")
    except IOError as e:
        logger.error(f"Error writing to file '{file_path}': {e}")