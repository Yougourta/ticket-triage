import os

from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("MODEL", "claude-haiku-4-5-20251001")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
TEMPERATURE = int(os.getenv("TEMPERATURE", "0"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
INPUT_FILE = os.getenv("INPUT_FILE", "data/tickets.json")
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "data/classified_tickets.json")