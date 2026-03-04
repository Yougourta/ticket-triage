import logging
from logging.handlers import RotatingFileHandler
from config import LOG_LEVEL

logger = logging.getLogger(__name__)
logger

file_handler = RotatingFileHandler('ticket_triage.log', mode='a', encoding='utf-8', maxBytes=1024*1024, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.ERROR)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(LOG_LEVEL)

logger.addHandler(file_handler)
logger.addHandler(console_handler)