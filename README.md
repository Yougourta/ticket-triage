# Ticket Triage CLI

## Overview
Triage CLI takes a "tickets.json" of JIRA tickets, sends them to an AI agent for classification. The agent applies the appropriate priority and the correct category. It then stores the output file of the classified tickets inside a "classified_tickets.json" file.

## Architecture
ticket-triage/
├── .env
├── .gitignore
├── requirements.txt
├── main.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── logger.py
│   ├── models.py
│   ├── classifier.py
│   └── file_handler.py
├── data/
│   └── tickets.json
└── output/
    └── classified_tickets.json

## Setup
1. Update the .env file with your Anthropic API_KEY
2. Run the following command : python3 -m pip install -r requirements.txt

## Usage
1. Load the "data/ticket.json" file with your tickets.
2. Run the command : python main.py
3. Open the "output/classified_tickets.json" file

## Tech Stack
Available inside the "requirements.txt" file.