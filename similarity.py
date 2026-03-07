import numpy as np

from openai import OpenAI

from src.file_handler import load_tickets
from src.logger import logger

def generate_embeddings(input_text):
    client = OpenAI()
    result = client.embeddings.create(
        input=input_text,
        model="text-embedding-ada-002"
    )
    return result.data[0].embedding

tickets = load_tickets("data/tickets.json")
embeddings = []
for ticket in tickets:
    embedding = generate_embeddings(ticket["summary"] + " " + ticket["description"])
    embeddings.append(embedding)
    logger.info(f"Ticket ID: {ticket['id']}, Embedding generated")

# Calculate cosine similarity between the first ticket and all others
similarity_scores = []
for i, embedding in enumerate(embeddings[1:], start=1):
    similarity = np.dot(embeddings[0], embedding) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embedding))
    similarity_scores.append((i, similarity))

similarity_scores.sort(key=lambda x: x[1], reverse=True)
for similarity_score in similarity_scores:
    logger.info(f"Similarity between Ticket {tickets[0]['id']} and Ticket {tickets[similarity_score[0]]['id']}: {similarity_score[1]:.4f}")