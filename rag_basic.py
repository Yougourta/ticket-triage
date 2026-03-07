import numpy
import anthropic
from openai import OpenAI
from src.file_handler import load_tickets
from src.logger import logger
from src.config import MODEL, MAX_TOKENS, TEMPERATURE

def call_ai_agent(model, max_tokens, temperature, system_prompt, query):
    # Initialize the Anthropic client and send the classification request
    client = anthropic.Anthropic()
    # Call the AI model to classify the ticket
    try:
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": query
                },
                {
                    "role": "assistant",
                    "content": ""
                }
            ]
        )
        return message.content[0].text
    except Exception as e:   
        logger.error(f"Error calling AI agent: {e}")
        return "{}"

# Load tickets
tickets = load_tickets("data/tickets.json")
tickets_embeddings = []
# Generate embeddings for each ticket
client = OpenAI()
for ticket in tickets:
    embedding = client.embeddings.create(input=ticket["summary"]+" "+ticket["description"], model="text-embedding-ada-002")
    tickets_embeddings.append((ticket, embedding.data[0].embedding))
# Example retrieval query
retrieval_query = "Are there any login or access issues ?"
query_embedding = client.embeddings.create(input=retrieval_query, model="text-embedding-ada-002").data[0].embedding
# Calculate cosine similarity between the query and each ticket embedding
similarity_scores = []
for ticket, embedding in tickets_embeddings:
    similarity = numpy.dot(query_embedding, embedding) / (numpy.linalg.norm(query_embedding) * numpy.linalg.norm(embedding))
    similarity_scores.append((ticket, similarity))
# Sort tickets by similarity score
similarity_scores.sort(key=lambda x: x[1], reverse=True)
similarity_scores = similarity_scores[:2]  # Get top 2 most similar tickets
logger.info("Top 2 similar tickets:")
tickets_found = ""
for ticket, score in similarity_scores:
    logger.info(f"Ticket ID: {ticket['id']}, Similarity Score: {score:.4f}")
    tickets_found += 'Ticket '+ticket['id']+': '+ticket['summary']+' - '+ticket['description'] + "\n"

llm_query = f"""
Question: "Are there any login or access issues ?"

Relevant tickets found:
{tickets_found}

Based on these tickets, answer the question.
"""
# Call the LLM to answer the question based on the retrieved tickets
message = call_ai_agent(MODEL, MAX_TOKENS, TEMPERATURE, "", llm_query)
logger.info(f"LLM Result: {message}")