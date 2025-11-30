# app/embeddings.py
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_text(text: str) -> list:
    """
    Genera embedding usando OpenAI
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    
    return response.data[0].embedding