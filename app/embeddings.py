# app/embeddings.py - Probar modelo más grande
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_text(text: str) -> list:
    """
    Genera embedding usando modelo más potente
    """
    response = client.embeddings.create(
        model="text-embedding-3-large",  # ← Más preciso que "small"
        input=text,
        dimensions=1536  # Opcional: reducir dimensiones para velocidad
    )
    
    return response.data[0].embedding