# app/embeddings.py
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_text(text: str):
    print("📥 Texto recibido para embedding:", text)

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    embedding = response.data[0].embedding

    # Log seguro y compacto
    print("📏 Longitud del embedding:", len(embedding))
    print("🔍 Primeros 10 valores:", embedding[:10])

    return embedding
