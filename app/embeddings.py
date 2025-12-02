# app/embeddings.py
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_text(text: str) -> list:
    """
    Usa 1536 dimensiones - sweet spot de rendimiento/precisión
    
    Beneficios:
    - 2x más rápido que 3072
    - Solo ~1-2% menos preciso
    - Usa la mitad de almacenamiento
    - Perfecto para e-commerce con miles de productos
    """
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text,
        dimensions=1536  # ← RECOMENDADO
    )
    
    return response.data[0].embedding