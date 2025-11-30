# app/agents/router.py
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def route_query(user_message: str) -> dict:
    """
    Router agent: Determina si la consulta requiere búsqueda en la base de datos
    """
    
    prompt = f"""Eres un agente router para una tienda de ropa. 

Tu trabajo es clasificar la siguiente consulta del usuario:

"{user_message}"

Debes decidir:
- "search": Si el usuario busca productos específicos, recomienda ropa, pregunta por categorías, colores, precios, etc.
- "general": Si es un saludo, pregunta general sobre la tienda, o no requiere búsqueda de productos.

Responde SOLO con una palabra: "search" o "general"
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0
    )
    
    decision = response.choices[0].message.content.strip().lower()
    
    return {
        "decision": decision,
        "user_message": user_message
    }