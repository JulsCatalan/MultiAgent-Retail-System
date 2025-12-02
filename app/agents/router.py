# app/agents/router.py
from openai import OpenAI
import os
from typing import List, Optional
from models import ConversationMessage

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def route_query(
    user_message: str, 
    conversation_context: Optional[List[ConversationMessage]] = None
) -> dict:
    """
    Router agent: Determina si la consulta requiere búsqueda en la base de datos.
    
    Args:
        user_message: Mensaje actual del usuario
        conversation_context: Historial completo de conversación (los últimos mensajes tienen más peso)
    """
    
    # Construir contexto de conversación completo con más peso en los últimos mensajes
    context_text = ""
    if conversation_context:
        # Invertir para tener los más recientes primero
        recent_messages = list(reversed(conversation_context))
        
        # Incluir más mensajes para tener contexto completo, pero con pesos decrecientes
        # Los últimos 20 mensajes tienen peso, pero los últimos 10 tienen mucho más peso
        total_messages = len(recent_messages)
        weighted_messages = []
        
        # Mensajes más recientes (últimos 10): peso alto
        recent_count = min(10, total_messages)
        for i, msg in enumerate(recent_messages[:recent_count]):
            # Peso alto para mensajes recientes (10, 9, 8, ...)
            weight = recent_count - i
            sender_label = "Cliente" if msg.sender == "client" else "Asistente"
            weighted_messages.append(f"[Peso ALTO: {weight}] {sender_label}: {msg.message}")
        
        # Mensajes anteriores (11-20): peso medio
        if total_messages > 10:
            medium_count = min(10, total_messages - 10)
            for i, msg in enumerate(recent_messages[10:10+medium_count]):
                # Peso medio para mensajes anteriores (5, 4, 3, ...)
                weight = medium_count - i
                sender_label = "Cliente" if msg.sender == "client" else "Asistente"
                weighted_messages.append(f"[Peso MEDIO: {weight}] {sender_label}: {msg.message}")
        
        # Mensajes muy antiguos (21+): peso bajo pero incluidos para contexto
        if total_messages > 20:
            old_messages = recent_messages[20:]
            # Resumir mensajes muy antiguos
            sender_label = "Cliente" if old_messages[0].sender == "client" else "Asistente"
            weighted_messages.append(f"[Peso BAJO: contexto histórico] ... ({len(old_messages)} mensajes anteriores)")
        
        if weighted_messages:
            context_text = f"""

CONTEXTO COMPLETO DE CONVERSACIÓN (mensajes más recientes tienen MÁS peso):
{chr(10).join(weighted_messages)}

REGLAS CRÍTICAS DE PRIORIZACIÓN:
1. SOLO se puede hacer UNA consulta de producto a la vez
2. Los mensajes con "Peso ALTO" son los MÁS RECIENTES y tienen PRIORIDAD ABSOLUTA
3. Si en mensajes antiguos se habló de múltiples prendas (ej: camisa roja, pantalón azul) pero el mensaje actual o mensajes recientes (Peso ALTO) mencionan una prenda específica (ej: "verde", "esa prenda verde", "quiero ver la verde"), la consulta actual es SOLO sobre ESA prenda específica
4. Ejemplo: Si antes se habló de "camisa roja" y "pantalón azul", pero el mensaje actual dice "quiero ver la verde" o los últimos mensajes mencionan "prenda verde", la consulta es SOLO sobre la prenda verde, ignorando las otras mencionadas anteriormente"""
    
    prompt = f"""Eres un agente router para una tienda de ropa. 

Tu trabajo es clasificar la siguiente consulta del usuario:

MENSAJE ACTUAL: "{user_message}"
{context_text}

REGLAS DE DECISIÓN:
1. SOLO se puede hacer UNA consulta de producto a la vez - esto es CRÍTICO
2. Si el mensaje actual o los mensajes recientes (Peso ALTO) mencionan una prenda específica (color, tipo, etc.), esa es la consulta actual, ignorando otras prendas mencionadas anteriormente
3. Los mensajes más recientes tienen MÁS peso que los antiguos

Debes decidir entre TRES opciones:
- "cart": Si el usuario quiere interactuar con su carrito de compras (ver carrito, agregar producto al carrito, quitar del carrito, etc.). Ejemplos: "muéstrame mi carrito", "agrega el producto 1 al carrito", "quiero agregar el suéter blanco", "qué tengo en el carrito"
- "search": Si el usuario busca productos específicos, recomienda ropa, pregunta por categorías, colores, precios, tallas, etc. Incluso si menciona "verde", "esa prenda", "quiero ver la verde" en el mensaje actual o reciente PERO NO menciona carrito, es "search"
- "general": Si es un saludo, pregunta general sobre la tienda, agradecimiento, o no requiere búsqueda de productos específicos ni interacción con carrito

IMPORTANTE: Si el usuario menciona "carrito", "carro", "agregar al carrito", "ver carrito", etc., debe ser "cart". Si solo describe productos o hace preguntas sobre productos SIN mencionar carrito, es "search".

Responde SOLO con una palabra: "cart", "search" o "general"
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0
    )
    
    decision = response.choices[0].message.content.strip().lower()
    
    # Validar que la decisión sea una de las opciones válidas
    if decision not in ["cart", "search", "general"]:
        # Fallback: si no es válido, intentar inferir
        if "carrito" in user_message.lower() or "carro" in user_message.lower():
            decision = "cart"
        else:
            decision = "search"  # Por defecto, asumir búsqueda
    
    return {
        "decision": decision,
        "user_message": user_message
    }