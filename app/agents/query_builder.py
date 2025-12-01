# app/agents/query_builder.py
from openai import OpenAI
import os
from typing import List, Optional
from models import ConversationMessage

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def build_search_query(
    user_message: str,
    conversation_context: Optional[List[ConversationMessage]] = None
) -> str:
    """
    Query Builder agent: Analiza el contexto completo de la conversación y el mensaje actual
    para crear una query optimizada que busca SOLO UNA prenda a la vez.
    
    Prioriza los mensajes más recientes y extrae solo la información de la prenda actual
    que se está consultando.
    
    Args:
        user_message: Mensaje actual del usuario
        conversation_context: Historial completo de conversación
        
    Returns:
        Query optimizada para búsqueda de productos
    """
    
    # Construir contexto completo con pesos decrecientes
    context_text = ""
    if conversation_context:
        # Invertir para tener los más recientes primero
        recent_messages = list(reversed(conversation_context))
        
        total_messages = len(recent_messages)
        weighted_messages = []
        
        # Mensajes más recientes (últimos 15): peso alto
        recent_count = min(15, total_messages)
        for i, msg in enumerate(recent_messages[:recent_count]):
            # Peso alto para mensajes recientes (15, 14, 13, ...)
            weight = recent_count - i
            sender_label = "Cliente" if msg.sender == "client" else "Asistente"
            weighted_messages.append(f"[Peso ALTO: {weight}] {sender_label}: {msg.message}")
        
        # Mensajes anteriores (16-30): peso medio
        if total_messages > 15:
            medium_count = min(15, total_messages - 15)
            for i, msg in enumerate(recent_messages[15:15+medium_count]):
                # Peso medio para mensajes anteriores
                weight = medium_count - i
                sender_label = "Cliente" if msg.sender == "client" else "Asistente"
                weighted_messages.append(f"[Peso MEDIO: {weight}] {sender_label}: {msg.message}")
        
        # Mensajes muy antiguos (31+): resumen
        if total_messages > 30:
            old_messages = recent_messages[30:]
            weighted_messages.append(f"[Peso BAJO: contexto histórico] ... ({len(old_messages)} mensajes anteriores)")
        
        if weighted_messages:
            context_text = f"""

CONTEXTO COMPLETO DE CONVERSACIÓN (mensajes más recientes tienen MÁS peso):
{chr(10).join(weighted_messages)}

REGLAS CRÍTICAS PARA EXTRAER LA QUERY:
1. SOLO puedes extraer información de UNA prenda a la vez - esto es ABSOLUTAMENTE CRÍTICO
2. Los mensajes con "Peso ALTO" son los MÁS RECIENTES y tienen PRIORIDAD ABSOLUTA
3. Si en mensajes antiguos se habló de múltiples prendas (ej: camisa roja, pantalón azul, vestido negro) pero el mensaje actual o mensajes recientes (Peso ALTO) mencionan una prenda específica (ej: "verde", "esa prenda verde", "quiero ver la verde", "la camisa verde"), extrae SOLO los atributos de ESA prenda específica
4. Ejemplo: Si antes se habló de "camisa roja" y "pantalón azul", pero el mensaje actual dice "quiero ver la verde" o los últimos mensajes mencionan "prenda verde", la query debe ser SOLO sobre la prenda verde, ignorando completamente las otras mencionadas anteriormente
5. Si el mensaje actual menciona un color específico (ej: "verde"), ese es el color a buscar, incluso si antes se habló de otros colores
6. Si el mensaje actual menciona un tipo de prenda específico (ej: "camisa"), ese es el tipo a buscar, incluso si antes se habló de otros tipos"""
    
    prompt = f"""Eres un agente experto en moda que analiza conversaciones para extraer información de productos.

Tu trabajo es analizar el mensaje actual y el contexto completo de la conversación para crear una query de búsqueda optimizada.

MENSAJE ACTUAL: "{user_message}"
{context_text}

INSTRUCCIONES:
1. Analiza el mensaje actual y los mensajes recientes (Peso ALTO) para identificar QUÉ prenda específica se está consultando AHORA
2. Extrae SOLO los atributos de ESA prenda específica:
   - Tipo de producto (camisa, pantalón, vestido, zapatos, etc.)
   - Color (si se menciona en el mensaje actual o mensajes recientes con alto peso)
   - Estilo/ocasión (casual, formal, deportivo, etc.)
   - Características especiales (manga larga, con bolsillos, etc.)
3. IGNORA completamente cualquier prenda mencionada en mensajes antiguos (Peso MEDIO o BAJO) si no está relacionada con la consulta actual
4. Si el mensaje actual dice "verde", "esa prenda verde", "quiero ver la verde", busca SOLO prendas verdes, ignorando otras prendas mencionadas antes

REGLAS CRÍTICAS:
- SOLO extrae información de UNA prenda
- Prioriza información del mensaje actual y mensajes recientes (Peso ALTO)
- Si hay conflicto entre información antigua y reciente, usa SOLO la información reciente

Responde con UNA frase corta y directa que describa la prenda actual que se está buscando.
Ejemplos:
- "camisa formal manga larga color verde"
- "pantalón casual color azul"
- "vestido color verde"
- "zapatos deportivos"

Si no puedes identificar una prenda específica, responde con el mensaje actual tal cual."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.1  # Baja temperatura para respuestas más deterministas
    )
    
    optimized_query = response.choices[0].message.content.strip()
    print("🔍 Query optimizada construida: %s", optimized_query)
    
    return optimized_query

