# app/agents/query_builder.py
from openai import OpenAI
import os
import re
from typing import List, Optional, Dict, Any
from models import ConversationMessage

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extract_price_constraints(user_message: str) -> Dict[str, Any]:
    """
    Extrae restricciones de precio del mensaje del usuario.
    
    Ejemplos:
    - "productos de menos de 500" → max_price: 500
    - "productos de más de 1000" → min_price: 1000
    - "productos entre 200 y 500" → min_price: 200, max_price: 500
    - "productos baratos" → max_price: 500 (default para "barato")
    
    Returns:
        Dict con min_price y/o max_price si se detectan
    """
    constraints = {}
    msg_lower = user_message.lower()
    
    # Patrones para detectar restricciones de precio
    # "menos de X", "menor a X", "máximo X", "bajo X", "debajo de X"
    max_patterns = [
        r'menos de (\d+)',
        r'menor a (\d+)',
        r'menor de (\d+)',
        r'maximo (\d+)',
        r'máximo (\d+)',
        r'bajo (\d+)',
        r'debajo de (\d+)',
        r'no más de (\d+)',
        r'hasta (\d+)',
        r'por debajo de (\d+)',
    ]
    
    # "más de X", "mayor a X", "mínimo X", "arriba de X"
    min_patterns = [
        r'más de (\d+)',
        r'mas de (\d+)',
        r'mayor a (\d+)',
        r'mayor de (\d+)',
        r'mínimo (\d+)',
        r'minimo (\d+)',
        r'arriba de (\d+)',
        r'por encima de (\d+)',
        r'desde (\d+)',
    ]
    
    # "entre X y Y"
    range_patterns = [
        r'entre (\d+) y (\d+)',
        r'de (\d+) a (\d+)',
    ]
    
    # Buscar patrones de rango primero
    for pattern in range_patterns:
        match = re.search(pattern, msg_lower)
        if match:
            constraints['min_price'] = float(match.group(1))
            constraints['max_price'] = float(match.group(2))
            return constraints
    
    # Buscar patrones de máximo
    for pattern in max_patterns:
        match = re.search(pattern, msg_lower)
        if match:
            constraints['max_price'] = float(match.group(1))
            break
    
    # Buscar patrones de mínimo
    for pattern in min_patterns:
        match = re.search(pattern, msg_lower)
        if match:
            constraints['min_price'] = float(match.group(1))
            break
    
    # Palabras clave para precios
    if 'barato' in msg_lower or 'económico' in msg_lower or 'economico' in msg_lower:
        if 'max_price' not in constraints:
            constraints['max_price'] = 500.0  # Default para "barato"
    
    if 'caro' in msg_lower or 'premium' in msg_lower or 'exclusivo' in msg_lower:
        if 'min_price' not in constraints:
            constraints['min_price'] = 1000.0  # Default para "caro"
    
    return constraints

def build_search_query(
    user_message: str,
    conversation_context: Optional[List[ConversationMessage]] = None,
    user_preferences: Optional[List[dict]] = None,
    is_suggestion: bool = False
) -> str:
    """
    Query Builder agent: Analiza el contexto completo de la conversación y el mensaje actual
    para crear una query optimizada que busca SOLO UNA prenda a la vez.
    
    Prioriza los mensajes más recientes y extrae solo la información de la prenda actual
    que se está consultando. Si hay preferencias del usuario, las incorpora para mejorar
    la búsqueda.
    
    Args:
        user_message: Mensaje actual del usuario
        conversation_context: Historial completo de conversación
        user_preferences: Lista de preferencias del usuario guardadas
        is_suggestion: Si es True, el usuario solo mencionó preferencias sin buscar algo específico
        
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
{chr(10).join(weighted_messages)}"""
    
    # Formatear preferencias del usuario
    preferences_text = ""
    if user_preferences:
        from ..preferences import format_preferences_for_prompt
        preferences_text = format_preferences_for_prompt(user_preferences)
    
    preferences_section = ""
    if preferences_text:
        preferences_section = f"""

PREFERENCIAS DEL USUARIO (usa esto para mejorar la búsqueda):
{preferences_text}

IMPORTANTE: Incorpora estas preferencias en la query cuando sea relevante:
- Si el usuario es friolento → busca prendas abrigadas, suéteres, abrigos, manga larga
- Si tiene sobrepeso → busca prendas cómodas, holgadas, estilos que favorezcan
- Si prefiere estilo casual → enfócate en prendas casuales, no formales
- Si menciona ocasión (trabajo, fiesta) → busca prendas adecuadas para esa ocasión"""
    
    # Si es suggestion, usar un prompt completamente diferente
    if is_suggestion:
        prompt = f"""Eres un agente experto en moda que crea queries de búsqueda basadas en las PREFERENCIAS y NECESIDADES del usuario.

El usuario mencionó información sobre sus preferencias o necesidades personales, pero NO especificó exactamente qué producto busca. Tu trabajo es crear una query de búsqueda que encuentre productos adecuados para esas necesidades.

MENSAJE ACTUAL DEL USUARIO: "{user_message}"
{context_text}
{preferences_section}

INSTRUCCIONES ESPECÍFICAS PARA SUGERENCIAS:
1. El usuario NO está buscando un producto específico, solo mencionó sus preferencias/necesidades
2. Analiza las preferencias del usuario (tanto del mensaje actual como las guardadas) para entender qué tipo de productos necesita
3. Crea una query que busque productos que se ADAPTEN a esas necesidades:
   - Si es friolento → busca "prendas abrigadas", "suéteres", "abrigos", "manga larga"
   - Si tiene sobrepeso → busca "prendas cómodas", "holgadas", "talla grande"
   - Si prefiere estilo casual → busca "prendas casuales"
   - Si menciona ocasión (trabajo) → busca "prendas formales" o "para trabajo"
   - Si tiene alergias → evita esos materiales, busca alternativas
4. Si el mensaje actual menciona algo específico (ej: "camisa"), puedes incluirlo, pero prioriza las preferencias
5. Si hay múltiples preferencias, combínalas en la query (ej: "prendas abrigadas cómodas" si es friolento y tiene sobrepeso)

REGLAS CRÍTICAS:
- La query debe reflejar las NECESIDADES del usuario, no solo lo que mencionó
- Prioriza las preferencias guardadas sobre el mensaje actual si el mensaje es ambiguo
- Crea una query amplia pero relevante que capture la esencia de lo que el usuario necesita
- NO busques un producto específico, busca CATEGORÍAS o TIPOS de productos que se adapten

Responde con UNA frase corta que describa qué tipo de productos buscar.
Ejemplos de queries para sugerencias:
- "prendas abrigadas suéteres" (si es friolento)
- "prendas cómodas holgadas" (si tiene sobrepeso)
- "prendas casuales" (si prefiere estilo casual)
- "prendas formales para trabajo" (si menciona trabajo)
- "prendas abrigadas cómodas" (si es friolento Y tiene sobrepeso)
- "camisas abrigadas manga larga" (si menciona camisa y es friolento)

Si no hay preferencias claras, crea una query general basada en el mensaje actual."""
    else:
        # Prompt normal para búsquedas específicas
        rules_section = """
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
{preferences_section}
{rules_section}

INSTRUCCIONES:
1. Analiza el mensaje actual y los mensajes recientes (Peso ALTO) para identificar QUÉ prenda específica se está consultando AHORA
2. Extrae SOLO los atributos de ESA prenda específica:
   - Tipo de producto (camisa, pantalón, vestido, zapatos, etc.)
   - Color (si se menciona en el mensaje actual o mensajes recientes con alto peso)
   - Estilo/ocasión (casual, formal, deportivo, etc.)
   - Características especiales (manga larga, con bolsillos, etc.)
3. Si hay preferencias del usuario, INCORPÓRALAS en la query cuando sea relevante:
   - Si es friolento → agrega "abrigado", "manga larga", "suéter", "abrigo"
   - Si tiene sobrepeso → agrega "cómodo", "holgado"
   - Si prefiere estilo casual → agrega "casual"
4. IGNORA completamente cualquier prenda mencionada en mensajes antiguos (Peso MEDIO o BAJO) si no está relacionada con la consulta actual
5. Si el mensaje actual dice "verde", "esa prenda verde", "quiero ver la verde", busca SOLO prendas verdes, ignorando otras prendas mencionadas antes

REGLAS CRÍTICAS:
- SOLO extrae información de UNA prenda
- Prioriza información del mensaje actual y mensajes recientes (Peso ALTO)
- Si hay conflicto entre información antigua y reciente, usa SOLO la información reciente
- Si hay preferencias del usuario, úsalas para mejorar la query

Responde con UNA frase corta y directa que describa la prenda actual que se está buscando.
Ejemplos:
- "camisa formal manga larga color verde" (con preferencias: "camisa formal manga larga color verde abrigada" si es friolento)
- "pantalón casual color azul"
- "vestido color verde"
- "zapatos deportivos"

Si no puedes identificar una prenda específica, responde con el mensaje actual tal cual."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0  # Baja temperatura para respuestas más deterministas
    )
    
    optimized_query = response.choices[0].message.content.strip()
    print("🔍 Query optimizada construida: %s", optimized_query)
    
    return optimized_query

