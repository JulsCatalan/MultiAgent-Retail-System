"""Agente que extrae preferencias del usuario del contexto de conversación"""
import json
import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from models import ConversationMessage
from ..preferences import save_preference

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extract_user_preferences(
    user_message: str,
    conversation_context: Optional[List[ConversationMessage]] = None
) -> List[Dict[str, Any]]:
    """
    Extrae preferencias del usuario del mensaje actual y contexto de conversación.
    
    Args:
        user_message: Mensaje actual del usuario
        conversation_context: Historial de conversación
        
    Returns:
        Lista de preferencias extraídas con formato:
        [{
            "preference_type": "body_type",
            "preference_value": "sobrepeso",
            "confidence": 0.9,
            "source_message": "tengo sobrepeso"
        }, ...]
    """
    # Formatear contexto de conversación
    context_text = ""
    if conversation_context:
        recent_messages = list(reversed(conversation_context))[:10]  # Últimos 10 mensajes
        context_lines = []
        for msg in recent_messages:
            sender_label = "Cliente" if msg.sender == "client" else "Asistente"
            context_lines.append(f"{sender_label}: {msg.message}")
        context_text = "\n".join(context_lines)
    
    context_section = ""
    if context_text:
        context_section = f"""

CONTEXTO DE CONVERSACIÓN RECIENTE:
{context_text}"""
    
    prompt = f"""Eres un agente experto que extrae información personal y preferencias del usuario en conversaciones de una tienda de ropa.

MENSAJE ACTUAL DEL USUARIO: "{user_message}"
{context_section}

Tu tarea es identificar información personal relevante que pueda ayudar a hacer recomendaciones de productos más personalizadas.

TIPOS DE PREFERENCIAS QUE DEBES BUSCAR:
1. body_type: Tipo de cuerpo (ej: "sobrepeso", "delgado", "atlético", "grande", "pequeño")
2. temperature_sensitivity: Sensibilidad a la temperatura (ej: "friolento", "caluroso", "siento mucho frío", "me da mucho calor")
3. style_preference: Estilo preferido (ej: "casual", "formal", "deportivo", "elegante", "moderno")
4. size_preference: Preferencia de talla (ej: "talla grande", "talla pequeña", "talla extra grande")
5. color_preference: Preferencia de color (ej: "me gustan los colores oscuros", "prefiero colores claros")
6. occasion: Ocasión de uso (ej: "para trabajo", "para fiestas", "para hacer ejercicio")
7. budget: Presupuesto mencionado (ej: "barato", "económico", "hasta 500 pesos")
8. allergies: Alergias o sensibilidades (ej: "alérgico a la lana", "piel sensible")
9. other: Cualquier otra información relevante sobre necesidades o preferencias

REGLAS:
- Solo extrae información EXPLÍCITA o CLARAMENTE INFERIDA del mensaje o contexto
- NO inventes información que no esté presente
- Si el usuario menciona algo como "tengo sobrepeso", extrae: {{"preference_type": "body_type", "preference_value": "sobrepeso"}}
- Si el usuario dice "soy muy friolento", extrae: {{"preference_type": "temperature_sensitivity", "preference_value": "friolento"}}
- Asigna un confidence entre 0.0 y 1.0 basado en qué tan explícita es la información
- Si no encuentras ninguna preferencia relevante, devuelve una lista vacía

Responde SOLO con un JSON válido, sin texto adicional:
{{
  "preferences": [
    {{
      "preference_type": "<tipo>",
      "preference_value": "<valor>",
      "confidence": <número entre 0.0 y 1.0>,
      "source_message": "<mensaje donde se mencionó>"
    }}
  ]
}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.1
    )
    
    content = response.choices[0].message.content.strip()
    
    try:
        data = json.loads(content)
        preferences = data.get("preferences", [])
        
        # Validar y normalizar preferencias
        valid_preferences = []
        for pref in preferences:
            if all(key in pref for key in ["preference_type", "preference_value"]):
                valid_preferences.append({
                    "preference_type": pref["preference_type"],
                    "preference_value": pref["preference_value"],
                    "confidence": float(pref.get("confidence", 0.8)),
                    "source_message": pref.get("source_message", user_message)
                })
        
        return valid_preferences
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        print(f"⚠️ Error extrayendo preferencias: {e}")
        return []


def process_and_save_preferences(
    conversation_id: str,
    user_message: str,
    conversation_context: Optional[List[ConversationMessage]] = None
) -> List[Dict[str, Any]]:
    """
    Extrae preferencias del usuario y las guarda en la base de datos.
    
    Args:
        conversation_id: ID de la conversación del usuario
        user_message: Mensaje actual del usuario
        conversation_context: Historial de conversación
        
    Returns:
        Lista de preferencias extraídas y guardadas
    """
    if not conversation_id:
        return []
    
    # Extraer preferencias
    preferences = extract_user_preferences(user_message, conversation_context)
    
    # Guardar en la base de datos
    for pref in preferences:
        try:
            save_preference(
                conversation_id=conversation_id,
                preference_type=pref["preference_type"],
                preference_value=pref["preference_value"],
                confidence=pref["confidence"],
                source_message=pref["source_message"]
            )
        except Exception as e:
            print(f"⚠️ Error guardando preferencia: {e}")
            continue
    
    if preferences:
        print(f"✅ Preferencias extraídas y guardadas: {len(preferences)}")
    
    return preferences

