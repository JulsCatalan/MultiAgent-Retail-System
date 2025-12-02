"""Funciones para manejar preferencias del usuario"""
from typing import List, Dict, Any, Optional
from .db import get_connection


def save_preference(
    conversation_id: str,
    preference_type: str,
    preference_value: str,
    confidence: float = 1.0,
    source_message: Optional[str] = None
) -> None:
    """
    Guarda o actualiza una preferencia del usuario.
    
    Args:
        conversation_id: ID de la conversación del usuario
        preference_type: Tipo de preferencia (ej: 'body_type', 'temperature_sensitivity', 'style')
        preference_value: Valor de la preferencia (ej: 'sobrepeso', 'friolento', 'casual')
        confidence: Confianza en la extracción (0.0 a 1.0)
        source_message: Mensaje original donde se mencionó la preferencia
    """
    conn = get_connection()
    cur = conn.cursor()
    
    # Intentar actualizar primero, si no existe, insertar
    cur.execute("""
        INSERT INTO user_preferences 
        (conversation_id, preference_type, preference_value, confidence, source_message, updated_at)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(conversation_id, preference_type) 
        DO UPDATE SET 
            preference_value = excluded.preference_value,
            confidence = excluded.confidence,
            source_message = excluded.source_message,
            updated_at = CURRENT_TIMESTAMP
    """, (conversation_id, preference_type, preference_value, confidence, source_message))
    
    conn.commit()
    conn.close()


def get_user_preferences(conversation_id: str) -> List[Dict[str, Any]]:
    """
    Obtiene todas las preferencias guardadas de un usuario.
    
    Args:
        conversation_id: ID de la conversación del usuario
        
    Returns:
        Lista de diccionarios con las preferencias del usuario
    """
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute("""
        SELECT preference_type, preference_value, confidence, source_message, updated_at
        FROM user_preferences
        WHERE conversation_id = ?
        ORDER BY updated_at DESC
    """, (conversation_id,))
    
    preferences = []
    for row in cur.fetchall():
        preferences.append({
            "preference_type": row[0],
            "preference_value": row[1],
            "confidence": row[2],
            "source_message": row[3],
            "updated_at": row[4]
        })
    
    conn.close()
    return preferences


def format_preferences_for_prompt(preferences: List[Dict[str, Any]]) -> str:
    """
    Formatea las preferencias para incluir en un prompt.
    
    Args:
        preferences: Lista de preferencias del usuario
        
    Returns:
        String formateado con las preferencias
    """
    if not preferences:
        return ""
    
    lines = []
    for pref in preferences:
        pref_type = pref["preference_type"]
        pref_value = pref["preference_value"]
        
        # Mapear tipos técnicos a descripciones naturales
        type_labels = {
            "body_type": "Tipo de cuerpo",
            "temperature_sensitivity": "Sensibilidad a la temperatura",
            "style_preference": "Estilo preferido",
            "size_preference": "Preferencia de talla",
            "color_preference": "Preferencia de color",
            "occasion": "Ocasión de uso",
            "budget": "Presupuesto",
            "allergies": "Alergias",
            "other": "Otra información"
        }
        
        label = type_labels.get(pref_type, pref_type.replace("_", " ").title())
        lines.append(f"- {label}: {pref_value}")
    
    return "\n".join(lines)


def delete_preference(conversation_id: str, preference_type: str) -> None:
    """
    Elimina una preferencia específica del usuario.
    
    Args:
        conversation_id: ID de la conversación del usuario
        preference_type: Tipo de preferencia a eliminar
    """
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute("""
        DELETE FROM user_preferences
        WHERE conversation_id = ? AND preference_type = ?
    """, (conversation_id, preference_type))
    
    conn.commit()
    conn.close()

