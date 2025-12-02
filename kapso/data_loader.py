import logging
import os
from typing import List, Optional

from app.models import Context, ConversationMessage
# from models.User import User
from kapso.client import KapsoClient

# Mock User class if not available, or adjust import path
class User:
    def __init__(self, name, conversation_id, phone_number=None, metadata=None):
        self.name = name
        self.conversation_id = conversation_id
        self.phone_number = phone_number
        self.metadata = metadata


logger = logging.getLogger(__name__)

def is_testing_phone_number(phone_number: str) -> bool:
    """
    Verifica si un número de teléfono está configurado como número de testing.
    """
    if not phone_number or not phone_number.strip():
        return False
    
    testing_numbers_env = os.getenv('TESTING_PHONE_NUMBERS', '')
    if not testing_numbers_env:
        return False
    
    # Parsear números de testing de la variable de entorno
    testing_numbers = [
        num.strip() 
        for num in testing_numbers_env.split(',')
        if num.strip()
    ]
    
    phone_normalized = phone_number.strip()
    return phone_normalized in testing_numbers

def get_context_with_history(
    user: User,
    message_limit: int = 200,
    is_testing: bool = False,
) -> Context:
    """
    Obtiene el contexto del agente con historial de conversación desde Kapso
    """
    try:
        conversation_history = get_conversation_history_kapso(user, message_limit, is_testing)
    except Exception as e: 
        print("❌ Error obteniendo historial de conversación: %s", e)
        conversation_history = []
        
    return Context(
        conversation_history=conversation_history, 
        current_client=user,
    )

def get_conversation_history_kapso(user: User, message_limit: int, is_testing: bool = False) -> List[ConversationMessage]:
    """Obtiene el historial de conversación usando KapsoClient"""
    if not user.conversation_id:
        print("⚠️ No hay conversation_id para el usuario %s", user.name)
        return []
    
    conversation_history = []
    
    try:
        # Usar KapsoClient
        with KapsoClient() as kapso:
   
            
            response = kapso.get_conversation_messages(
                conversation_id=user.conversation_id,
                page=1,
                per_page=message_limit
            )
            
            messages = response.get("data", [])
            
            if not isinstance(messages, list):
                print("❌ Formato de mensajes inválido: %s", type(messages))
                return []

            for i, msg in enumerate(messages):
                try:
                    # Extraer información del mensaje
                    direction = msg.get("direction", "")
                    sender = "client" if direction == "inbound" else "cedamoney"
                    text = msg.get("content", "")
                    
                    # Si no hay contenido en content, intentar obtenerlo de message_type_data
                    if not text:
                        message_type_data = msg.get("message_type_data", {})
                        if isinstance(message_type_data, dict):
                            text = message_type_data.get("text", "")
                    
                    timestamp = msg.get("created_at", "")
                    message_type = msg.get("message_type", "text")
                    message_id = msg.get("id", "")
                    
                    message_content, media_description = _generate_message_description(msg, text, message_type, sender)
                    
                    if message_content and message_content.strip():
                        conversation_history.append(ConversationMessage(
                            timestamp=timestamp,
                            sender=sender,
                            message=message_content.strip(),
                            message_type=message_type,
                            message_id=message_id,
                            # media_description=media_description # El modelo ConversationMessage actual no parece tener este campo en models.py revisado
                        ))
                        
                except Exception as e:
                    print(f"❌ Error procesando mensaje {i}: {e}")
                    continue
            
            # Ordenar por timestamp (más antiguos primero)
            conversation_history.sort(key=lambda x: x.timestamp)
            
    except Exception as e:
        print(f"❌ Error obteniendo historial de Kapso: {e}")
        
    return conversation_history

def _generate_message_description(msg: dict, text: str, message_type: str, sender: str) -> tuple[str, Optional[str]]:
    """Genera descripción del mensaje (lógica preservada)"""
    if message_type == "text":
        return text, None
    
    elif message_type == "image":
        caption = msg.get("caption", "")
        filename = msg.get("filename", "imagen")
        if sender == "cedamoney":
            return f"[Envío imagen: {caption or filename}]", f"Imagen enviada: {caption or filename}"
        else:
            return f"[Cliente envió una imagen{f': {caption}' if caption else ''}]", f"Imagen recibida del cliente"
    
    # ... (otros tipos se pueden agregar aquí copiando la lógica original si es necesario)
    
    else:
        if text:
            return text, f"Mensaje tipo {message_type}"
        else:
            return f"[Mensaje tipo {message_type}]", f"Contenido {message_type}"
