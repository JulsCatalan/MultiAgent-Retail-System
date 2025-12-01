"""Process user query and return response with products and routing decision"""
import logging
from typing import List
from .router import route_query
from .retriever import search_products
from .generator import generate_response
from models import User, ConversationMessage
from kapso.client import KapsoClient

logger = logging.getLogger(__name__)


def get_conversation_messages(user: User, message_limit: int = 200) -> List[ConversationMessage]:
    """
    Obtiene el historial de conversación usando KapsoClient directamente.
    
    Args:
        user: Usuario con conversation_id
        message_limit: Número máximo de mensajes a obtener
        
    Returns:
        Lista de ConversationMessage ordenados por timestamp
    """
    if not user.conversation_id:
        logger.warning("⚠️ No hay conversation_id para el usuario %s", user.name)
        return []
    
    conversation_history = []
    
    try:
        with KapsoClient() as kapso:
            response = kapso.get_conversation_messages(
                conversation_id=user.conversation_id,
                page=1,
                per_page=message_limit
            )
            
            messages = response.get("data", [])
            
            if not isinstance(messages, list):
                logger.error("❌ Formato de mensajes inválido: %s", type(messages))
                return []

            for msg in messages:
                try:
                    # Extraer información del mensaje
                    direction = msg.get("direction", "")
                    # Mapear direction a sender: "inbound" = "client", "outbound" = "cedamoney"
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
                    
                    # Generar descripción del mensaje según su tipo
                    message_content = _format_message_content(msg, text, message_type, sender)
                    
                    if message_content and message_content.strip():
                        conversation_history.append(ConversationMessage(
                            timestamp=timestamp,
                            sender=sender,
                            message=message_content.strip(),
                            message_id=message_id
                        ))
                        
                except Exception as e:
                    logger.error("❌ Error procesando mensaje: %s", e)
                    continue
            
            # Ordenar por timestamp (más antiguos primero)
            conversation_history.sort(key=lambda x: x.timestamp or "")
            
    except Exception as e:
        logger.error("❌ Error obteniendo historial de Kapso: %s", e)
        
    
    logger.info("📝 Historial obtenido: %s", conversation_history)
        
    return conversation_history


def _format_message_content(msg: dict, text: str, message_type: str, sender: str) -> str:
    """
    Formatea el contenido del mensaje según su tipo.
    
    Args:
        msg: Diccionario con datos del mensaje
        text: Texto del mensaje
        message_type: Tipo de mensaje (text, image, etc.)
        sender: Sender del mensaje (client o cedamoney)
        
    Returns:
        Contenido formateado del mensaje
    """
    if message_type == "text":
        return text
    
    elif message_type == "image":
        caption = msg.get("caption", "")
        filename = msg.get("filename", "imagen")
        if sender == "cedamoney":
            return f"[Envío imagen: {caption or filename}]"
        else:
            return f"[Cliente envió una imagen{f': {caption}' if caption else ''}]"
    
    elif message_type == "reaction":
        emoji = msg.get("emoji", "👍")
        return f"[Reacción: {emoji}]"
    
    elif message_type == "sticker":
        return "[Sticker enviado]"
    
    else:
        if text:
            return text
        else:
            return f"[Mensaje tipo {message_type}]"


async def process_user_query(user: User, user_message: str) -> dict:
    """
    Process user query and return response with products and routing decision.
    
    Args:
        user: Usuario con conversation_id y metadata
        user_message: Mensaje del usuario
        
    Returns:
        Dict con response, products y routing_decision
    """
    # Obtener historial de conversación usando KapsoClient directamente
    conversation_context = get_conversation_messages(user)
    
    logger.info("📝 Historial obtenido: %d mensajes", len(conversation_context))
    
    routing = route_query(user_message)
    products = search_products(user_message)
    response = generate_response(user_message, products)
    
    return {
        "response": response,
        "products": products,
        "routing_decision": routing["decision"],
        "conversation_history": conversation_context
    }