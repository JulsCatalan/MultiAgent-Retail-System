"""Process user query and return response with products and routing decision"""
import logging
from typing import List, Optional
from .router import route_query
from .query_builder import build_search_query
from .retriever import search_products
from .generator import generate_response
from .cart_agent import handle_cart_interaction
from ..cart import save_recent_products
from models import User, ConversationMessage
from kapso.client import KapsoClient

logger = logging.getLogger(__name__)


def get_conversation_messages(
    user: User, 
    message_limit: int = 200,
    kapso_client: Optional[KapsoClient] = None
) -> List[ConversationMessage]:
    """
    Obtiene el historial de conversación usando KapsoClient.
    
    Args:
        user: Usuario con conversation_id
        message_limit: Número máximo de mensajes a obtener
        kapso_client: Cliente de Kapso opcional. Si se proporciona, se usa este cliente.
                     Si no, se crea uno nuevo.
        
    Returns:
        Lista de ConversationMessage ordenados por timestamp
    """
    if not user.conversation_id:
        print("⚠️ No hay conversation_id para el usuario %s", user.name)
        return []
    
    conversation_history = []
    should_close_client = False
    
    try:
        # Usar el cliente proporcionado o crear uno nuevo
        if kapso_client is None:
            kapso_client = KapsoClient()
            should_close_client = True
        
        response = kapso_client.get_conversation_messages(
            conversation_id=user.conversation_id,
            page=1,
            per_page=message_limit
        )
        
        messages = response.get("data", [])
        
        if not isinstance(messages, list):
            print("❌ Formato de mensajes inválido: %s", type(messages))
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
                print("❌ Error procesando mensaje: %s", e)
                continue
        
        # Ordenar por timestamp (más antiguos primero)
        conversation_history.sort(key=lambda x: x.timestamp or "")
        
    except Exception as e:
        print("❌ Error obteniendo historial de Kapso: %s", e)
    finally:
        # Cerrar el cliente solo si lo creamos nosotros
        if should_close_client and kapso_client is not None:
            try:
                kapso_client.close()
            except Exception as e:
                print("⚠️ Error cerrando cliente de Kapso: %s", e)
    
    print("📝 Historial obtenido: %d mensajes", len(conversation_history))
        
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


async def process_user_query(
    user: User, 
    user_message: str, 
    kapso_client: Optional[KapsoClient] = None
) -> dict:
    """
    Process user query and return response with products and routing decision.
    
    Args:
        user: Usuario con conversation_id y metadata
        user_message: Mensaje del usuario
        kapso_client: Cliente de Kapso opcional. Si se proporciona, se enviará
                     la respuesta del agente a través de Kapso.
        
    Returns:
        Dict con response, products y routing_decision
    """
    # Obtener historial de conversación usando el cliente proporcionado (si existe)
    conversation_context = get_conversation_messages(user, kapso_client=kapso_client)

    # 1) Revisar si el mensaje es sobre el carrito (ver carrito, agregar algo, etc.)
    if user.conversation_id:
        cart_result = handle_cart_interaction(
            conversation_id=user.conversation_id,
            user_message=user_message,
        )
    else:
        cart_result = {"handled": False}

    if cart_result.get("handled"):
        response = cart_result.get("response", "")
        products = cart_result.get("products", [])
        routing_decision = "general"

        # Enviar mensaje a través de Kapso si el cliente está disponible
        message_sent = False
        if kapso_client is not None and user.conversation_id:
            try:
                print("📤 Enviando respuesta a conversación %s", user.conversation_id)
                kapso_client.send_message(user.conversation_id, response)
                message_sent = True
                print("✅ Mensaje enviado exitosamente")
            except Exception as e:
                print("❌ Error enviando mensaje por Kapso: %s", e)
        else:
            if kapso_client is None:
                print("ℹ️ No se proporcionó cliente de Kapso, mensaje no enviado")
            elif not user.conversation_id:
                print("⚠️ Usuario sin conversation_id, mensaje no enviado")

        return {
            "response": response,
            "products": products,
            "routing_decision": routing_decision,
            "conversation_history": conversation_context,
            "message_sent": message_sent,
        }
    
    # 2) Pasar el contexto al router para que tome una mejor decisión
    routing = route_query(user_message, conversation_context=conversation_context)
    print("📝 Routing: %s", routing)
    
    # Si el routing es "general", generar respuesta sin buscar productos
    if routing["decision"] == "general":
        products = []
        response = generate_response(
            user_message=user_message,
            products=products,
            conversation_context=conversation_context,
            routing_decision=routing["decision"],
        )
    else:
        # Si es "search", primero construir la query optimizada usando todo el contexto
        optimized_query = build_search_query(
            user_message, conversation_context=conversation_context
        )

        # Luego buscar productos usando la query optimizada
        products = search_products(optimized_query)

        # Guardar productos recientes para poder referenciarlos como "Producto 1", etc.
        if user.conversation_id:
            try:
                save_recent_products(user.conversation_id, products)
            except Exception as e:
                print("⚠️ Error guardando productos recientes: %s", e)

        response = generate_response(
            user_message=user_message,
            products=products,
            conversation_context=conversation_context,
            routing_decision=routing["decision"],
        )
    
    # Enviar mensaje a través de Kapso si el cliente está disponible
    message_sent = False
    if kapso_client is not None and user.conversation_id:
        try:
            print("📤 Enviando respuesta a conversación %s", user.conversation_id)
            kapso_client.send_message(user.conversation_id, response)
            message_sent = True
            print("✅ Mensaje enviado exitosamente")
        except Exception as e:
            print("❌ Error enviando mensaje por Kapso: %s", e)
    else:
        if kapso_client is None:
            print("ℹ️ No se proporcionó cliente de Kapso, mensaje no enviado")
        elif not user.conversation_id:
            print("⚠️ Usuario sin conversation_id, mensaje no enviado")
    
    return {
        "response": response,
        "products": products,
        "routing_decision": routing["decision"],
        "conversation_history": conversation_context,
        "message_sent": message_sent
    }