from .utils import normalize_kapso_webhook, extract_message_ids_from_webhook, mark_whatsapp_messages_as_read_batch
from .message_deduplicator import message_deduplicator
import logging
from models import User, UserMetadata
from app.agents.process_user_query import process_user_query
from .client import KapsoClient

logger = logging.getLogger(__name__)

async def use_kapso(webhook_data: dict):
    """
    Procesa webhooks de Kapso para manejar mensajes entrantes de WhatsApp
    """
    try:
        webhook_type = webhook_data.get("type", "unknown")
        
        logger.info("🔍 Webhook recibido de Kapso - Tipo: %s", webhook_type)
        
        if webhook_type == "whatsapp.message.received":
            data_list = normalize_kapso_webhook(webhook_data)
            
            if not data_list:
                logger.warning("⚠️ No se pudo normalizar el webhook o está vacío")
                return {
                    "status": "success",
                    "message": "Webhook sin datos válidos",
                    "processed": False,
                    "agent_response": False
                }
            
            message_ids = extract_message_ids_from_webhook(webhook_data)
            
            
            if message_ids and message_deduplicator.are_messages_already_processed(message_ids):
                logger.warning("⚠️ Mensaje DUPLICADO detectado y descartado: %s", message_ids)
                return {
                    "status": "success",
                    "message": "Webhook duplicado - mensajes ya procesados",
                    "processed": False,
                    "agent_response": False,
                    "duplicate": True,
                    "message_ids": message_ids
                }
            
            message_deduplicator.mark_messages_as_processed(message_ids)
            
            result = await handle_response(data_list=data_list)
            
            return result

    except Exception as e:
        logger.error("❌ Error procesando webhook de Kapso: %s: %s", type(e).__name__, str(e))
        import traceback
        logger.error("❌ Traceback: %s", traceback.format_exc())
        return {"status": "error", "message": "Error interno del servidor"}

async def handle_response(data_list: list) -> dict:
    """
    Maneja la respuesta del agente usando KapsoClient
    """
    first_data = data_list[0]
    conversation = first_data.get("conversation", {})
    reached_from_phone_number = first_data.get("whatsapp_config", {}).get("display_phone_number_normalized")
    
    # NO MOVER ESTO PUES ESTAMOS USANDO SANDBOX
    if (reached_from_phone_number is None):
        reached_from_phone_number = "56920403095"
        
    phone_number = conversation.get("phone_number", None)
    whatsapp_conversation_id = conversation.get("id", None)
    contact_name = conversation.get("contact_name", "Usuario")
    whatsapp_config_id = conversation.get("whatsapp_config_id")
    if not whatsapp_config_id:
        whatsapp_config_id = first_data.get("phone_number_id")
        
    if not whatsapp_conversation_id or not whatsapp_config_id:
        return {"status": "error", "message": "whatsapp_conversation_id es None"}
            


    # Combinar mensajes
    combined_message_parts = []
    message_ids = []
    
    for data in data_list:
        message_data = data.get("message", {})
        msg_id = message_data.get("id", None)
        message_ids.append(msg_id)
        
        message_type = message_data.get("message_type", "").lower()
        message_content_raw = message_data.get("content", "")
        message_content = message_content_raw.strip() if message_content_raw else ""
        
        if message_type in ("reaction", "sticker", "image"):
            continue
        
        if message_content:
            combined_message_parts.append(message_content)
        

    combined_message = " ".join(combined_message_parts)
    
    if not combined_message:
        return {"status": "success", "message": "Mensajes sin contenido omitidos"}
    
    try:
        await mark_whatsapp_messages_as_read_batch(message_ids, enable_typing_on_last=True, background_processing=False)
    except Exception as e:
        logger.error("❌ Error marcando mensajes como leídos: %s", e)
        
    user = User(
            name=contact_name,
            phone_number=phone_number,
            conversation_id=whatsapp_conversation_id,
            metadata=UserMetadata(whatsapp_config_id=whatsapp_config_id, reached_from_phone_number=reached_from_phone_number)
        )
    
    # Crear cliente de Kapso y pasarlo al agente para que envíe la respuesta
    try:
        with KapsoClient() as kapso_client:
            agent_response = await process_user_query(user, combined_message, kapso_client=kapso_client)
            
            
            
            return {
                "status": "success",
                "message": "Respuesta enviada",
                "agent_response": agent_response,
                "data": data_list
            }
    except Exception as e:
        logger.error("❌ Error ejecutando agente o enviando respuesta: %s", e)
        import traceback
        logger.error("Traceback: %s", traceback.format_exc())
        return {"status": "error", "message": "Error ejecutando agente o enviando respuesta"}
