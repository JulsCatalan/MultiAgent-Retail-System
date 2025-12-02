from .utils import normalize_kapso_webhook, extract_message_ids_from_webhook, mark_whatsapp_messages_as_read_batch
from .message_deduplicator import message_deduplicator
import logging
import asyncio
from models import User, UserMetadata
from app.agents.process_user_query import process_user_query
from .client import KapsoClient

logger = logging.getLogger(__name__)

async def use_kapso(webhook_data: dict):
    """
    Procesa webhooks de Kapso para manejar mensajes entrantes de WhatsApp
    """
    try:
        if not webhook_data or not isinstance(webhook_data, dict):
            print("❌ webhook_data inválido o None")
            return {"status": "error", "message": "webhook_data inválido"}
        
        webhook_type = webhook_data.get("type", "unknown")
        
        print("🔍 Webhook recibido de Kapso - Tipo: %s", webhook_type)
        
        if webhook_type == "whatsapp.message.received":
            data_list = normalize_kapso_webhook(webhook_data)
            
            if not data_list:
                print("⚠️ No se pudo normalizar el webhook o está vacío")
                return {
                    "status": "success",
                    "message": "Webhook sin datos válidos",
                    "processed": False,
                    "agent_response": False
                }
            
            message_ids = extract_message_ids_from_webhook(webhook_data)
            
            
            if message_ids and message_deduplicator.are_messages_already_processed(message_ids):
                print("⚠️ Mensaje DUPLICADO detectado y descartado: %s", message_ids)
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
        print("❌ Error procesando webhook de Kapso: %s: %s", type(e).__name__, str(e))
        
        
        return {"status": "error", "message": "Error interno del servidor"}

async def handle_response(data_list: list) -> dict:
    """
    Maneja la respuesta del agente usando KapsoClient
    """
    if not data_list or len(data_list) == 0:
        return {"status": "error", "message": "data_list está vacío"}
    
    first_data = data_list[0]
    if not first_data or not isinstance(first_data, dict):
        return {"status": "error", "message": "first_data inválido"}
    
    conversation = first_data.get("conversation") or {}
    if not isinstance(conversation, dict):
        conversation = {}
    
    whatsapp_config = first_data.get("whatsapp_config")
    if not isinstance(whatsapp_config, dict):
        whatsapp_config = {}
    
    reached_from_phone_number = whatsapp_config.get("display_phone_number_normalized")
    
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
        if not data or not isinstance(data, dict):
            continue
            
        message_data = data.get("message")
        if not isinstance(message_data, dict):
            message_data = {}
            
        msg_id = message_data.get("id", None)
        if msg_id:
            message_ids.append(msg_id)
        
        message_type = message_data.get("message_type", "").lower() if isinstance(message_data.get("message_type"), str) else ""
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
        print("❌ Error marcando mensajes como leídos: %s", e)
        
    user = User(
            name=contact_name,
            phone_number=phone_number,
            conversation_id=whatsapp_conversation_id,
            metadata=UserMetadata(whatsapp_config_id=whatsapp_config_id, reached_from_phone_number=reached_from_phone_number)
        )
    
    # --- FLUJO DE AGENTE CON CART INTEGRATION Y SOPORTE DE IMÁGENES ---
    try:
        from app.agents.cart_agent import handle_cart_interaction
        from app.cart import save_recent_products
        
        # Ejecutar en thread pool para no bloquear
        loop = asyncio.get_event_loop()
        
        def _handle_cart_and_agent():
            """Maneja cart agent y agente normal, luego envía respuesta con imágenes"""
            with KapsoClient() as kapso:
                # PASO 1: Verificar si es interacción de carrito
                cart_result = handle_cart_interaction(
                    conversation_id=whatsapp_conversation_id,
                    user_message=combined_message,
                    user_name=contact_name,
                    phone_number=phone_number
                )
                
                if cart_result.get("handled"):
                    logger.info("🛒 Cart agent manejó la solicitud")
                    
                    # Check if we need to send images
                    if cart_result.get("send_images"):
                        image_type = cart_result.get("image_type", "")
                        cart_items = cart_result.get("cart_items", [])
                        cart_total = cart_result.get("cart_total", 0.0)
                        
                        if image_type == "cart":
                            # Show cart with images
                            logger.info(f"📷 Enviando carrito con {len(cart_items)} imágenes")
                            kapso.send_cart_with_images(
                                whatsapp_conversation_id,
                                cart_items,
                                cart_total,
                                header_message=cart_result.get("response")
                            )
                        
                        elif image_type == "checkout":
                            # Show checkout summary with images
                            checkout_url = cart_result.get("checkout_url", "")
                            logger.info(f"📷 Enviando checkout con {len(cart_items)} imágenes")
                            kapso.send_checkout_with_images(
                                whatsapp_conversation_id,
                                cart_items,
                                cart_total,
                                checkout_url
                            )
                        
                        elif image_type == "added_to_cart":
                            # Show added product with image
                            added_product = cart_result.get("added_product", {})
                            logger.info(f"📷 Mostrando producto agregado con imagen")
                            kapso.send_message(whatsapp_conversation_id, cart_result.get("response", ""))
                            
                            if added_product:
                                # Build caption for added product
                                name = added_product.get("prod_name", "Producto")
                                color = added_product.get("colour_group_name", "")
                                price = added_product.get("price_mxn", 0)
                                image_url = added_product.get("image_url", "")
                                
                                caption = (
                                    f"*{name}* ({color})\n"
                                    f"💰 Precio: ${price:.2f} MXN\n\n"
                                    f"¿Qué deseas hacer?\n"
                                    f"• \"Ver carrito\" - Ver todos tus productos\n"
                                    f"• \"Seguir comprando\" - Buscar más\n"
                                    f"• \"Proceder al pago\" - Finalizar"
                                )
                                
                                if image_url and image_url.startswith("http"):
                                    kapso.send_image_message(whatsapp_conversation_id, image_url, caption)
                                else:
                                    kapso.send_message(whatsapp_conversation_id, caption)
                        
                        elif image_type == "search":
                            # Show search results with images
                            search_products = cart_result.get("search_products", [])
                            logger.info(f"📷 Enviando {len(search_products)} productos de búsqueda con imágenes")
                            kapso.send_products_with_images(
                                whatsapp_conversation_id,
                                search_products,
                                intro_message=cart_result.get("response"),
                                max_images=5
                            )
                            # Save as recent products
                            save_recent_products(whatsapp_conversation_id, search_products)
                    else:
                        # Just send text response
                        response_text = cart_result["response"]
                        kapso.send_message(whatsapp_conversation_id, response_text)
                    
                    return {
                        "cart_handled": True, 
                        "response": cart_result.get("response", ""),
                        "image_type": cart_result.get("image_type"),
                        "items_count": len(cart_result.get("cart_items", []))
                    }
                
                # PASO 2: Agente normal (Router → Retriever → Generator)
                logger.info("🤖 Usando agente normal")
                
                try:
                    # Ejecutar async function en nuevo loop
                    agent_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(agent_loop)
                    
                    result = agent_loop.run_until_complete(
                        process_user_query(user, combined_message, kapso_client=None)
                    )
                    
                    agent_loop.close()
                    
                    # Get products and agent response
                    agent_response = result.get("response", "")
                    products = result.get("products", [])
                    
                    # Decide how to send response based on products
                    if products and len(products) > 0:
                        # Send products WITH IMAGES
                        logger.info(f"📷 Enviando {min(len(products), 5)} productos con imágenes")
                        kapso.send_products_with_images(
                            whatsapp_conversation_id,
                            products,
                            intro_message=agent_response,
                            max_images=5  # WhatsApp best practice
                        )
                        
                        # Save products as recent for cart operations
                        logger.info(f"📦 Guardando {len(products)} productos recientes")
                        save_recent_products(whatsapp_conversation_id, products)
                    else:
                        # No products - just send text response
                        kapso.send_message(whatsapp_conversation_id, agent_response)
                    
                    return {
                        "cart_handled": False,
                        "response": agent_response,
                        "products_count": len(products),
                        "products_with_images": min(len(products), 5) if products else 0,
                        "routing_decision": result.get("routing_decision")
                    }
                    
                except Exception as agent_error:
                    logger.error(f"❌ Error en agente: {agent_error}")
                    import traceback
                    logger.error(traceback.format_exc())
                    
                    error_msg = "Disculpa, tuve un problema. ¿Podrías intentar de nuevo?"
                    kapso.send_message(whatsapp_conversation_id, error_msg)
                    
                    return {"cart_handled": False, "error": str(agent_error)}
        
        # Ejecutar flujo completo en executor
        agent_result = await loop.run_in_executor(None, _handle_cart_and_agent)
        
        logger.info("✅ Respuesta enviada a conversación %s", whatsapp_conversation_id)
        
        return {
            "status": "success",
            "message": "Respuesta enviada",
            "agent_result": agent_result,
            "data": data_list
        }
        
    except Exception as e:
        logger.error(f"❌ Error en flujo del agente: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"status": "error", "message": "Error ejecutando agente o enviando respuesta"}
