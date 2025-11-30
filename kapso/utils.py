"""
Parser y normalizador de webhooks de Kapso.

Este módulo maneja diferentes formatos de webhooks de Kapso y los normaliza
a una estructura común para su procesamiento.
"""

import asyncio
import logging
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


def normalize_kapso_webhook(webhook_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Normaliza el webhook de Kapso a una estructura común.
    
    Soporta dos formatos:
    1. Formato antiguo: data[].message, data[].conversation, etc. en estructura plana
    2. Formato nuevo (batch): data[].message con campos anidados, data[].conversation, data[].whatsapp_config
    3. Formato raw/CloudAPI: data[].message con campos como 'type', 'text.body', 'kapso', etc.
    
    Args:
        webhook_data: Datos del webhook de Kapso
        
    Returns:
        Lista de mensajes normalizados con estructura común:
        [
            {
                "message": {
                    "id": "...",
                    "message_type": "text",
                    "content": "...",
                    ...
                },
                "conversation": {...},
                "whatsapp_config": {...},
                "is_new_conversation": bool
            },
            ...
        ]
    """
    if not webhook_data:
        logger.warning("⚠️ webhook_data vacío")
        return []
    
    data_list = webhook_data.get("data", [])
    if not data_list:
        logger.warning("⚠️ No hay 'data' en webhook_data")
        return []
    
    normalized_list = []
    
    for item in data_list:
        message_data = item.get("message", {})
        if not isinstance(message_data, dict):
            message_data = {}
            
        conversation_data = item.get("conversation", {})
        whatsapp_config_data = item.get("whatsapp_config", {})
        is_new_conversation = item.get("is_new_conversation", False)
        
        # --- Normalización del mensaje ---
        normalized_message = message_data.copy()
        
        # 1. Normalizar 'message_type'
        # Si falta 'message_type' pero existe 'type', usarlo (formato raw)
        if "message_type" not in normalized_message and "type" in normalized_message:
            normalized_message["message_type"] = normalized_message["type"]
            
        # 2. Normalizar 'content'
        # Si falta 'content', intentar extraerlo según el tipo
        if "content" not in normalized_message:
            msg_type = normalized_message.get("message_type", "")
            
            if msg_type == "text":
                # En formato raw, el texto está en message['text']['body']
                text_data = normalized_message.get("text", {})
                if isinstance(text_data, dict):
                    normalized_message["content"] = text_data.get("body", "")
            
            # TODO: Agregar soporte para otros tipos si es necesario (image, etc.)
            # Por ahora nos enfocamos en texto que es lo crítico
            
        # --- Fin normalización mensaje ---
        
        # --- Normalización de IDs y Config ---
        # Intentar extraer phone_number_id del item raíz o de conversation
        phone_number_id = item.get("phone_number_id") or conversation_data.get("phone_number_id")
        
        # Si conversation data tiene 'kapso' metadata (formato raw), a veces el ID de conv está ahí o es el del item
        if "id" not in conversation_data and "conversation_id" in item.get("batch_info", {}):
            conversation_data["id"] = item.get("batch_info", {}).get("conversation_id")
             
        # Asegurar whatsapp_config
        if not whatsapp_config_data:
            whatsapp_config_data = {}
             
        # Si falta display_phone_number_normalized, usar phone_number_id como fallback
        if "display_phone_number_normalized" not in whatsapp_config_data and phone_number_id:
            whatsapp_config_data["display_phone_number_normalized"] = str(phone_number_id)
             
        # Si falta whatsapp_config_id en conversation, intentar usar phone_number_id como fallback
        if "whatsapp_config_id" not in conversation_data:
            if phone_number_id:
                conversation_data["whatsapp_config_id"] = phone_number_id
            elif whatsapp_config_data.get("id"):
                conversation_data["whatsapp_config_id"] = whatsapp_config_data.get("id")
                 
        # Si falta reached_from_phone_number en whatsapp_config, intentar obtenerlo de conversation.phone_number_id
        # (aunque phone_number_id no es el número en sí, es un ID, pero es lo mejor que tenemos si falta lo otro)
        # pipeline.py lo usará para el config.
        
        normalized_item = {
            "message": normalized_message,
            "conversation": conversation_data,
            "whatsapp_config": whatsapp_config_data,
            "is_new_conversation": is_new_conversation,
            "phone_number_id": phone_number_id # Extra field for pipeline fallback
        }
        
        normalized_list.append(normalized_item)
    
    logger.info("✅ Webhook normalizado: %d mensaje(s) procesado(s)", len(normalized_list))
    return normalized_list


def extract_message_ids_from_webhook(webhook_data: Dict[str, Any]) -> List[str]:
    """
    Extrae los IDs de mensajes del webhook (soporta ambos formatos).
    
    Args:
        webhook_data: Datos del webhook de Kapso
        
    Returns:
        Lista de IDs de mensajes
    """
    message_ids = []
    data_list = webhook_data.get("data", [])
    
    for item in data_list:
        message_data = item.get("message", {})
        if isinstance(message_data, dict):
            msg_id = message_data.get("id")
            if msg_id:
                message_ids.append(msg_id)
    
    return message_ids


# Thread pool para procesamiento en background
_thread_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="kapso_read_marker")

from .client import KapsoClient

async def mark_whatsapp_message_as_read_single(
    message_id: str, 
    typing_indicator: bool = True
) -> Dict[str, Any]:
    """
    Marca un mensaje individual de WhatsApp como leído usando la API de Kapso
    """
    try:
        # Usar KapsoClient de forma síncrona en un ejecutor para no bloquear el loop
        # Ya que KapsoClient actual es síncrono (usa httpx.Client, no httpx.AsyncClient)
        loop = asyncio.get_event_loop()
        
        def _sync_mark():
            with KapsoClient() as client:
                return client.mark_as_read(message_id, typing_indicator)
        
        response_data = await loop.run_in_executor(_thread_pool, _sync_mark)
        
        if "error" not in response_data:
            logger.debug("✅ Mensaje %s marcado como leído exitosamente", message_id)
            return {
                "success": True,
                "message_id": message_id,
                "status_code": 200,
                "data": response_data
            }
        else:
            logger.warning("⚠️ Error marcando mensaje %s como leído: %s", message_id, response_data.get("status_code"))
            return {
                "success": False,
                "message_id": message_id,
                "status_code": response_data.get("status_code"),
                "error": response_data,
                "message": f"Error de API de Kapso"
            }
                    
    except Exception as e:
        logger.error("❌ Error marcando mensaje %s como leído: %s", message_id, e)
        return {
            "success": False,
            "message_id": message_id,
            "error": str(e),
            "message": "Error interno marcando mensaje como leído"
        }

async def mark_whatsapp_messages_as_read_batch(
    message_ids: List[str], 
    enable_typing_on_last: bool = True,
    background_processing: bool = True
) -> Dict[str, Any]:
    """
    Marca múltiples mensajes de WhatsApp como leídos
    
    Args:
        message_ids (List[str]): Lista de IDs de mensajes a marcar como leídos
        enable_typing_on_last (bool): Si activar typing indicator en el último mensaje
        background_processing (bool): Si procesar en background para mayor velocidad
        
    Returns:
        Dict[str, Any]: Resultado consolidado de la operación
    """
    if not message_ids:
        return {
            "success": True,
            "message": "No hay mensajes para marcar como leídos",
            "results": []
        }
    
    # Filtrar IDs válidos
    valid_message_ids = [msg_id for msg_id in message_ids if msg_id and msg_id.strip()]
    
    if not valid_message_ids:
        logger.warning("⚠️ Todos los message_ids están vacíos o son inválidos")
        return {
            "success": False,
            "message": "Todos los message_ids son inválidos",
            "results": []
        }
    
    logger.info("📖 Marcando %d mensajes como leídos (background: %s)", len(valid_message_ids), background_processing)
    
    try:
        if background_processing:
            # Procesamiento en background - no esperamos los resultados
            loop = asyncio.get_event_loop()
            loop.run_in_executor(
                _thread_pool,
                lambda: asyncio.run(_mark_messages_background(valid_message_ids, enable_typing_on_last))
            )
            
            return {
                "success": True,
                "message": f"Procesando {len(valid_message_ids)} mensajes en background",
                "message_count": len(valid_message_ids),
                "background": True,
                "typing_enabled_on_last": enable_typing_on_last
            }
        else:
            # Procesamiento síncrono - esperamos resultados
            results = await _mark_messages_foreground(valid_message_ids, enable_typing_on_last)
            
            success_count = sum(1 for r in results if r.get("success"))
            
            return {
                "success": success_count > 0,
                "message": f"Marcados {success_count}/{len(valid_message_ids)} mensajes como leídos",
                "message_count": len(valid_message_ids),
                "success_count": success_count,
                "background": False,
                "results": results
            }
            
    except Exception as e:
        logger.error("❌ Error en batch marking: %s", e)
        return {
            "success": False,
            "message": f"Error procesando mensajes: {str(e)}",
            "message_count": len(valid_message_ids)
        }

async def _mark_messages_background(message_ids: List[str], enable_typing_on_last: bool):
    """Función helper para procesar mensajes en background"""
    try:
        results = await _mark_messages_foreground(message_ids, enable_typing_on_last)
        success_count = sum(1 for r in results if r.get("success"))
        logger.info("🔄 Background processing completado: %d/%d mensajes marcados", success_count, len(message_ids))
    except Exception as e:
        logger.error("❌ Error en background processing: %s", e)

async def _mark_messages_foreground(message_ids: List[str], enable_typing_on_last: bool) -> List[Dict[str, Any]]:
    """Función helper para procesar mensajes en foreground"""
    tasks = []
    
    # Crear tareas para todos los mensajes
    for i, message_id in enumerate(message_ids):
        # Solo el último mensaje debe tener typing indicator si está habilitado
        is_last_message = (i == len(message_ids) - 1)
        typing_indicator = enable_typing_on_last and is_last_message
        
        task = mark_whatsapp_message_as_read_single(message_id, typing_indicator)
        tasks.append(task)
    
    # Ejecutar todas las tareas en paralelo
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Procesar resultados y manejar excepciones
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append({
                "success": False,
                "message_id": message_ids[i],
                "error": str(result),
                "message": "Excepción durante el procesamiento"
            })
        else:
            processed_results.append(result)
    
    return processed_results

async def disable_typing_indicator(conversation_id: str) -> Dict[str, Any]:
    """
    Desactiva el typing indicator para una conversación usando KapsoClient
    """
    try:
        # Usar KapsoClient de forma síncrona en un ejecutor
        loop = asyncio.get_event_loop()
        
        def _sync_disable_typing():
            with KapsoClient() as client:
                return client.disable_typing_indicator(conversation_id)

        response_data = await loop.run_in_executor(_thread_pool, _sync_disable_typing)
        
        if response_data.get("success"):
            logger.debug("✅ Typing indicator desactivado para conversación %s", conversation_id)
            return {
                "success": True,
                "conversation_id": conversation_id,
                "message": "Typing indicator desactivado"
            }
        else:
            logger.debug("⚠️ No se pudo desactivar typing indicator: %s", response_data.get("status_code"))
            return {
                "success": False,
                "conversation_id": conversation_id,
                "message": "Endpoint de typing no disponible o falló",
                "status_code": response_data.get("status_code")
            }
                    
    except Exception as e:
        logger.debug("⚠️ Error desactivando typing indicator: %s", e)
        return {
            "success": False,
            "conversation_id": conversation_id,
            "error": str(e),
            "message": "Error desactivando typing indicator"
        }

# Función de conveniencia para mantener compatibilidad
def mark_whatsapp_messages_as_read(message_ids: List[str]):
    """
    Función de conveniencia para marcar mensajes como leídos (mantiene compatibilidad)
    Procesa en background por defecto para no bloquear el flujo principal
    
    Args:
        message_ids (List[str]): Lista de IDs de mensajes
    """
    if not message_ids:
        return
    
    # Ejecutar en background de forma fire-and-forget
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Si ya hay un loop corriendo, usar run_in_executor
            loop.run_in_executor(
                _thread_pool,
                lambda: asyncio.run(mark_whatsapp_messages_as_read_batch(
                    message_ids, 
                    enable_typing_on_last=True, 
                    background_processing=True
                ))
            )
        else:
            # Si no hay loop, crear uno nuevo
            asyncio.run(mark_whatsapp_messages_as_read_batch(
                message_ids, 
                enable_typing_on_last=True, 
                background_processing=True
            ))
    except Exception as e:
        logger.error("❌ Error iniciando background marking: %s", e)