"""
Sistema de deduplicación de mensajes para evitar respuestas duplicadas

Este módulo evita que se procesen múltiples veces los mismos mensajes de WhatsApp
cuando Kapso reenvía webhooks o hay retries automáticos.
"""

import time
import logging
from typing import Set, Dict, List
from threading import Lock

logger = logging.getLogger(__name__)

class MessageDeduplicator:
    """
    Deduplicador de mensajes basado en IDs únicos de WhatsApp
    
    Mantiene un cache en memoria de mensajes ya procesados con TTL automático
    para evitar crecimiento infinito de memoria.
    """
    
    def __init__(self, ttl_seconds: int = 3600):  # 1 hora por defecto
        """
        Inicializa el deduplicador
        
        Args:
            ttl_seconds: Tiempo de vida de los IDs en cache (segundos)
        """
        self.ttl_seconds = ttl_seconds
        self.processed_messages: Dict[str, float] = {}  # message_id -> timestamp
        self.lock = Lock()
        

    
    def extract_message_ids(self, webhook_data: dict) -> List[str]:
        """
        Extrae todos los IDs únicos de mensajes del webhook
        
        Args:
            webhook_data: Datos del webhook de Kapso
            
        Returns:
            Lista de IDs únicos encontrados
        """
        message_ids = []
        
        try:
            data_list = webhook_data.get("data", [])
            
            for data in data_list:
                message = data.get("message", {})
                
                # Priorizar whatsapp_message_id (más único)
                whatsapp_id = message.get("whatsapp_message_id")
                if whatsapp_id:
                    message_ids.append(f"wa:{whatsapp_id}")
                
                # Fallback al ID interno de Kapso
                internal_id = message.get("id")
                if internal_id:
                    message_ids.append(f"kapso:{internal_id}")
            
            logger.debug(f"🔍 IDs extraídos del webhook: {message_ids}")
            return message_ids
            
        except Exception as e:
            logger.error(f"❌ Error extrayendo IDs de mensaje: {e}")
            return []
    
    def are_messages_already_processed(self, message_ids: List[str]) -> bool:
        """
        Verifica si alguno de los mensajes ya fue procesado
        
        Args:
            message_ids: Lista de IDs de mensajes a verificar
            
        Returns:
            True si algún mensaje ya fue procesado, False si todos son nuevos
        """
        if not message_ids:
            return False
        
        with self.lock:
            # Limpiar cache de entradas expiradas
            self._cleanup_expired_entries()
            
            current_time = time.time()
            
            for message_id in message_ids:
                if message_id in self.processed_messages:
                    processed_time = self.processed_messages[message_id]
                    return True
            
            return False
    
    def mark_messages_as_processed(self, message_ids: List[str]) -> None:
        """
        Marca los mensajes como procesados en el cache
        
        Args:
            message_ids: Lista de IDs de mensajes a marcar
        """
        if not message_ids:
            return
        
        with self.lock:
            current_time = time.time()
            
            for message_id in message_ids:
                self.processed_messages[message_id] = current_time
            
            
    
    def _cleanup_expired_entries(self) -> None:
        """
        Limpia entradas expiradas del cache (llamado internamente)
        """
        current_time = time.time()
        expired_ids = []
        
        for message_id, timestamp in self.processed_messages.items():
            if current_time - timestamp > self.ttl_seconds:
                expired_ids.append(message_id)
        
        for message_id in expired_ids:
            del self.processed_messages[message_id]
        
    
    def get_cache_stats(self) -> dict:
        """
        Obtiene estadísticas del cache de deduplicación
        
        Returns:
            Diccionario con estadísticas
        """
        with self.lock:
            self._cleanup_expired_entries()
            return {
                "total_entries": len(self.processed_messages),
                "ttl_seconds": self.ttl_seconds,
                "oldest_entry_age": min(
                    [time.time() - timestamp for timestamp in self.processed_messages.values()],
                    default=0
                )
            }

# Instancia global del deduplicador
message_deduplicator = MessageDeduplicator(ttl_seconds=3600)  # 1 hora TTL