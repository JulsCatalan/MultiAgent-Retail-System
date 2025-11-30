from pydantic import BaseModel, model_validator
from typing import Optional, Literal, List
import os

class UserMetadata(BaseModel):
    whatsapp_config_id: Optional[str] = None
    reached_from_phone_number: Optional[str] = None
    preferred_language: Optional[str] = None  
    language_confidence: Optional[float] = None  
    
    
class User(BaseModel):
    name: str
    phone_number: Optional[str] = None
    conversation_id: str
    metadata: Optional[UserMetadata] = None
    

class ConversationConfig(BaseModel):
    reached_from_phone_number: str
    whatsapp_conversation_id: str
    whatsapp_config_id: str
    phone_number: str
    contact_name: str
    is_new_conversation: bool
    direction: Literal["inbound", "outbound"] = "inbound"
    
class ConversationMessage(BaseModel):
    """Represents a message in the conversation history"""
    timestamp: str
    sender: str  # 'client' or 'cedamoney'
    message: str
    message_id: Optional[str] = None
class Context(BaseModel):
    """
    Contexto simplificado para el agente
    El agente ahora lee archivos .md directamente cuando los necesita
    """
    agent_name: str = "Cedamoney"
    agency_name: str = "Cedamanco"
    config: Optional[ConversationConfig] = None
    
    
    current_client: User 
    
    conversation_history: List[ConversationMessage] = []
