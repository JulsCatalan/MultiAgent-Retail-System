# app/agents/generator.py
from openai import OpenAI
import os
from typing import List, Optional
from models import ConversationMessage

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _format_conversation_context(
    conversation_context: Optional[List[ConversationMessage]] = None,
    max_messages: int = 15
) -> str:
    """
    Formatea el contexto de conversación para incluir en el prompt.
    Prioriza los mensajes más recientes.
    
    Args:
        conversation_context: Lista de mensajes de conversación
        max_messages: Número máximo de mensajes a incluir
        
    Returns:
        String formateado con el contexto de conversación
    """
    if not conversation_context:
        return ""
    
    # Invertir para tener los más recientes primero
    recent_messages = list(reversed(conversation_context))
    
    # Tomar solo los últimos mensajes
    messages_to_include = recent_messages[:max_messages]
    
    formatted_messages = []
    for msg in messages_to_include:
        sender_label = "Cliente" if msg.sender == "client" else "Asistente"
        formatted_messages.append(f"{sender_label}: {msg.message}")
    
    if formatted_messages:
        return "\n".join(formatted_messages)
    
    return ""


def generate_response(
    user_message: str,
    products: list = None,
    conversation_context: Optional[List[ConversationMessage]] = None,
    routing_decision: str = "general"
) -> str:
    """
    Generator agent: Genera respuesta natural basada en productos encontrados y contexto.
    
    Args:
        user_message: Mensaje actual del usuario
        products: Lista de productos encontrados (puede estar vacía)
        conversation_context: Historial de conversación para contexto
        routing_decision: Decisión del router ("general" o "search")
        
    Returns:
        Respuesta generada para el usuario
    """
    
    # Formatear contexto de conversación
    context_text = _format_conversation_context(conversation_context)
    
    # Construir sección de contexto si existe
    context_section = ""
    if context_text:
        context_section = f"""

CONTEXTO DE CONVERSACIÓN RECIENTE:
{context_text}

IMPORTANTE: Usa este contexto para dar respuestas coherentes y naturales. Si el cliente hace referencia a algo mencionado antes, tenlo en cuenta."""
    
    # ESCENARIO A: Routing "general" (sin búsqueda de productos)
    if routing_decision == "general":
        prompt = f"""Eres un asistente amigable y profesional de una tienda de ropa llamada CedaMoney.

MENSAJE ACTUAL DEL CLIENTE: "{user_message}"
{context_section}

INSTRUCCIONES:
- Responde de manera natural, amigable y profesional
- Si es un saludo (hola, buenos días, etc.), saluda de vuelta de forma cálida pero breve
- Si pregunta sobre la tienda (horarios, ubicación, políticas, métodos de pago, envíos), proporciona información útil
- Si es un agradecimiento, responde apropiadamente
- Si hace una pregunta que no puedes responder, sé honesto y ofrece ayuda alternativa
- NUNCA DEBES contestar pregunta que no están relacionadas a la tienda de ecommerce. Debes de decir que solo ofreces información referenta a la tienda de ropa.
- Mantén un tono conversacional y amigable
- NO uses formato de lista numerada, sé conversacional
- Si hay contexto de conversación previa, úsalo para dar respuestas más coherentes
- lee la conversación que hemos tenido y genera una respuesta con base en eso

Responde de forma natural y conversacional."""
    
    # ESCENARIO B: Routing "search" CON productos encontrados
    elif products and len(products) > 0:
        # Formatear productos
        products_text = "\n\n".join([
            f"""Producto {i+1}:
- Nombre: {p['prod_name']}
- Tipo: {p['product_type_name']}
- Grupo: {p['product_group_name']}
- Color: {p['colour_group_name']}
- Descripción: {p['detail_desc']}
- Precio: ${p['price_mxn']:.2f} MXN
- ID: {p['article_id']}"""
            for i, p in enumerate(products)
        ])
        
        prompt = f"""Eres un asistente de ventas amigable y experto de una tienda de ropa llamada CedaMoney.

MENSAJE ACTUAL DEL CLIENTE: "{user_message}"
{context_section}

PRODUCTOS ENCONTRADOS:
{products_text}

INSTRUCCIONES:
- Responde directamente a la pregunta o solicitud del cliente
- Recomienda los productos más adecuados basándote en lo que el cliente busca
- Destaca características clave que sean relevantes (color, estilo, precio, etc.)
- Menciona los precios en pesos mexicanos (MXN)
- Si hay contexto de conversación previa, úsalo para hacer recomendaciones más personalizadas
- Sé conversacional y amigable, como si estuvieras ayudando a un cliente en la tienda
- NO uses formato de lista numerada, sé conversacional
- Si el cliente mencionó algo específico antes (ej: "verde", "formal"), enfócate en eso

Responde de forma natural, destacando los productos más relevantes."""
    
    # ESCENARIO C: Routing "search" SIN productos encontrados
    else:
        prompt = f"""Eres un asistente de ventas amigable y experto de una tienda de ropa llamada CedaMoney.

MENSAJE ACTUAL DEL CLIENTE: "{user_message}"
{context_section}

PROBLEMA: No se encontraron productos que coincidan exactamente con la búsqueda del cliente.

INSTRUCCIONES:
- Sé empático y reconoce que no encontraste exactamente lo que busca
- Sugiere alternativas o reformulaciones de la búsqueda
- Si el cliente mencionó características específicas (color, tipo, estilo), sugiere buscar con criterios más amplios
- Ofrece ayuda para refinar la búsqueda
- Mantén un tono positivo y útil
- NO uses formato de lista numerada, sé conversacional
- Si hay contexto de conversación previa, úsalo para entender mejor qué busca el cliente

Responde de forma natural, ofreciendo ayuda y alternativas."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.7
    )
    
    return response.choices[0].message.content
