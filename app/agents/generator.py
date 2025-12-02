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
    routing_decision: str = "general",
    user_preferences: Optional[List[dict]] = None
) -> str:
    """
    Generator agent: Genera respuesta natural basada en productos encontrados y contexto.
    
    Args:
        user_message: Mensaje actual del usuario
        products: Lista de productos encontrados (puede estar vacía)
        conversation_context: Historial de conversación para contexto
        routing_decision: Decisión del router ("general" o "search")
        user_preferences: Lista de preferencias del usuario guardadas
        
    Returns:
        Respuesta generada para el usuario
    """
    
    # Formatear contexto de conversación
    context_text = _format_conversation_context(conversation_context)
    
    # Formatear preferencias del usuario
    from ..preferences import format_preferences_for_prompt
    preferences_text = format_preferences_for_prompt(user_preferences or [])
    
    # Construir sección de contexto si existe
    context_section = ""
    if context_text:
        context_section = f"""

CONTEXTO DE CONVERSACIÓN RECIENTE:
{context_text}

IMPORTANTE: Usa este contexto para dar respuestas coherentes y naturales. Si el cliente hace referencia a algo mencionado antes, tenlo en cuenta."""
    
    # Construir sección de preferencias si existen
    preferences_section = ""
    if preferences_text:
        preferences_section = f"""

INFORMACIÓN PERSONAL DEL CLIENTE (usa esto para hacer recomendaciones personalizadas):
{preferences_text}

IMPORTANTE: Usa esta información para recomendar productos que se adapten a las necesidades del cliente. Por ejemplo:
- Si es friolento → recomienda prendas abrigadas, suéteres, abrigos
- Si tiene sobrepeso → recomienda prendas cómodas, tallas grandes, estilos que favorezcan
- Si prefiere estilo casual → enfócate en prendas casuales, no formales"""
    
    # ESCENARIO A: Routing "general" (sin búsqueda de productos)
    if routing_decision == "general":
        prompt = f"""Eres un asistente amigable y profesional de una tienda de ropa llamada CedaMoney.

MENSAJE ACTUAL DEL CLIENTE: "{user_message}"
{context_section}
{preferences_section}

INSTRUCCIONES:
- Responde de manera natural, amigable y profesional
- Si es un saludo (hola, buenos días, etc.), saluda de vuelta de forma cálida pero breve, de otra forma NUNCA saludes al cliente.
- Si pregunta sobre la tienda (horarios, ubicación, políticas, métodos de pago, envíos), proporciona información útil
- Si es un agradecimiento, responde apropiadamente
- Si hace una pregunta que no puedes responder, sé honesto y ofrece ayuda alternativa
- NUNCA DEBES contestar pregunta que no están relacionadas a la tienda de ecommerce. Debes de decir que solo ofreces información referenta a la tienda de ropa.
- Mantén un tono conversacional y amigable
- NO uses formato de lista numerada, sé conversacional
- Si hay contexto de conversación previa, úsalo para dar respuestas más coherentes
- lee la conversación que hemos tenido y genera una respuesta con base en eso



Responde de forma natural y conversacional."""
    
    # ESCENARIO B: Routing "suggestion" CON productos encontrados (recomendaciones basadas en preferencias)
    elif routing_decision == "suggestion" and products and len(products) > 0:
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

CRITICO: NUNCA DEBES DE SALUDAR AL CLIENTE.
MENSAJE ACTUAL DEL CLIENTE: "{user_message}"
{context_section}
{preferences_section}

PRODUCTOS RECOMENDADOS (basados en las preferencias y necesidades del cliente):
{products_text}

CONTEXTO IMPORTANTE:
El cliente mencionó información sobre sus preferencias o necesidades personales (ej: "soy friolento", "tengo sobrepeso", "prefiero estilo casual"), pero NO especificó exactamente qué producto busca. Estos productos fueron seleccionados específicamente porque se adaptan a esas necesidades.

INSTRUCCIONES ESPECÍFICAS PARA SUGERENCIAS:
- Explica que basándote en las preferencias/necesidades que mencionó, le estás recomendando estos productos que se adaptan perfectamente a lo que necesita
- Para CADA producto, explica ESPECÍFICAMENTE cómo se relaciona con sus preferencias/necesidades:
  * Si es friolento → explica cómo el producto lo mantendrá abrigado (manga larga, material abrigado, tipo de prenda)
  * Si tiene sobrepeso → explica cómo el producto es cómodo, holgado, o tiene un estilo que favorece
  * Si prefiere estilo casual → explica cómo el producto es casual y versátil
  * Si mencionó otra preferencia → relaciona el producto con esa necesidad específica
- Destaca características clave (color, estilo, precio, material) y cómo se relacionan con las preferencias del cliente
- Menciona los precios en pesos mexicanos (MXN)
- Sé conversacional y amigable, como si estuvieras ayudando personalmente al cliente
- NO uses formato de lista numerada, sé conversacional
- Si hay contexto de conversación previa, úsalo para hacer la recomendación más personalizada
- El tono debe ser proactivo y empático, mostrando que entiendes sus necesidades

REGLAS CRÍTICAS:
- NUNCA digas que el cliente "buscó" algo específico, porque no lo hizo - solo mencionó preferencias
- Enfócate en explicar POR QUÉ estos productos son adecuados para SUS necesidades específicas
- Si hay múltiples preferencias, combínalas en tus explicaciones (ej: "perfecto para alguien friolento que también prefiere comodidad")

Responde de forma natural, explicando cómo cada producto se adapta a las necesidades específicas del cliente."""
    
    # ESCENARIO C: Routing "search" CON productos encontrados (búsqueda específica)
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
        
       CRITICO: NUNCA DEBES DE SALUDAR AL CLIENTE.

MENSAJE ACTUAL DEL CLIENTE: "{user_message}"
{context_section}
{preferences_section}

PRODUCTOS ENCONTRADOS:
{products_text}

INSTRUCCIONES:
- Responde directamente a la pregunta o solicitud del cliente
- Recomienda los productos más adecuados basándote en lo que el cliente busca Y sus preferencias personales
- Si el cliente tiene preferencias guardadas (ej: es friolento, tiene sobrepeso), prioriza productos que se adapten a esas necesidades
- Destaca características clave que sean relevantes (color, estilo, precio, etc.) y cómo se relacionan con las preferencias del cliente
- Menciona los precios en pesos mexicanos (MXN)
- Si hay contexto de conversación previa, úsalo para hacer recomendaciones más personalizadas
- Sé conversacional y amigable, como si estuvieras ayudando a un cliente en la tienda
- NO uses formato de lista numerada, sé conversacional
- Si el cliente mencionó algo específico antes (ej: "verde", "formal"), enfócate en eso
- Si hay información personal del cliente, úsala para explicar POR QUÉ un producto es adecuado para él/ella

Responde de forma natural, destacando los productos más relevantes y explicando cómo se adaptan a las necesidades del cliente."""
    
    # ESCENARIO C: Routing "search" SIN productos encontrados
    else:
        prompt = f"""Eres un asistente de ventas amigable y experto de una tienda de ropa llamada CedaMoney.


MENSAJE ACTUAL DEL CLIENTE: "{user_message}"
{context_section}
{preferences_section}

Si no existe un contexto de conversación, saluda al usario y dile que eres CEDAMONEY que estás ahí para ayudarlo a encontrar su prenda de ropa ideal.  

PROBLEMA: No se encontraron productos que coincidan exactamente con la búsqueda del cliente.

INSTRUCCIONES:
- CRITICO: NO debes de saludar al cliente salvo que sea el inicio de una conversación.
- Sé empático y reconoce que no encontraste exactamente lo que busca
- Sugiere alternativas o reformulaciones de la búsqueda considerando las preferencias del cliente
- Si el cliente tiene preferencias guardadas (ej: es friolento), sugiere buscar productos que se adapten a esas necesidades
- Si el cliente mencionó características específicas (color, tipo, estilo), sugiere buscar con criterios más amplios
- Ofrece ayuda para refinar la búsqueda
- Mantén un tono positivo y útil
- NO uses formato de lista numerada, sé conversacional
- Si hay contexto de conversación previa, úsalo para entender mejor qué busca el cliente

No debes de saludar, NUNCA

Responde de forma natural, ofreciendo ayuda y alternativas."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.6
    )
    
    return response.choices[0].message.content
