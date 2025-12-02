import json
import os
from typing import Dict, Any, List, Optional

from openai import OpenAI
import logging

logger = logging.getLogger(__name__)

from ..cart import (
    add_to_cart,
    remove_from_cart,
    get_cart,
    get_recent_products,
    create_stripe_checkout_for_whatsapp,
    format_checkout_message,
    format_cart_summary,
    calculate_cart_total,
    get_cart_by_conversation,
    clear_cart as clear_cart_func,
    save_recent_products,
)
from .retriever import search_products
from models import ConversationMessage

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _format_conversation_context_for_cart(
    conversation_context: Optional[List[ConversationMessage]] = None,
    max_messages: int = 10,
) -> str:
    """
    Formatea el contexto de conversación reciente para los prompts del cart agent.
    """
    if not conversation_context:
        return ""

    # Tomar los últimos mensajes (más recientes)
    recent_messages = list(reversed(conversation_context))[:max_messages]
    lines = []
    for msg in recent_messages:
        sender = "Cliente" if msg.sender == "client" else "Asistente"
        lines.append(f"{sender}: {msg.message}")

    if not lines:
        return ""

    return "\n".join(lines)


def detect_cart_intent_llm(
    user_message: str,
    recent_products: List[Dict[str, Any]],
    conversation_context: Optional[List[ConversationMessage]] = None,
) -> Dict[str, Any]:
    """
    Agente LLM que detecta la intención del usuario respecto al carrito.

    Retorna un dict con:
    - mode: 'none' | 'add_to_cart' | 'remove_from_cart' | 'show_cart'
    - product_index: int | None  (para 'add_to_cart' o 'remove_from_cart', índice del producto)
    - needs_confirmation: bool
    - confidence: float (0.0 a 1.0)
    """
    # Formatear productos recientes para el prompt
    if recent_products:
        products_text = "\n".join(
            [
                f"Producto {p['position']}: {p['prod_name']} ({p['colour_group_name']}) - {p['product_group_name']}"
                for p in recent_products
            ]
        )
    else:
        products_text = "No hay productos recientes asociados a esta conversación."

    # Formatear contexto reciente de conversación
    context_text = _format_conversation_context_for_cart(conversation_context)
    context_section = ""
    if context_text:
        context_section = f"""

CONVERSACIÓN RECIENTE:
{context_text}
"""

    prompt = f"""Eres un agente que interpreta la intención del usuario en una tienda de ropa (CedaMoney).

Tu tarea es decidir si el usuario quiere:
- SOLO hablar o preguntar sobre productos (información general)
- VER su carrito actual
- AGREGAR un producto específico de la lista reciente al carrito
- QUITAR/ELIMINAR un producto del carrito
- PROCEDER AL PAGO / CHECKOUT (finalizar compra)
- SEGUIR COMPRANDO / MODIFICAR CARRITO (después de ver checkout)
- VACIAR el carrito completamente

MENSAJE DEL USUARIO:
"{user_message}"

PRODUCTOS RECIENTES MOSTRADOS AL USUARIO:
{products_text}
{context_section}

REGLAS:
1. Si el usuario solo describe gustos o hace preguntas sobre productos, pero NO habla explícitamente de carrito:
   - mode = "none"

2. Si el usuario quiere ver su carrito ("ver carrito", "qué tengo", "muéstrame el carrito"):
   - mode = "show_cart"

3. Si el usuario quiere AGREGAR un producto ("agrega el producto X", "quiero el suéter blanco"):
   - mode = "add_to_cart"
   - product_index = número del producto en la lista reciente (1, 2, 3, ...) SI se puede inferir.

4. Si el usuario quiere QUITAR/ELIMINAR un producto del carrito ("quita el producto X", "elimina", "no quiero el producto 1"):
   - mode = "remove_from_cart"
   - product_index = número del producto EN EL CARRITO (1, 2, 3, ...)

5. Si el usuario quiere PROCEDER AL PAGO ("pagar", "checkout", "proceder al pago", "comprar", "finalizar compra", "quiero pagar", "pagar ahora"):
   - mode = "checkout"

6. Si el usuario quiere SEGUIR COMPRANDO después de ver el checkout ("seguir comprando", "agregar más", "modificar carrito", "cancelar pago", "espera", "añadir más productos"):
   - mode = "continue_shopping"

7. Si el usuario quiere VACIAR todo el carrito ("vaciar carrito", "eliminar todo", "borrar carrito", "empezar de nuevo"):
   - mode = "clear_cart"

8. Si el usuario es ambiguo, establece needs_confirmation = true y confidence menor (0.5).

9. NUNCA inventes productos que no están en la lista reciente.

Formato de respuesta:
Responde SOLO con un JSON válido, sin texto adicional:
{{
  "mode": "none" | "add_to_cart" | "remove_from_cart" | "show_cart" | "checkout" | "continue_shopping" | "clear_cart",
  "product_index": <número del producto o null>,
  "needs_confirmation": true | false,
  "confidence": <número entre 0.0 y 1.0>
}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0,
    )

    content = response.choices[0].message.content.strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Fallback defensivo
        return {
            "mode": "none",
            "product_index": None,
            "needs_confirmation": True,
            "confidence": 0.0,
        }

    # Normalizar campos mínimos
    mode = data.get("mode", "none")
    product_index = data.get("product_index")
    needs_confirmation = bool(data.get("needs_confirmation", False))
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    return {
        "mode": mode,
        "product_index": product_index,
        "needs_confirmation": needs_confirmation,
        "confidence": confidence,
    }


def resolve_product_reference(
    user_message: str,
    recent_products: List[Dict[str, Any]],
    conversation_context: Optional[List[ConversationMessage]] = None,
) -> Dict[str, Any]:
    """
    Resuelve referencias a productos por descripción (ej: "el suéter blanco", "esa camisa verde")
    en lugar de solo por número.
    
    Retorna:
    - resolved: bool
    - product_index: int | None
    - article_id: str | None
    - confidence: float
    - reason: str
    """
    if not recent_products:
        return {
            "resolved": False,
            "product_index": None,
            "article_id": None,
            "confidence": 0.0,
            "reason": "No hay productos recientes disponibles",
        }

    # Formatear productos con más detalles para mejor matching
    products_text = "\n".join(
        [
            f"Producto {p['position']}: {p['prod_name']} | Color: {p['colour_group_name']} | Tipo: {p['product_type_name']} | Grupo: {p['product_group_name']}"
            for p in recent_products
        ]
    )

    # Formatear contexto reciente de conversación
    context_text = _format_conversation_context_for_cart(conversation_context)
    context_section = ""
    if context_text:
        context_section = f"""

CONVERSACIÓN RECIENTE:
{context_text}
"""

    prompt = f"""Eres un agente experto que resuelve referencias a productos en una conversación de tienda de ropa.

El usuario mencionó algo sobre un producto en este mensaje:
"{user_message}"

PRODUCTOS RECIENTES DISPONIBLES:
{products_text}
{context_section}

Tu tarea es identificar QUÉ producto específico de la lista el usuario está mencionando basándote en:
- Descripción del color (ej: "blanco", "verde", "azul")
- Tipo de prenda (ej: "suéter", "camisa", "playera", "pantalón")
- Características mencionadas
- Referencias como "ese", "esa", "el", "la"

INSTRUCCIONES:
1. Analiza el mensaje del usuario y busca coincidencias con los productos de la lista
2. Si encuentras una coincidencia clara (ej: usuario dice "suéter blanco" y hay un Producto X que es suéter y blanco), devuelve ese producto
3. Si hay múltiples coincidencias posibles o la referencia es ambigua, devuelve confidence bajo (< 0.7)
4. Si no hay coincidencias claras, devuelve resolved = false

Responde SOLO con un JSON válido, sin texto adicional:
{{
  "resolved": true | false,
  "product_index": <número del producto (1, 2, 3...) o null>,
  "confidence": <número entre 0.0 y 1.0>,
  "reason": "<explicación breve de por qué elegiste este producto o por qué no se pudo resolver>"
}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0,
    )
    
    content = response.choices[0].message.content.strip()
    
    try:
        data = json.loads(content)
        resolved = bool(data.get("resolved", False))
        product_index = data.get("product_index")
        confidence = float(data.get("confidence", 0.0))
        reason = data.get("reason", "")
        
        # Si se resolvió, obtener el article_id
        article_id = None
        if resolved and product_index:
            product = next((p for p in recent_products if p["position"] == product_index), None)
            if product:
                article_id = product["article_id"]
            else:
                resolved = False
                reason = f"Producto {product_index} no encontrado en la lista reciente"
        
        return {
            "resolved": resolved,
            "product_index": product_index,
            "article_id": article_id,
            "confidence": confidence,
            "reason": reason,
        }
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        return {
            "resolved": False,
            "product_index": None,
            "article_id": None,
            "confidence": 0.0,
            "reason": f"Error procesando respuesta: {str(e)}",
        }


def resolve_cart_product_reference(
    user_message: str,
    cart_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Resuelve referencias a productos que están en el carrito (ej: "quita el producto 1", "elimina esa camisa verde").
    
    Retorna:
    - resolved: bool
    - product_index: int | None (posición en el carrito, 1-indexed)
    - article_id: str | None
    - confidence: float
    - reason: str
    """
    if not cart_items:
        return {
            "resolved": False,
            "product_index": None,
            "article_id": None,
            "confidence": 0.0,
            "reason": "El carrito está vacío",
        }
    
    # Formatear productos del carrito con más detalles
    cart_text = "\n".join(
        [
            f"Producto {i+1}: {item['prod_name']} ({item['colour_group_name']}) | Tipo: {item['product_type_name']} | Grupo: {item['product_group_name']} | Cantidad: {item['quantity']}"
            for i, item in enumerate(cart_items)
        ]
    )
    
    prompt = f"""Eres un agente experto que resuelve referencias a productos en el carrito de compras.

El usuario quiere quitar/eliminar un producto del carrito. Mensaje:
"{user_message}"

PRODUCTOS EN EL CARRITO:
{cart_text}

Tu tarea es identificar QUÉ producto específico del carrito el usuario quiere quitar basándote en:
- Número de posición (ej: "producto 1", "el primero")
- Descripción del color (ej: "blanco", "verde", "azul")
- Tipo de prenda (ej: "suéter", "camisa", "playera")
- Características mencionadas
- Referencias como "ese", "esa", "el", "la"

INSTRUCCIONES:
1. Analiza el mensaje del usuario y busca coincidencias con los productos del carrito
2. Si encuentras una coincidencia clara (ej: usuario dice "quita el producto 1" o "elimina la camisa verde"), devuelve ese producto
3. Si hay múltiples coincidencias posibles o la referencia es ambigua, devuelve confidence bajo (< 0.7)
4. Si no hay coincidencias claras, devuelve resolved = false

Responde SOLO con un JSON válido, sin texto adicional:
{{
  "resolved": true | false,
  "product_index": <número del producto en el carrito (1, 2, 3...) o null>,
  "confidence": <número entre 0.0 y 1.0>,
  "reason": "<explicación breve de por qué elegiste este producto o por qué no se pudo resolver>"
}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.1,
    )
    
    content = response.choices[0].message.content.strip()
    
    try:
        data = json.loads(content)
        resolved = bool(data.get("resolved", False))
        product_index = data.get("product_index")
        confidence = float(data.get("confidence", 0.0))
        reason = data.get("reason", "")
        
        # Si se resolvió, obtener el article_id
        article_id = None
        if resolved and product_index:
            # product_index es 1-indexed, convertir a 0-indexed para array
            idx = product_index - 1
            if 0 <= idx < len(cart_items):
                article_id = cart_items[idx]["article_id"]
            else:
                resolved = False
                reason = f"Producto {product_index} no encontrado en el carrito"
        
        return {
            "resolved": resolved,
            "product_index": product_index,
            "article_id": article_id,
            "confidence": confidence,
            "reason": reason,
        }
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        return {
            "resolved": False,
            "product_index": None,
            "article_id": None,
            "confidence": 0.0,
            "reason": f"Error procesando respuesta: {str(e)}",
        }


def extract_catalog_search_query(
    user_message: str,
    conversation_context: Optional[List[ConversationMessage]] = None,
) -> str:
    """
    Extrae una query corta para buscar en el catálogo cuando el usuario menciona
    un producto por nombre (ej: "datguy pants") aunque no esté en recent_products.
    """
    context_text = _format_conversation_context_for_cart(conversation_context)
    context_section = ""
    if context_text:
        context_section = f"""

CONVERSACIÓN RECIENTE:
{context_text}
"""

    prompt = f"""Eres un asistente que debe extraer una consulta corta para buscar un producto en el catálogo.

MENSAJE ACTUAL DEL USUARIO:
"{user_message}"
{context_section}

INSTRUCCIONES:
- Identifica si el usuario menciona explícitamente un nombre de producto o prenda específica (ej: "datguy pants", "zack tee", "happy hoodie").
- Si existe un nombre claro de producto, genera una query de búsqueda corta usando ese nombre y, opcionalmente, algún atributo relevante (color, tipo).
- Si no hay un nombre claro, genera una query corta basada en lo más probable que esté buscando.
- La respuesta DEBE ser solo la query de búsqueda, SIN texto adicional, SIN comillas.

Ejemplos de respuesta:
- datguy pants
- zack tee playera azul
- happy hoodie sudadera gris
- suéter blanco abrigado
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0,
    )

    return response.choices[0].message.content.strip()


def handle_cart_interaction(
    conversation_id: str,
    user_message: str,
    user_name: str = "Cliente",
    phone_number: str = None,
    conversation_context: Optional[List[ConversationMessage]] = None,
) -> Dict[str, Any]:
    """
    Maneja la lógica de carrito si aplica.
    
    Args:
        conversation_id: ID de la conversación
        user_message: Mensaje del usuario
        user_name: Nombre del usuario (de WhatsApp)
        phone_number: Teléfono del usuario (de WhatsApp)
    
    Devuelve:
        - handled: bool  -> True si ya se generó una respuesta
    - response: str  -> Mensaje para el usuario (si handled=True)
        - products: list -> Lista de productos a devolver (opcional)
    """
    recent = get_recent_products(conversation_id)
    intent = detect_cart_intent_llm(user_message, recent, conversation_context=conversation_context)
    mode = intent["mode"]
    
    logger.info(
        f"🛒 [CART-AGENT] Intent detectado - "
        f"Mode: {mode}, "
        f"ConvID: {conversation_id}, "
        f"Confidence: {intent.get('confidence', 0.0):.2f}"
    )

    if mode == "none":
        logger.debug("➡️ Cart agent no maneja este mensaje, pasando a agente normal")
        return {"handled": False}

    # Ver carrito
    if mode == "show_cart":
        cart_items = get_cart(conversation_id)
        if not cart_items:
            response = (
                "Por ahora tu carrito está vacío. 🛒\n\n"
                "Cuando veas productos que te gusten, puedes decirme:\n"
                "• \"Agrega el producto 1\"\n"
                "• \"Quiero el suéter azul\""
            )
            return {"handled": True, "response": response, "products": []}

        # Calcular total
        cart_id = get_cart_by_conversation(conversation_id)
        total = calculate_cart_total(cart_id) if cart_id else 0.0
        
        # Usar el formateador mejorado
        response = format_cart_summary(cart_items, total)
        return {"handled": True, "response": response, "products": []}
    
    # Proceder al pago / Checkout
    if mode == "checkout":
        logger.info(f"💳 [CHECKOUT] Usuario solicita checkout - ConvID: {conversation_id}")
        
        cart_items = get_cart(conversation_id)
        
        # Validar que el carrito no esté vacío
        if not cart_items:
            logger.warning(f"⚠️ Intento de checkout con carrito vacío - ConvID: {conversation_id}")
            response = (
                "Tu carrito está vacío. 🛒\n\n"
                "Primero busca y agrega algunos productos que te gusten, "
                "luego podrás proceder al pago."
            )
            return {"handled": True, "response": response, "products": []}

        # Validar que tengamos nombre y teléfono del usuario
        if not user_name or not phone_number:
            logger.warning(f"⚠️ Checkout sin datos de usuario - Name: {user_name}, Phone: {phone_number}")
            response = (
                "Para proceder con el pago, necesito confirmar tus datos. 📝\n\n"
                "Por favor, asegúrate de que tu perfil de WhatsApp tenga:\n"
                "• Tu nombre completo\n"
                "• Tu número de teléfono\n\n"
                "Luego intenta de nuevo."
            )
            return {"handled": True, "response": response, "products": []}

        # Crear sesión de checkout de Stripe
        logger.info(f"💳 Creando sesión Stripe para {user_name} ({phone_number})")
        checkout_result = create_stripe_checkout_for_whatsapp(
            conversation_id=conversation_id,
            user_name=user_name,
            phone_number=phone_number
        )
        
        # Manejar error
        if not checkout_result.get("success"):
            error_msg = checkout_result.get("error", "Error desconocido")
            logger.error(f"❌ Error creando checkout: {error_msg}")
            response = (
                f"❌ Lo siento, hubo un problema al crear tu sesión de pago.\n\n"
                f"Error: {error_msg}\n\n"
                f"Por favor intenta de nuevo en unos momentos o contacta a soporte."
            )
            return {"handled": True, "response": response, "products": []}

        # Éxito - formatear mensaje con link de pago
        logger.info(
            f"✅ Checkout creado exitosamente - "
            f"SessionID: {checkout_result['session_id']}, "
            f"Total: ${checkout_result['total_amount']:.2f} MXN, "
            f"Items: {checkout_result['items_count']}"
        )
        
        response = format_checkout_message(
            cart_items=checkout_result["cart_items"],
            total=checkout_result["total_amount"],
            checkout_url=checkout_result["checkout_url"]
        )
        
        return {"handled": True, "response": response, "products": []}
    
    # Seguir comprando / Modificar carrito (después de checkout)
    if mode == "continue_shopping":
        logger.info(f"🛍️ Usuario decidió seguir comprando - ConvID: {conversation_id}")
        
        response = (
            "¡Perfecto! Puedes seguir explorando productos. 🛍️\n\n"
            "Dime qué estás buscando y te ayudo a encontrarlo:\n"
            "• \"Busco vestidos rojos\"\n"
            "• \"Muéstrame jeans\"\n"
            "• \"¿Tienes camisetas?\"\n\n"
            "También puedes:\n"
            "• Ver tu carrito: \"Ver carrito\"\n"
            "• Modificar items: \"Quita el producto 1\"\n"
            "• Proceder al pago cuando estés listo"
        )
        return {"handled": True, "response": response, "products": []}
    
    # Vaciar carrito completamente
    if mode == "clear_cart":
        logger.info(f"🗑️ Usuario solicita vaciar carrito - ConvID: {conversation_id}")
        
        cart_id = get_cart_by_conversation(conversation_id)
        
        if not cart_id:
            logger.debug("Carrito ya vacío (no existe cart_id)")
            response = "Tu carrito ya está vacío. 🛒"
            return {"handled": True, "response": response, "products": []}
        
        cart_items = get_cart(conversation_id)
        
        if not cart_items:
            logger.debug("Carrito ya vacío (sin items)")
            response = "Tu carrito ya está vacío. 🛒"
            return {"handled": True, "response": response, "products": []}
        
        # Confirmación antes de vaciar (si la confianza es baja)
        confidence = intent.get("confidence", 0.0)
        if confidence < 0.8:
            items_count = len(cart_items)
            logger.info(f"⚠️ Solicitando confirmación para vaciar (confidence: {confidence:.2f})")
            response = (
                f"⚠️ ¿Estás seguro de que quieres vaciar tu carrito?\n\n"
                f"Tienes {items_count} producto(s) en el carrito.\n\n"
                f"Responde \"sí, vaciar carrito\" para confirmar."
            )
            return {"handled": True, "response": response, "products": []}
        
        # Vaciar el carrito
        items_count = len(cart_items)
        clear_cart_func(cart_id)
        logger.info(f"✅ Carrito vaciado - {items_count} items eliminados")
        
        response = (
            "✅ Tu carrito ha sido vaciado completamente.\n\n"
            "¿Qué te gustaría buscar ahora?"
        )
        return {"handled": True, "response": response, "products": []}

    # Quitar del carrito
    if mode == "remove_from_cart":
        logger.info(f"➖ Usuario quiere quitar del carrito - ConvID: {conversation_id}")
        
        cart_items = get_cart(conversation_id)
        if not cart_items:
            logger.debug("Intento de quitar de carrito vacío")
            return {
                "handled": True,
                "response": "Tu carrito está vacío, no hay nada que quitar. 🛒",
                "products": [],
            }
        
        index = intent.get("product_index")
        needs_confirmation = intent.get("needs_confirmation", False)
        confidence = float(intent.get("confidence", 0.0))
        
        product = None
        resolved_index = index
        article_id = None
        
        # Si no hay índice numérico, intentar resolver por descripción usando productos del carrito
        if not index or index <= 0:
            reference_result = resolve_cart_product_reference(user_message, cart_items)
            
            if reference_result["resolved"]:
                resolved_index = reference_result["product_index"]
                article_id = reference_result["article_id"]
                confidence = reference_result["confidence"]
                # Si la confianza de la resolución es baja, necesita confirmación
                if confidence < 0.7:
                    needs_confirmation = True
            else:
                # No se pudo resolver la referencia
                cart_list = []
                for i, item in enumerate(cart_items[:5], start=1):
                    cart_list.append(
                        f"Producto {i}: {item['prod_name']} ({item['colour_group_name']})"
                    )
                cart_text = ", ".join(cart_list)
                
                return {
                    "handled": True,
                    "response": (
                        f"No pude identificar exactamente qué producto quieres quitar del carrito. "
                        f"Puedes referirte a los productos por número (del 1 al {len(cart_items)}) "
                        "o por descripción (ej: \"quita el producto 1\", \"elimina la camisa verde\"). "
                        f"Productos en tu carrito: {cart_text}"
                    ),
                    "products": [],
                }
        else:
            # Buscar producto por posición en el carrito
            if 1 <= index <= len(cart_items):
                product = cart_items[index - 1]  # Convertir a 0-indexed
                article_id = product["article_id"]
            else:
                return {
                    "handled": True,
                    "response": (
                        f"No encontré el producto {index} en tu carrito. "
                        f"Actualmente tienes {len(cart_items)} producto(s). "
                        "Puedes decirme, por ejemplo, \"quita el producto 1\"."
                    ),
                    "products": [],
                }
        
        # Si no se encontró el producto, intentar resolver
        if not product and not article_id:
            reference_result = resolve_cart_product_reference(user_message, cart_items)
            if reference_result["resolved"]:
                resolved_index = reference_result["product_index"]
                article_id = reference_result["article_id"]
                confidence = reference_result["confidence"]
                if 1 <= resolved_index <= len(cart_items):
                    product = cart_items[resolved_index - 1]
                if confidence < 0.7:
                    needs_confirmation = True
        
        if not product and not article_id:
            return {
                "handled": True,
                "response": (
                    f"No pude identificar qué producto quieres quitar. "
                    f"Tu carrito tiene {len(cart_items)} producto(s). "
                    "Puedes decirme, por ejemplo, \"quita el producto 1\" o \"elimina la camisa verde\"."
                ),
                "products": [],
            }
        
        # Si necesita confirmación o la confianza es baja, NO modificamos el carrito
        if needs_confirmation or confidence < 0.8:
            product_name = product.get("prod_name", "ese producto") if product else "ese producto"
            product_color = product.get("colour_group_name", "") if product else ""
            color_text = f" ({product_color})" if product_color else ""
            
            response = (
                f"Entiendo que quieres quitar el Producto {resolved_index}: {product_name}{color_text}. "
                "Solo para confirmar, ¿quieres que lo elimine del carrito? "
                "Puedes decirme \"sí, quita el producto "
                f"{resolved_index}\" o simplemente \"sí\"."
            )
            return {"handled": True, "response": response, "products": []}
        
        # Confianza alta: procedemos a quitar del carrito
        remove_from_cart(conversation_id, article_id)
        
        product_name = product.get("prod_name", "el producto") if product else "el producto"
        product_color = product.get("colour_group_name", "") if product else ""
        color_text = f" ({product_color})" if product_color else ""
        
        logger.info(
            f"✅ Producto quitado del carrito - "
            f"ArticleID: {article_id}, "
            f"Nombre: {product_name}, "
            f"ConvID: {conversation_id}"
        )
        
        response = (
            f"✅ He quitado del carrito:\n\n"
            f"*{product_name}*{color_text}\n\n"
            f"¿Qué deseas hacer ahora?\n"
            f"• \"Ver carrito\" - Ver tu carrito actualizado\n"
            f"• \"Seguir comprando\" - Buscar más productos\n"
            f"• \"Proceder al pago\" - Si estás listo"
        )
        return {"handled": True, "response": response, "products": []}

    # Agregar al carrito usando posición de producto reciente
    if mode == "add_to_cart":
        logger.info(f"➕ Usuario quiere agregar al carrito - ConvID: {conversation_id}")
        
        index = intent.get("product_index")
        needs_confirmation = intent.get("needs_confirmation", False)
        confidence = float(intent.get("confidence", 0.0))

        if not recent:
            logger.warning(f"⚠️ No hay productos recientes para agregar")
            return {
                "handled": True,
                "response": (
                    "Aún no tengo productos recientes asociados a esta conversación. 🔍\n\n"
                    "Primero busca productos (ej: \"Busco vestidos rojos\") y luego podrás:\n"
                    "• \"Agrega el producto 1\"\n"
                    "• \"Quiero el suéter blanco\""
                ),
                "products": [],
            }

        product = None
        resolved_index = index
        
        # Si no hay índice numérico, intentar resolver por descripción usando productos recientes
        if not index or index <= 0:
            reference_result = resolve_product_reference(
                user_message,
                recent,
                conversation_context=conversation_context,
            )

            if reference_result["resolved"]:
                resolved_index = reference_result["product_index"]
                confidence = reference_result["confidence"]
                # Si la confianza de la resolución es baja, necesita confirmación
                if confidence < 0.7:
                    needs_confirmation = True
            else:
                # No se pudo resolver la referencia en la lista reciente:
                # intentar buscar directamente en el catálogo usando el contexto.
                search_query = extract_catalog_search_query(
                    user_message, conversation_context=conversation_context
                )

                if not search_query:
                    max_idx = max(p["position"] for p in recent) if recent else 0
                    products_list = []
                    for p in recent[:5]:
                        products_list.append(
                            f"Producto {p['position']}: {p['prod_name']} ({p['colour_group_name']})"
                        )
                    products_text = ", ".join(products_list)

                    return {
                        "handled": True,
                        "response": (
                            "No pude identificar exactamente qué producto quieres agregar. "
                            f"Puedes referirte a los productos por número (del 1 al {max_idx}) "
                            "o por descripción (ej: \"el suéter blanco\", \"esa camisa verde\"). "
                            f"Productos disponibles: {products_text}"
                        ),
                        "products": [],
                    }

                # Buscar en el catálogo: esto SOLO devuelve productos que realmente existen
                catalog_products = search_products(search_query)

                if not catalog_products:
                    return {
                        "handled": True,
                        "response": (
                            "He buscado en nuestro catálogo y no encontré un producto que coincida "
                            "claramente con lo que mencionas. "
                            "Puede que ese modelo específico no exista en nuestra tienda. "
                            "Si quieres, puedo sugerirte alternativas similares."
                        ),
                        "products": [],
                    }

                # Guardar estos productos como recientes para futuras referencias (Producto 1, etc.)
                save_recent_products(conversation_id, catalog_products)

                # Mostrar al usuario la mejor coincidencia y pedir confirmación explícita
                best = catalog_products[0]
                response = (
                    "No encontré ese producto exacto entre los últimos que vimos, "
                    "pero en el catálogo encontré esta opción que parece coincidir con lo que dices:\n\n"
                    f"Producto 1: {best['prod_name']} ({best['colour_group_name']}) - "
                    f"{best['product_group_name']} - ${best['price_mxn']:.2f} MXN.\n\n"
                    "Si quieres que lo agregue al carrito, puedes decirme por ejemplo "
                    "\"agrega el producto 1 al carrito\" o \"sí, agrega ese producto\"."
                )

                return {
                    "handled": True,
                    "response": response,
                    "products": catalog_products,
                }
        
        # Buscar producto por posición resuelta
        if resolved_index:
            product = next((p for p in recent if p["position"] == resolved_index), None)
        
        if not product:
            max_idx = max(p["position"] for p in recent) if recent else 0
            return {
                "handled": True,
                "response": (
                    f"No encontré el producto {resolved_index if resolved_index else 'mencionado'}. "
                    f"En este momento solo tengo disponibles los productos del 1 al {max_idx} "
                    "de la última lista que te mostré."
                ),
                "products": [],
            }

        # Si el modelo indica que necesita confirmación o la confianza es baja, NO modificamos el carrito
        if needs_confirmation or confidence < 0.8:
            response = (
                f"Entiendo que te refieres al Producto {resolved_index}: {product['prod_name']} "
                f"({product['colour_group_name']}). "
                "Solo para confirmar, ¿quieres que agregue ese producto a tu carrito? "
                "Puedes decirme \"sí, agrega el producto "
                f"{resolved_index} al carrito\" o simplemente \"sí\"."
            )
            return {"handled": True, "response": response, "products": []}

        # Confianza alta: procedemos a agregar al carrito
        add_to_cart(conversation_id, product["article_id"], quantity=1)
        
        logger.info(
            f"✅ Producto agregado al carrito - "
            f"ArticleID: {product['article_id']}, "
            f"Nombre: {product['prod_name']}, "
            f"ConvID: {conversation_id}"
        )
        
        response = (
            f"✅ He agregado al carrito:\n\n"
            f"*{product['prod_name']}* ({product['colour_group_name']})\n"
            f"Precio: ${product['price_mxn']:.2f} MXN\n\n"
            f"¿Qué deseas hacer ahora?\n"
            f"• \"Ver carrito\" - Ver todos tus productos\n"
            f"• \"Seguir comprando\" - Buscar más productos\n"
            f"• \"Proceder al pago\" - Finalizar compra"
        )
        return {"handled": True, "response": response, "products": []}

    return {"handled": False}
