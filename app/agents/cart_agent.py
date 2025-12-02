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
    format_checkout_message_simple,
    calculate_cart_total,
    get_cart_by_conversation,
    get_cart_items_for_display,
    clear_cart as clear_cart_func,
    save_recent_products,
    update_cart_item_quantity,
    remove_cart_items_by_article_ids,
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
    cart_items: Optional[List[Dict[str, Any]]] = None,
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

    # Formatear items del carrito actual
    if cart_items:
        cart_text = "\n".join(
            [
                f"Item {i+1}: {item['prod_name']} ({item['colour_group_name']}) - {item['product_type_name']}"
                for i, item in enumerate(cart_items)
            ]
        )
        cart_section = f"""

PRODUCTOS ACTUALMENTE EN EL CARRITO DEL USUARIO:
{cart_text}
"""
    else:
        cart_section = "\nEl carrito del usuario está vacío."

    # Formatear contexto reciente de conversación
    context_text = _format_conversation_context_for_cart(conversation_context)
    context_section = ""
    if context_text:
        context_section = f"""

CONVERSACIÓN RECIENTE:
{context_text}
"""

    prompt = f"""Eres un agente que interpreta la intención del usuario en una tienda de ropa.

MENSAJE DEL USUARIO:
"{user_message}"

PRODUCTOS RECIENTES (mostrados al usuario):
{products_text}
{cart_section}
{context_section}

REGLAS IMPORTANTES:
1. mode = "none" → Solo si habla de productos SIN mencionar carrito ni querer agregar/quitar

2. mode = "show_cart" → "ver carrito", "qué tengo", "muéstrame el carrito"

3. mode = "add_to_cart" → SOLO si quiere agregar UN producto
   - product_index = número del producto en la lista reciente

4. mode = "remove_from_cart" → SOLO si quiere quitar UN producto
   - Usa palabras: "quita", "elimina", "remove", "saca", "borra"
   - product_index = NULL (el parser avanzado lo maneja)

5. mode = "multi_action" → Si el usuario quiere hacer MÚLTIPLES acciones en una sola petición:
   - "agrega el 1 y el 3" → agregar múltiples productos
   - "quita el 2 y el 4" → eliminar múltiples
   - "agrega la chaqueta azul y los pantalones" → múltiples por descripción
   - "quita los calcetines pero agrega la camisa" → mezcla de agregar Y quitar
   - CLAVE: Si menciona "y", "también", múltiples números separados por coma, o mezcla agregar/quitar → multi_action

6. mode = "checkout" → "pagar", "checkout", "proceder al pago", "comprar", "finalizar compra", "quiero pagar"

7. mode = "continue_shopping" → "seguir comprando", "agregar más", "modificar carrito", "cancelar pago"

8. mode = "clear_cart" → "vaciar carrito", "eliminar todo", "borrar carrito"

9. Si el usuario es ambiguo, establece needs_confirmation = true y confidence menor (0.5).

10. NUNCA inventes productos que no están en la lista.

EJEMPLOS DE MULTI_ACTION:
- "agrega el 1, 2 y 5" → multi_action
- "quita el rojo y agrega el azul" → multi_action
- "agrega la chaqueta y los pantalones" → multi_action
- "el 1 y el 3" → multi_action (asumiendo agregar múltiples)

Formato de respuesta:
Responde SOLO con un JSON válido, sin texto adicional:
{{
  "mode": "none" | "add_to_cart" | "remove_from_cart" | "show_cart" | "checkout" | "continue_shopping" | "clear_cart" | "multi_action",
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


def parse_multi_action_cart_request(
    user_message: str,
    recent_products: List[Dict[str, Any]],
    cart_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Analiza solicitudes que involucran múltiples acciones de carrito:
    - "agrega la chaqueta azul y los pantalones naranjas"
    - "agrega el producto 3 y los últimos jeans que me mostraste"
    - "quita los calcetines verdes pero agrega los azules"
    - "agrega 1, 2 y 5"
    - "quita el 2 pero agrega el 4"
    
    Returns:
        Dict con:
        - has_multi_action: bool
        - items_to_add: List[Dict] con {reference, resolved_product, confidence}
        - items_to_remove: List[Dict] con {reference, resolved_article_id, confidence}
        - description: str
        - needs_confirmation: bool
    """
    # Formatear productos recientes
    if recent_products:
        recent_text = "\n".join([
            f"Producto {p['position']}: {p['prod_name']} ({p['colour_group_name']}) - {p['product_type_name']} - ${p['price_mxn']:.2f}"
            for p in recent_products
        ])
    else:
        recent_text = "No hay productos recientes."
    
    # Formatear carrito actual
    if cart_items:
        cart_text = "\n".join([
            f"Carrito Item {i+1}: article_id={item['article_id']} | {item['prod_name']} ({item['colour_group_name']}) - {item['product_type_name']}"
            for i, item in enumerate(cart_items)
        ])
    else:
        cart_text = "Carrito vacío."
    
    prompt = f"""Analiza este mensaje para identificar TODAS las acciones de carrito que el usuario quiere hacer.

MENSAJE: "{user_message}"

PRODUCTOS RECIENTES (para AGREGAR):
{recent_text}

CARRITO ACTUAL (para QUITAR):
{cart_text}

INSTRUCCIONES:
1. Identifica CADA producto que el usuario quiere AGREGAR (de productos recientes)
2. Identifica CADA producto que el usuario quiere QUITAR (del carrito)
3. El usuario puede mezclar: "agrega X y Y", "quita A pero agrega B", "1, 2 y 5"
4. Referencias pueden ser:
   - Por número: "producto 3", "el 1 y el 2", "1, 3, 5"
   - Por descripción: "la chaqueta azul", "los jeans"
   - Por posición: "el último", "el primero"
   - Mixtas: "producto 3 y la camisa blanca"

Responde SOLO JSON:
{{
  "has_multi_action": true/false,
  "items_to_add": [
    {{"reference": "descripción del usuario", "product_position": número_en_recientes_o_null, "confidence": 0.0-1.0}}
  ],
  "items_to_remove": [
    {{"reference": "descripción del usuario", "cart_item_number": número_en_carrito_o_null, "article_id": "id_o_null", "confidence": 0.0-1.0}}
  ],
  "description": "Resumen de acciones",
  "needs_confirmation": true/false
}}

EJEMPLOS:
- "agrega el 1 y el 3" → items_to_add: [{{product_position: 1}}, {{product_position: 3}}]
- "quita el 2 pero agrega el 4" → items_to_remove: [{{cart_item_number: 2}}], items_to_add: [{{product_position: 4}}]
- "agrega la chaqueta azul y los pantalones" → items_to_add con referencias por descripción"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0.1,
    )
    
    content = response.choices[0].message.content.strip()
    
    try:
        data = json.loads(content)
        return {
            "has_multi_action": data.get("has_multi_action", False),
            "items_to_add": data.get("items_to_add", []),
            "items_to_remove": data.get("items_to_remove", []),
            "description": data.get("description", ""),
            "needs_confirmation": data.get("needs_confirmation", False),
        }
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.error(f"Error parsing multi-action response: {e}")
        return {
            "has_multi_action": False,
            "items_to_add": [],
            "items_to_remove": [],
            "description": "Error procesando solicitud",
            "needs_confirmation": True,
        }


def execute_multi_action_cart(
    conversation_id: str,
    multi_action: Dict[str, Any],
    recent_products: List[Dict[str, Any]],
    cart_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Ejecuta múltiples acciones de carrito (agregar y/o quitar).
    
    Returns:
        Dict con:
        - success: bool
        - added: List[str] nombres de productos agregados
        - removed: List[str] nombres de productos quitados
        - failed: List[str] acciones que fallaron
        - summary: str resumen para el usuario
    """
    added = []
    removed = []
    failed = []
    
    # Procesar items a agregar
    for item_add in multi_action.get("items_to_add", []):
        try:
            position = item_add.get("product_position")
            reference = item_add.get("reference", "")
            
            product = None
            
            # Si tiene posición, buscar por posición
            if position and recent_products:
                product = next((p for p in recent_products if p["position"] == position), None)
            
            # Si no se encontró por posición, buscar por referencia
            if not product and reference and recent_products:
                # Buscar coincidencia parcial en nombre
                ref_lower = reference.lower()
                for p in recent_products:
                    name_lower = p["prod_name"].lower()
                    color_lower = p.get("colour_group_name", "").lower()
                    type_lower = p.get("product_type_name", "").lower()
                    
                    if (ref_lower in name_lower or 
                        name_lower in ref_lower or
                        ref_lower in color_lower or
                        ref_lower in type_lower):
                        product = p
                        break
            
            # Si aún no se encontró y hay referencia, buscar en catálogo
            if not product and reference:
                logger.info(f"🔍 Buscando en catálogo: {reference}")
                catalog_results = search_products(reference)
                if catalog_results:
                    product = catalog_results[0]
                    logger.info(f"📦 Encontrado en catálogo: {product['prod_name']}")
            
            if product:
                add_to_cart(conversation_id, product["article_id"], quantity=1)
                added.append(f"{product['prod_name']} ({product.get('colour_group_name', '')})")
                logger.info(f"✅ Agregado: {product['prod_name']}")
            else:
                failed.append(f"No encontré: {reference or f'producto {position}'}")
                
        except Exception as e:
            logger.error(f"Error agregando item: {e}")
            failed.append(f"Error agregando: {item_add.get('reference', 'producto')}")
    
    # Procesar items a quitar
    for item_remove in multi_action.get("items_to_remove", []):
        try:
            cart_num = item_remove.get("cart_item_number")
            article_id = item_remove.get("article_id")
            reference = item_remove.get("reference", "")
            
            target_article_id = None
            target_name = ""
            
            # Si tiene número de carrito
            if cart_num and cart_items and 1 <= cart_num <= len(cart_items):
                item = cart_items[cart_num - 1]
                target_article_id = item["article_id"]
                target_name = f"{item['prod_name']} ({item['colour_group_name']})"
            
            # Si tiene article_id directo
            elif article_id:
                target_article_id = article_id
                item = next((i for i in cart_items if str(i["article_id"]) == str(article_id)), None)
                if item:
                    target_name = f"{item['prod_name']} ({item['colour_group_name']})"
            
            # Si solo tiene referencia, buscar en carrito
            elif reference and cart_items:
                ref_lower = reference.lower()
                for item in cart_items:
                    name_lower = item["prod_name"].lower()
                    color_lower = item.get("colour_group_name", "").lower()
                    type_lower = item.get("product_type_name", "").lower()
                    
                    if (ref_lower in name_lower or 
                        name_lower in ref_lower or
                        ref_lower in color_lower or
                        ref_lower in type_lower):
                        target_article_id = item["article_id"]
                        target_name = f"{item['prod_name']} ({item['colour_group_name']})"
                        break
            
            if target_article_id:
                remove_from_cart(conversation_id, str(target_article_id))
                removed.append(target_name or f"Item {cart_num}")
                logger.info(f"✅ Quitado: {target_name}")
            else:
                failed.append(f"No encontré en carrito: {reference or f'item {cart_num}'}")
                
        except Exception as e:
            logger.error(f"Error quitando item: {e}")
            failed.append(f"Error quitando: {item_remove.get('reference', 'producto')}")
    
    # Construir resumen
    summary_parts = []
    if added:
        summary_parts.append(f"Agregué: {', '.join(added)}")
    if removed:
        summary_parts.append(f"Quité: {', '.join(removed)}")
    if failed:
        summary_parts.append(f"No pude: {', '.join(failed)}")
    
    return {
        "success": len(added) > 0 or len(removed) > 0,
        "added": added,
        "removed": removed,
        "failed": failed,
        "summary": " | ".join(summary_parts) if summary_parts else "No se realizaron cambios",
    }


def parse_advanced_removal_request(
    user_message: str,
    cart_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Analiza solicitudes complejas de eliminación del carrito como:
    - "quita todos los jeans"
    - "elimina todos los bottoms"
    - "quita cualquier prenda bajo 500"
    - "quita 3 de mis 5 sweaters"
    - "elimina las camisas azules"
    - "quita todo lo que sea rojo"
    
    Returns:
        Dict con:
        - removal_type: "all" | "by_category" | "by_type" | "by_price" | "by_color" | "by_name" | "partial_quantity" | "specific"
        - items_to_remove: List de article_ids a eliminar
        - quantity_changes: Dict de {article_id: new_quantity} para cambios parciales
        - description: Descripción de lo que se va a eliminar
        - confidence: float
        - needs_confirmation: bool
    """
    if not cart_items:
        return {
            "removal_type": "none",
            "items_to_remove": [],
            "quantity_changes": {},
            "description": "El carrito está vacío",
            "confidence": 0.0,
            "needs_confirmation": False,
        }
    
    # Formatear items del carrito con detalles
    cart_text = "\n".join(
        [
            f"Item {i+1}: article_id={item['article_id']} | "
            f"Nombre: {item['prod_name']} | "
            f"Color: {item['colour_group_name']} | "
            f"Tipo: {item['product_type_name']} | "
            f"Categoría: {item['product_group_name']} | "
            f"Precio: ${item['price_mxn']:.2f} MXN | "
            f"Cantidad: {item['quantity']}"
            for i, item in enumerate(cart_items)
        ]
    )
    
    prompt = f"""Eres un agente que interpreta qué productos quiere ELIMINAR el usuario de su carrito.

MENSAJE DEL USUARIO:
"{user_message}"

PRODUCTOS EN EL CARRITO:
{cart_text}

CATEGORÍAS DE ROPA (¡MUY IMPORTANTE!):
- "tops" = CUALQUIER prenda de arriba: Blouse, Blusa, Shirt, Camisa, T-shirt, Camiseta, Sweater, Hoodie, Top, Tank
- "bottoms" = CUALQUIER prenda de abajo: Shorts, Jeans, Pants, Pantalón, Skirt, Falda, Trousers
- "socks/calcetines" = Socks, Ankle Socks, Calcetines
- "outerwear" = Jacket, Coat, Chamarra, Abrigo

FORMAS DE PEDIR ELIMINACIÓN:
1. Por NÚMERO: "2 y 3", "el 1", "1, 2", "quita 2 y 3" → USA LOS NÚMEROS DE ITEM
2. Por NOMBRE: "quita EDC vanilla", "elimina Milli" → BUSCA EN EL NOMBRE
3. Por CATEGORÍA: "quita los tops", "elimina las blusas" → BUSCA EN TIPO
4. Por COLOR: "quita lo blanco" → BUSCA EN COLOR
5. Por PRECIO: "quita lo menor a 500" → COMPARA PRECIO

REGLAS CRÍTICAS:
1. Si el usuario dice "2 y 3" o "1, 2" → son NÚMEROS DE ITEM, elimina Item 2 y Item 3
2. Si dice "tops" y hay "Blouse" en Tipo → ESO ES UN TOP, elimínalo
3. Si dice un nombre parcial como "EDC" y hay "EDC VANILLA BLOUSE" → COINCIDE
4. SIEMPRE devuelve los article_id de los items que coinciden

Responde SOLO JSON:
{{
  "removal_type": "by_number" | "by_name" | "by_category" | "by_color" | "by_price" | "all" | "none",
  "items_to_remove": ["article_id1", "article_id2"],
  "quantity_changes": {{}},
  "description": "Qué se va a eliminar",
  "confidence": 0.0-1.0,
  "needs_confirmation": false
}}

EJEMPLOS:
- Usuario: "2 y 3" con items 1,2,3 → eliminar article_id del Item 2 y Item 3
- Usuario: "tops" con Blouse en carrito → eliminar porque Blouse ES un top
- Usuario: "EDC vanilla blouse" → buscar ese nombre exacto"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.1,
    )
    
    content = response.choices[0].message.content.strip()
    
    try:
        data = json.loads(content)
        
        return {
            "removal_type": data.get("removal_type", "none"),
            "items_to_remove": data.get("items_to_remove", []),
            "quantity_changes": data.get("quantity_changes", {}),
            "description": data.get("description", ""),
            "matched_items_summary": data.get("matched_items_summary", ""),
            "confidence": float(data.get("confidence", 0.0)),
            "needs_confirmation": bool(data.get("needs_confirmation", True)),
        }
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.error(f"Error parsing advanced removal response: {e}")
        return {
            "removal_type": "none",
            "items_to_remove": [],
            "quantity_changes": {},
            "description": "Error procesando solicitud",
            "matched_items_summary": "",
            "confidence": 0.0,
            "needs_confirmation": True,
        }


def execute_cart_removal(
    conversation_id: str,
    removal_result: Dict[str, Any],
    cart_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Ejecuta la eliminación de productos del carrito basado en el resultado del parsing.
    
    Returns:
        Dict con:
        - success: bool
        - items_removed: int
        - items_updated: int
        - details: str (descripción de lo que se hizo)
    """
    items_removed = 0
    items_updated = 0
    removed_names = []
    updated_names = []
    
    # Procesar eliminaciones completas
    items_to_remove = removal_result.get("items_to_remove", [])
    if items_to_remove:
        # Find names for removed items
        for article_id in items_to_remove:
            item = next((i for i in cart_items if str(i["article_id"]) == str(article_id)), None)
            if item:
                removed_names.append(f"{item['prod_name']} ({item['colour_group_name']})")
        
        count = remove_cart_items_by_article_ids(conversation_id, items_to_remove)
        items_removed = count
    
    # Procesar cambios de cantidad
    quantity_changes = removal_result.get("quantity_changes", {})
    for article_id, new_quantity in quantity_changes.items():
        item = next((i for i in cart_items if str(i["article_id"]) == str(article_id)), None)
        if item:
            old_qty = item["quantity"]
            if update_cart_item_quantity(conversation_id, str(article_id), int(new_quantity)):
                items_updated += 1
                updated_names.append(
                    f"{item['prod_name']}: {old_qty} → {new_quantity}"
                )
    
    # Build details message
    details_parts = []
    if removed_names:
        details_parts.append(f"Eliminados: {', '.join(removed_names)}")
    if updated_names:
        details_parts.append(f"Actualizados: {', '.join(updated_names)}")
    
    details = " | ".join(details_parts) if details_parts else "No se realizaron cambios"
    
    return {
        "success": items_removed > 0 or items_updated > 0,
        "items_removed": items_removed,
        "items_updated": items_updated,
        "removed_names": removed_names,
        "updated_names": updated_names,
        "details": details,
        }


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
    
    # Get current cart items for smarter intent detection
    current_cart = get_cart(conversation_id)
    
    intent = detect_cart_intent_llm(
        user_message, 
        recent, 
        conversation_context=conversation_context,
        cart_items=current_cart
    )
    mode = intent["mode"]
    
    logger.info(
        f"🛒 [CART-AGENT] Intent detectado - "
        f"Mode: {mode}, "
        f"ConvID: {conversation_id}, "
        f"Confidence: {intent.get('confidence', 0.0):.2f}, "
        f"Cart items: {len(current_cart) if current_cart else 0}"
    )

    if mode == "none":
        logger.debug("➡️ Cart agent no maneja este mensaje, pasando a agente normal")
        return {"handled": False}

    # Ver carrito
    if mode == "show_cart":
        cart_items = get_cart(conversation_id)
        if not cart_items:
            response = (
                "Tu carrito está vacío por ahora 🛒\n\n"
                "Cuando encuentres algo que te guste, solo dime y lo agrego."
            )
            return {"handled": True, "response": response, "products": [], "send_images": False}

        # Calcular total
        cart_id = get_cart_by_conversation(conversation_id)
        total = calculate_cart_total(cart_id) if cart_id else 0.0
        
        # Get cart items with normalized fields for image display
        display_items = get_cart_items_for_display(conversation_id)
        
        # Header message (images will be sent separately)
        response = "🛒 *TU CARRITO ACTUAL*\n\nEstos son los productos en tu carrito:"
        
        # Return cart items for image display
        return {
            "handled": True,
            "response": response, 
            "products": [],
            "send_images": True,
            "image_type": "cart",
            "cart_items": display_items,
            "cart_total": total
        }
    
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
            return {"handled": True, "response": response, "products": [], "send_images": False}

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
            return {"handled": True, "response": response, "products": [], "send_images": False}

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
            return {"handled": True, "response": response, "products": [], "send_images": False}

        # Éxito - formatear mensaje con link de pago
        logger.info(
            f"✅ Checkout creado exitosamente - "
            f"SessionID: {checkout_result['session_id']}, "
            f"Total: ${checkout_result['total_amount']:.2f} MXN, "
            f"Items: {checkout_result['items_count']}"
        )
        
        # Get cart items for image display
        display_items = get_cart_items_for_display(conversation_id)
        
        # Header for checkout with images
        response = "🛒 *RESUMEN DE TU ORDEN*\n\nEstos son los productos que vas a comprar:"
        
        # NOTE: Cart is NOT cleared here - it stays until payment is confirmed
        # This allows the user to go back and modify the cart
        
        return {
            "handled": True,
            "response": response, 
            "products": [],
            "send_images": True,
            "image_type": "checkout",
            "cart_items": display_items,
            "cart_total": checkout_result["total_amount"],
            "checkout_url": checkout_result["checkout_url"]
        }
    
    # Seguir comprando / Modificar carrito (después de checkout)
    if mode == "continue_shopping":
        logger.info(f"🛍️ Usuario decidió seguir comprando - ConvID: {conversation_id}")
        
        # IMPORTANT: Cart is NOT cleared here - the user can continue shopping
        # and add more items, then checkout again with the updated cart
        
        # Check if user has items in cart to remind them
        cart_items = get_cart(conversation_id)
        cart_count = len(cart_items) if cart_items else 0
        
        if cart_count > 0:
            cart_id = get_cart_by_conversation(conversation_id)
            total = calculate_cart_total(cart_id) if cart_id else 0.0
            
            response = (
                f"¡Claro! Tu carrito sigue guardado con {cart_count} producto(s) "
                f"(${total:.2f} MXN) 🛒\n\n"
                f"¿Qué más te gustaría buscar? Puedes pedirme lo que necesites "
                f"o decirme cuando quieras pagar."
            )
        else:
            response = (
                "¡Perfecto! ¿Qué te gustaría buscar? 🛍️\n\n"
                "Dime qué tienes en mente y te ayudo a encontrarlo."
            )
        
        return {"handled": True, "response": response, "products": [], "send_images": False}
    
    # Vaciar carrito completamente
    if mode == "clear_cart":
        logger.info(f"🗑️ Usuario solicita vaciar carrito - ConvID: {conversation_id}")
        
        cart_id = get_cart_by_conversation(conversation_id)
        
        if not cart_id:
            logger.debug("Carrito ya vacío (no existe cart_id)")
            response = "Tu carrito ya está vacío. 🛒"
            return {"handled": True, "response": response, "products": [], "send_images": False}
        
        cart_items = get_cart(conversation_id)
        
        if not cart_items:
            logger.debug("Carrito ya vacío (sin items)")
            response = "Tu carrito ya está vacío. 🛒"
            return {"handled": True, "response": response, "products": [], "send_images": False}
        
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
            return {"handled": True, "response": response, "products": [], "send_images": False}
        
        # Vaciar el carrito
        items_count = len(cart_items)
        clear_cart_func(cart_id)
        logger.info(f"✅ Carrito vaciado - {items_count} items eliminados")
        
        response = (
            f"Listo, vacié tu carrito ({items_count} productos eliminados). 🛒\n\n"
            f"¿Empezamos de nuevo? Dime qué te gustaría buscar."
        )
        return {"handled": True, "response": response, "products": [], "send_images": False}

    # Quitar del carrito - VERSIÓN AVANZADA
    # Soporta: por nombre, categoría, tipo, color, precio, cantidad parcial
    if mode == "remove_from_cart":
        logger.info(f"➖ Usuario quiere quitar del carrito - ConvID: {conversation_id}")
        
        cart_items = get_cart(conversation_id)
        if not cart_items:
            logger.debug("Intento de quitar de carrito vacío")
            return {
                "handled": True,
                "response": "Tu carrito está vacío, no hay nada que quitar. 🛒",
                "products": [],
                "send_images": False,
            }
        
        # Usar el parser avanzado para entender qué quiere eliminar el usuario
        removal_result = parse_advanced_removal_request(user_message, cart_items)
        
        logger.info(
            f"🔍 Análisis de eliminación - "
            f"Tipo: {removal_result['removal_type']}, "
            f"Items a eliminar: {len(removal_result['items_to_remove'])}, "
            f"Cambios de cantidad: {len(removal_result['quantity_changes'])}, "
            f"Confianza: {removal_result['confidence']:.2f}"
        )
        
        # Si no se encontró nada que eliminar
        if removal_result["removal_type"] == "none" or (
            not removal_result["items_to_remove"] and not removal_result["quantity_changes"]
        ):
            # Fallback: mostrar el carrito para que el usuario pueda especificar
            cart_list = []
            for i, item in enumerate(cart_items, start=1):
                item_type = item.get('product_type_name', '')
                cart_list.append(
                    f"{i}. *{item['prod_name']}* ({item['colour_group_name']}) - {item_type} - "
                    f"${item['price_mxn']:.2f} MXN x{item['quantity']}"
                )
            cart_text = "\n".join(cart_list)
            
            response = (
                f"Mmm, no encontré eso en tu carrito. 🤔\n\n"
                f"Esto es lo que tienes:\n{cart_text}\n\n"
                f"¿Cuál quieres que quite?"
            )
            return {"handled": True, "response": response, "products": [], "send_images": False}
        
        # Si necesita confirmación (confianza baja o eliminación masiva)
        items_to_remove_count = len(removal_result["items_to_remove"])
        quantity_changes_count = len(removal_result["quantity_changes"])
        total_affected = items_to_remove_count + quantity_changes_count
        
        needs_confirmation = (
            removal_result["needs_confirmation"] or 
            removal_result["confidence"] < 0.75 or
            total_affected > 2  # Pedir confirmación si se afectan más de 2 items
        )
        
        if needs_confirmation:
            # Construir mensaje de confirmación
            affected_items = []
            
            for article_id in removal_result["items_to_remove"]:
                item = next((i for i in cart_items if str(i["article_id"]) == str(article_id)), None)
                if item:
                    affected_items.append(f"- {item['prod_name']} ({item['colour_group_name']})")
            
            for article_id, new_qty in removal_result["quantity_changes"].items():
                item = next((i for i in cart_items if str(i["article_id"]) == str(article_id)), None)
                if item:
                    qty_to_remove = item['quantity'] - int(new_qty)
                    affected_items.append(
                        f"- {qty_to_remove} de tus {item['quantity']} {item['prod_name']}"
                    )
            
            affected_text = "\n".join(affected_items) if affected_items else removal_result.get("description", "productos")
            
            response = (
                f"Voy a quitar esto de tu carrito:\n\n"
                f"{affected_text}\n\n"
                f"¿Está bien?"
            )
            
            # Store the pending removal for confirmation
            # For now, we'll just return and let the user confirm with a clear statement
            return {"handled": True, "response": response, "products": [], "send_images": False}
        
        # Confianza alta: ejecutar la eliminación
        execution_result = execute_cart_removal(conversation_id, removal_result, cart_items)
        
        logger.info(
            f"✅ Eliminación ejecutada - "
            f"Items eliminados: {execution_result['items_removed']}, "
            f"Items actualizados: {execution_result['items_updated']}, "
            f"Detalles: {execution_result['details']}"
        )
        
        # Construir mensaje de respuesta natural
        removed_count = len(execution_result["removed_names"])
        updated_count = len(execution_result["updated_names"])
        
        if removed_count == 1 and updated_count == 0:
            removed_name = execution_result["removed_names"][0]
            response = f"Listo, quité {removed_name} de tu carrito. ✅"
        elif removed_count > 1:
            response = f"Listo, quité {removed_count} productos de tu carrito. ✅"
        elif updated_count > 0:
            response = f"Listo, actualicé las cantidades. ✅"
        else:
            response = "Listo, hice los cambios. ✅"
        
        # Mostrar resumen del carrito actualizado
        updated_cart = get_cart(conversation_id)
        if updated_cart:
            cart_id = get_cart_by_conversation(conversation_id)
            new_total = calculate_cart_total(cart_id) if cart_id else 0.0
            response += f"\n\nTu carrito ahora tiene {len(updated_cart)} producto(s) (${new_total:.2f} MXN)."
        else:
            response += "\n\nTu carrito quedó vacío."
        
        response += "\n\n¿Algo más?"
        
        return {"handled": True, "response": response, "products": [], "send_images": False}

    # Múltiples acciones de carrito en una sola petición
    if mode == "multi_action":
        logger.info(f"🔄 Usuario quiere realizar múltiples acciones - ConvID: {conversation_id}")
        
        # Obtener estado actual
        current_cart_items = get_cart(conversation_id)
        
        # Parsear la solicitud multi-acción
        multi_result = parse_multi_action_cart_request(
            user_message,
            recent,
            current_cart_items or []
        )
        
        logger.info(
            f"📋 Multi-acción parseada - "
            f"Agregar: {len(multi_result['items_to_add'])}, "
            f"Quitar: {len(multi_result['items_to_remove'])}, "
            f"Confirmación: {multi_result['needs_confirmation']}"
        )
        
        if not multi_result["has_multi_action"]:
            # No se detectó multi-acción clara, mostrar ayuda
            return {
                "handled": True,
                "response": (
                    "No entendí bien qué productos quieres modificar. 🤔\n\n"
                    "Puedes decirme por ejemplo:\n"
                    "• \"agrega el 1 y el 3\"\n"
                    "• \"quita los pantalones y agrega la camisa\"\n"
                    "• \"agrega la chaqueta azul y los zapatos\""
                ),
                "products": [],
                "send_images": False,
            }
        
        # Verificar que haya productos recientes para agregar
        if multi_result["items_to_add"] and not recent:
            return {
                "handled": True,
                "response": (
                    "No te he mostrado productos todavía. 🔍\n\n"
                    "Dime qué buscas y te muestro opciones para agregar."
                ),
                "products": [],
                "send_images": False,
            }
        
        # Verificar que haya items en carrito para quitar
        if multi_result["items_to_remove"] and not current_cart_items:
            return {
                "handled": True,
                "response": (
                    "Tu carrito está vacío, no hay nada que quitar. 🛒\n\n"
                    "¿Quieres que agregue los productos que mencionaste?"
                ),
                "products": [],
                "send_images": False,
            }
        
        # Si necesita confirmación por ambigüedad
        if multi_result["needs_confirmation"]:
            desc = multi_result.get("description", "realizar estos cambios")
            return {
                "handled": True,
                "response": (
                    f"Entiendo que quieres: {desc}\n\n"
                    f"¿Es correcto? Dime 'sí' para confirmar."
                ),
                "products": [],
                "send_images": False,
            }
        
        # Ejecutar las acciones
        execution_result = execute_multi_action_cart(
            conversation_id,
            multi_result,
            recent,
            current_cart_items or []
        )
        
        logger.info(
            f"✅ Ejecución multi-acción - "
            f"Agregados: {len(execution_result['added'])}, "
            f"Quitados: {len(execution_result['removed'])}, "
            f"Fallidos: {len(execution_result['failed'])}"
        )
        
        # Construir respuesta natural
        response_parts = []
        
        if execution_result["added"]:
            if len(execution_result["added"]) == 1:
                response_parts.append(f"✅ Agregué {execution_result['added'][0]}")
            else:
                added_text = ", ".join(execution_result["added"])
                response_parts.append(f"✅ Agregué: {added_text}")
        
        if execution_result["removed"]:
            if len(execution_result["removed"]) == 1:
                response_parts.append(f"✅ Quité {execution_result['removed'][0]}")
            else:
                removed_text = ", ".join(execution_result["removed"])
                response_parts.append(f"✅ Quité: {removed_text}")
        
        if execution_result["failed"]:
            failed_text = ", ".join(execution_result["failed"])
            response_parts.append(f"⚠️ {failed_text}")
        
        if not response_parts:
            response = "Mmm, no pude hacer esos cambios. ¿Puedes ser más específico?"
        else:
            response = "\n".join(response_parts)
        
        # Mostrar resumen del carrito actualizado
        updated_cart = get_cart(conversation_id)
        if updated_cart:
            cart_id = get_cart_by_conversation(conversation_id)
            new_total = calculate_cart_total(cart_id) if cart_id else 0.0
            response += f"\n\n🛒 Carrito: {len(updated_cart)} producto(s) - ${new_total:.2f} MXN"
        else:
            response += "\n\n🛒 Tu carrito quedó vacío."
        
        response += "\n\n¿Algo más?"
        
        # Return with images if products were added
        if execution_result["added"]:
            display_items = get_cart_items_for_display(conversation_id)
            return {
                "handled": True,
                "response": response,
                "products": [],
                "send_images": True,
                "image_type": "cart",
                "cart_items": display_items,
                "cart_total": calculate_cart_total(get_cart_by_conversation(conversation_id)) if get_cart_by_conversation(conversation_id) else 0.0
            }
        
        return {"handled": True, "response": response, "products": [], "send_images": False}

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
                    "Todavía no te he mostrado productos. 🔍\n\n"
                    "Dime qué estás buscando y te muestro opciones."
                ),
                "products": [],
                "send_images": False,
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
                    products_list = [f"{p['prod_name']}" for p in recent[:5]]
                    products_text = ", ".join(products_list)
                    
                    return {
                        "handled": True,
                        "response": (
                            f"No estoy seguro cuál quieres agregar. 🤔\n\n"
                            f"Te mostré: {products_text}\n\n"
                            f"¿Cuál te interesa?"
                        ),
                        "products": [],
                        "send_images": False,
                    }

                # Buscar en el catálogo: esto SOLO devuelve productos que realmente existen
                catalog_products = search_products(search_query)

                if not catalog_products:
                    # No match found - stop looking and inform user
                    return {
                        "handled": True,
                        "response": (
                            "No encontré ese producto en nuestro catálogo. 😕\n\n"
                            "¿Quieres que busque algo parecido?"
                        ),
                        "products": [],
                        "send_images": False,
                    }

                # Guardar estos productos como recientes para futuras referencias (Producto 1, etc.)
                save_recent_products(conversation_id, catalog_products)

                # Mostrar al usuario la mejor coincidencia
                best = catalog_products[0]
                response = (
                    f"Encontré esto que podría ser lo que buscas:\n\n"
                    f"*{best['prod_name']}* ({best['colour_group_name']}) - ${best['price_mxn']:.2f} MXN\n\n"
                    f"¿Lo agrego a tu carrito?"
                )

                # Send the found product with image
                return {
                    "handled": True,
                    "response": response,
                    "products": catalog_products,
                    "send_images": True,
                    "image_type": "search",
                    "search_products": catalog_products,
                }
        
        # Buscar producto por posición resuelta (esto se ejecuta si index existe o si se resolvió la referencia)
        if resolved_index:
            product = next((p for p in recent if p["position"] == resolved_index), None)
        
        if not product:
            max_idx = max(p["position"] for p in recent) if recent else 0
            return {
                "handled": True,
                "response": (
                    f"No encontré ese producto en la lista. Solo tengo {max_idx} productos para mostrarte. "
                    f"¿Cuál te interesa?"
                ),
                "products": [],
                "send_images": False,
            }

        # Si el modelo indica que necesita confirmación o la confianza es baja
        if needs_confirmation or confidence < 0.8:
            response = (
                f"¿Te refieres a *{product['prod_name']}* ({product['colour_group_name']})?\n\n"
                f"¿Lo agrego?"
            )
            return {"handled": True, "response": response, "products": [], "send_images": False}

        # Confianza alta: procedemos a agregar al carrito
        add_to_cart(conversation_id, product["article_id"], quantity=1)
        
        logger.info(
            f"✅ Producto agregado al carrito - "
            f"ArticleID: {product['article_id']}, "
            f"Nombre: {product['prod_name']}, "
            f"ConvID: {conversation_id}"
        )
        
        # Show the added product with image
        response = f"¡Agregado! ✅"
        
        added_product = {
            "prod_name": product['prod_name'],
            "colour_group_name": product['colour_group_name'],
            "price_mxn": product['price_mxn'],
            "image_url": product.get('image_url', ''),
            "product_type_name": product.get('product_type_name', ''),
            "product_group_name": product.get('product_group_name', ''),
        }
        
        return {
            "handled": True, 
            "response": response, 
            "products": [],
            "send_images": True,
            "image_type": "added_to_cart",
            "added_product": added_product,
        }

    return {"handled": False}
