import json
import os
from typing import Dict, Any, List

from openai import OpenAI

from ..cart import (
    add_to_cart,
    get_cart,
    get_recent_products,
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def detect_cart_intent_llm(
    user_message: str,
    recent_products: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Agente LLM que detecta la intención del usuario respecto al carrito.

    Retorna un dict con:
    - mode: 'none' | 'add_to_cart' | 'show_cart'
    - product_index: int | None  (para 'add_to_cart', índice de la lista mostrada)
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

    prompt = f"""Eres un agente que interpreta la intención del usuario en una tienda de ropa (CedaMoney).

Tu tarea es decidir si el usuario quiere:
- SOLO hablar o preguntar sobre productos (información general)
- VER su carrito actual
- AGREGAR un producto específico de la lista reciente al carrito

MENSAJE DEL USUARIO:
"{user_message}"

PRODUCTOS RECIENTES MOSTRADOS AL USUARIO:
{products_text}

REGLAS:
1. Si el usuario solo describe gustos o hace preguntas sobre productos (por ejemplo: "tienes camisas verdes", "qué opinas de esta playera"), pero NO habla explícitamente de carrito o de comprar/agregar, entonces:
   - mode = "none"
2. Si el usuario quiere ver su carrito ("ver carrito", "qué tengo en el carrito", "muéstrame el carrito", etc.):
   - mode = "show_cart"
3. Si el usuario quiere AGREGAR un producto de la lista reciente al carrito:
   - mode = "add_to_cart"
   - product_index = número del producto en la lista reciente (1, 2, 3, ...) SI se puede inferir con claridad.
4. Si el usuario es ambiguo (no está claro si quiere agregar o solo hablar), establece:
   - needs_confirmation = true
   - confidence menor (por ejemplo 0.5)
   y NO asumas que realmente quiere agregar algo al carrito.
5. NUNCA inventes productos que no están en la lista reciente para agregar al carrito.

Formato de respuesta:
Responde SOLO con un JSON válido, sin texto adicional, con el siguiente esquema:
{{
  "mode": "none" | "add_to_cart" | "show_cart",
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
    
    prompt = f"""Eres un agente experto que resuelve referencias a productos en una conversación de tienda de ropa.

El usuario mencionó algo sobre un producto en este mensaje:
"{user_message}"

PRODUCTOS RECIENTES DISPONIBLES:
{products_text}

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


def handle_cart_interaction(
    conversation_id: str,
    user_message: str,
) -> Dict[str, Any]:
    """
    Maneja la lógica de carrito si aplica.
    
    Devuelve:
    - handled: bool  -> True si ya se generó una respuesta y no se debe seguir con el flujo normal
    - response: str  -> Mensaje para el usuario (si handled=True)
    - products: list -> Lista de productos a devolver (opcional, normalmente vacía)
    """
    recent = get_recent_products(conversation_id)
    intent = detect_cart_intent_llm(user_message, recent)
    mode = intent["mode"]

    if mode == "none":
        return {"handled": False}

    # Ver carrito
    if mode == "show_cart":
        cart_items = get_cart(conversation_id)
        if not cart_items:
            response = (
                "Por ahora tu carrito está vacío. "
                "Cuando veas productos que te gusten, puedes decirme por ejemplo "
                "\"agrega el producto 1 al carrito\"."
            )
            return {"handled": True, "response": response, "products": []}

        lines = []
        total = 0.0
        for i, item in enumerate(cart_items, start=1):
            line_total = float(item["price_mxn"]) * int(item["quantity"])
            total += line_total
            lines.append(
                f"Producto {i}: {item['prod_name']} ({item['colour_group_name']}) x{item['quantity']} - "
                f"${line_total:.2f} MXN"
            )

        lines.append(f"Total aproximado: ${total:.2f} MXN")
        response = (
            "Este es el resumen de tu carrito actual:\n\n" + "\n".join(lines) +
            "\n\nSi quieres quitar algo o cambiar cantidades, dime qué producto quieres modificar."
        )
        return {"handled": True, "response": response, "products": []}

    # Agregar al carrito usando posición de producto reciente
    if mode == "add_to_cart":
        index = intent.get("product_index")
        needs_confirmation = intent.get("needs_confirmation", False)
        confidence = float(intent.get("confidence", 0.0))

        if not recent:
            return {
                "handled": True,
                "response": (
                    "Aún no tengo productos recientes asociados a esta conversación. "
                    "Primero te mostraré algunas opciones y luego podrás decirme, por ejemplo, "
                    "\"agrega el producto 1 al carrito\" o \"agrega el suéter blanco\"."
                ),
                "products": [],
            }

        product = None
        resolved_index = index
        
        # Si no hay índice numérico, intentar resolver por descripción
        if not index or index <= 0:
            reference_result = resolve_product_reference(user_message, recent)
            
            if reference_result["resolved"]:
                resolved_index = reference_result["product_index"]
                confidence = reference_result["confidence"]
                # Si la confianza de la resolución es baja, necesita confirmación
                if confidence < 0.7:
                    needs_confirmation = True
            else:
                # No se pudo resolver la referencia
                max_idx = max(p["position"] for p in recent) if recent else 0
                # Construir lista de productos disponibles
                products_list = []
                for p in recent[:5]:
                    products_list.append(
                        f"Producto {p['position']}: {p['prod_name']} ({p['colour_group_name']})"
                    )
                products_text = ", ".join(products_list)
                
                return {
                    "handled": True,
                    "response": (
                        f"No pude identificar exactamente qué producto quieres agregar. "
                        f"Puedes referirte a los productos por número (del 1 al {max_idx}) "
                        "o por descripción (ej: \"el suéter blanco\", \"esa camisa verde\"). "
                        f"Productos disponibles: {products_text}"
                    ),
                    "products": [],
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
        response = (
            f"He agregado al carrito el Producto {resolved_index}: {product['prod_name']} "
            f"({product['colour_group_name']}). "
            "Si quieres, puedo mostrarte tu carrito completo o seguir buscándote más opciones."
        )
        return {"handled": True, "response": response, "products": []}

    return {"handled": False}
