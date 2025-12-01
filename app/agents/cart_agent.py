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

        if not index or index <= 0:
            return {
                "handled": True,
                "response": (
                    "Para agregar algo al carrito dime claramente qué producto, por ejemplo: "
                    "\"agrega el producto 1 al carrito\"."
                ),
                "products": [],
            }

        if not recent:
            return {
                "handled": True,
                "response": (
                    "Aún no tengo productos recientes asociados a esta conversación. "
                    "Primero te mostraré algunas opciones y luego podrás decirme, por ejemplo, "
                    "\"agrega el producto 1 al carrito\"."
                ),
                "products": [],
            }

        # Buscar por posición
        product = next((p for p in recent if p["position"] == index), None)
        if not product:
            max_idx = max(p["position"] for p in recent)
            return {
                "handled": True,
                "response": (
                    f"No encontré el producto {index}. "
                    f"En este momento solo tengo disponibles los productos del 1 al {max_idx} "
                    "de la última lista que te mostré."
                ),
                "products": [],
            }

        # Si el modelo indica que necesita confirmación o la confianza es baja, NO modificamos el carrito
        if needs_confirmation or confidence < 0.8:
            response = (
                f"Entiendo que estás hablando del Producto {index}: {product['prod_name']} "
                f"({product['colour_group_name']}). "
                "Solo para confirmar, ¿quieres que agregue ese producto a tu carrito? "
                "Puedes decirme, por ejemplo, \"sí, agrega el producto "
                f"{index} al carrito\"."
            )
            return {"handled": True, "response": response, "products": []}

        # Confianza alta: procedemos a agregar al carrito
        add_to_cart(conversation_id, product["article_id"], quantity=1)
        response = (
            f"He agregado al carrito el Producto {index}: {product['prod_name']} "
            f"({product['colour_group_name']}). "
            "Si quieres, puedo mostrarte tu carrito completo o seguir buscándote más opciones."
        )
        return {"handled": True, "response": response, "products": []}

    return {"handled": False}
