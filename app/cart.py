from typing import List, Dict, Any, Optional
from .db import get_connection
import stripe
import os
import logging

logger = logging.getLogger(__name__)

# Initialize Stripe
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")


def _get_or_create_cart(conversation_id: str) -> int:
    """
    Obtiene el ID de carrito para una conversación.
    Si no existe, lo crea.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT id FROM carts WHERE conversation_id = ?",
        (conversation_id,),
    )
    row = cur.fetchone()

    if row:
        cart_id = row[0]
    else:
        cur.execute(
            "INSERT INTO carts (conversation_id) VALUES (?)",
            (conversation_id,),
        )
        cart_id = cur.lastrowid

    conn.commit()
    conn.close()
    return cart_id


def add_to_cart(conversation_id: str, article_id: str, quantity: int = 1) -> None:
    """
    Agrega un producto al carrito del usuario.
    Si ya existe, incrementa la cantidad.
    """
    if quantity <= 0:
        return

    cart_id = _get_or_create_cart(conversation_id)
    conn = get_connection()
    cur = conn.cursor()

    # Verificar si el producto ya está en el carrito
    cur.execute(
        """
        SELECT id, quantity FROM cart_items
        WHERE cart_id = ? AND article_id = ?
        """,
        (cart_id, article_id),
    )
    row = cur.fetchone()

    if row:
        item_id, current_qty = row
        new_qty = current_qty + quantity
        cur.execute(
            "UPDATE cart_items SET quantity = ? WHERE id = ?",
            (new_qty, item_id),
        )
    else:
        cur.execute(
            """
            INSERT INTO cart_items (cart_id, article_id, quantity)
            VALUES (?, ?, ?)
            """,
            (cart_id, article_id, quantity),
        )

    conn.commit()
    conn.close()


def remove_from_cart(conversation_id: str, article_id: str) -> None:
    """
    Elimina un producto del carrito del usuario.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT id FROM carts WHERE conversation_id = ?",
        (conversation_id,),
    )
    row = cur.fetchone()
    if not row:
        conn.close()
        return

    cart_id = row[0]
    cur.execute(
        "DELETE FROM cart_items WHERE cart_id = ? AND article_id = ?",
        (cart_id, article_id),
    )

    conn.commit()
    conn.close()


def clear_cart(conversation_id: str) -> None:
    """
    Vacía por completo el carrito del usuario.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT id FROM carts WHERE conversation_id = ?",
        (conversation_id,),
    )
    row = cur.fetchone()
    if not row:
        conn.close()
        return

    cart_id = row[0]
    cur.execute(
        "DELETE FROM cart_items WHERE cart_id = ?",
        (cart_id,),
    )

    conn.commit()
    conn.close()


def get_cart(conversation_id: str) -> List[Dict[str, Any]]:
    """
    Obtiene el carrito completo del usuario, con información de productos.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT id FROM carts WHERE conversation_id = ?",
        (conversation_id,),
    )
    row = cur.fetchone()
    if not row:
        conn.close()
        return []

    cart_id = row[0]
    cur.execute(
        """
        SELECT 
            ci.article_id,
            ci.quantity,
            p.prod_name,
            p.product_type_name,
            p.product_group_name,
            p.colour_group_name,
            p.detail_desc,
            p.price_mxn,
            p.image_url
        FROM cart_items ci
        JOIN products p ON ci.article_id = p.article_id
        WHERE ci.cart_id = ?
        """,
        (cart_id,),
    )

    items: List[Dict[str, Any]] = []
    for (
        article_id,
        quantity,
        prod_name,
        product_type_name,
        product_group_name,
        colour_group_name,
        detail_desc,
        price_mxn,
        image_url,
    ) in cur.fetchall():
        items.append(
            {
                "article_id": article_id,
                "quantity": quantity,
                "prod_name": prod_name,
                "product_type_name": product_type_name,
                "product_group_name": product_group_name,
                "colour_group_name": colour_group_name,
                "detail_desc": detail_desc,
                "price_mxn": price_mxn,
                "image_url": image_url,
            }
        )

    conn.close()
    return items


def save_recent_products(conversation_id: str, products: List[Dict[str, Any]]) -> None:
    """
    Guarda la lista de productos recientemente mostrados al usuario
    para poder mapear referencias como \"Producto 1\", \"Producto 2\", etc.
    """
    conn = get_connection()
    cur = conn.cursor()

    # Limpiar anteriores
    cur.execute(
        "DELETE FROM recent_products WHERE conversation_id = ?",
        (conversation_id,),
    )

    for idx, product in enumerate(products, start=1):
        article_id = str(product.get("article_id"))
        cur.execute(
            """
            INSERT INTO recent_products (conversation_id, article_id, position)
            VALUES (?, ?, ?)
            """,
            (conversation_id, article_id, idx),
        )

    conn.commit()
    conn.close()


def get_recent_products(conversation_id: str) -> List[Dict[str, Any]]:
    """
    Obtiene los últimos productos mostrados al usuario (con su posición).
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT 
            rp.position,
            p.article_id,
            p.prod_name,
            p.product_type_name,
            p.product_group_name,
            p.colour_group_name,
            p.detail_desc,
            p.price_mxn,
            p.image_url
        FROM recent_products rp
        JOIN products p ON rp.article_id = p.article_id
        WHERE rp.conversation_id = ?
        ORDER BY rp.position ASC
        """,
        (conversation_id,),
    )

    results: List[Dict[str, Any]] = []
    for (
        position,
        article_id,
        prod_name,
        product_type_name,
        product_group_name,
        colour_group_name,
        detail_desc,
        price_mxn,
        image_url,
    ) in cur.fetchall():
        results.append(
            {
                "position": position,
                "article_id": article_id,
                "prod_name": prod_name,
                "product_type_name": product_type_name,
                "product_group_name": product_group_name,
                "colour_group_name": colour_group_name,
                "detail_desc": detail_desc,
                "price_mxn": price_mxn,
                "image_url": image_url,
            }
        )

    conn.close()
    return results

def get_cart_by_conversation(conversation_id: str) -> Optional[int]:
    """
    Obtiene el ID del carrito por conversation_id
    Retorna None si no existe
    """
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute("""
        SELECT id FROM carts 
        WHERE conversation_id = ?
    """, [conversation_id])
    
    result = cur.fetchone()
    conn.close()
    
    return result[0] if result else None


def get_cart_items(cart_id: int) -> List[Dict]:
    """
    Obtiene todos los items del carrito con información del producto
    """
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute("""
        SELECT 
            ci.id,
            ci.article_id,
            ci.quantity,
            p.prod_name,
            p.price_mxn,
            p.image_url,
            p.product_type_name,
            p.colour_group_name
        FROM cart_items ci
        JOIN products p ON ci.article_id = p.article_id
        WHERE ci.cart_id = ?
        ORDER BY ci.added_at DESC
    """, [cart_id])
    
    items = []
    for row in cur.fetchall():
        items.append({
            "cart_item_id": row[0],
            "article_id": row[1],
            "quantity": row[2],
            "name": row[3],
            "price": row[4],
            "image_url": row[5],
            "type": row[6],
            "color": row[7],
            "subtotal": row[2] * row[4]  # quantity * price
        })
    
    conn.close()
    return items


def calculate_cart_total(cart_id: int) -> float:
    """
    Calcula el total del carrito
    """
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute("""
        SELECT SUM(ci.quantity * p.price_mxn) as total
        FROM cart_items ci
        JOIN products p ON ci.article_id = p.article_id
        WHERE ci.cart_id = ?
    """, [cart_id])
    
    result = cur.fetchone()
    conn.close()
    
    return result[0] if result and result[0] else 0.0


def clear_cart(cart_id: int):
    """
    Vacía el carrito después de completar la compra
    """
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute("DELETE FROM cart_items WHERE cart_id = ?", [cart_id])
    
    conn.commit()
    conn.close()


def create_stripe_checkout_for_whatsapp(
    conversation_id: str,
    user_name: str,
    phone_number: str
) -> Dict[str, Any]:
    """
    Crea una sesión de checkout de Stripe específicamente para usuarios de WhatsApp.
    
    Args:
        conversation_id: ID único de la conversación
        user_name: Nombre del usuario (de WhatsApp)
        phone_number: Teléfono del usuario (de WhatsApp)
        
    Returns:
        Dict con checkout_url, session_id, total, items_count, o error
    """
    try:
        # 1. Validar que Stripe esté configurado
        if not stripe.api_key:
            logger.error("❌ STRIPE_SECRET_KEY no configurado")
            return {
                "success": False,
                "error": "Sistema de pagos no configurado. Contacta al administrador."
            }
        
        # 2. Obtener carrito
        cart_id = get_cart_by_conversation(conversation_id)
        if not cart_id:
            logger.warning(f"⚠️ No existe carrito para: {conversation_id}")
            return {
                "success": False,
                "error": "No se encontró un carrito. Primero agrega algunos productos."
            }
        
        # 3. Obtener items del carrito
        cart_items = get_cart_items(cart_id)
        if not cart_items:
            logger.warning(f"⚠️ Carrito vacío para: {conversation_id}")
            return {
                "success": False,
                "error": "Tu carrito está vacío. Agrega algunos productos antes de pagar."
            }
        
        # 4. Calcular total
        total_amount = calculate_cart_total(cart_id)
        
        logger.info(
            f"💳 Creando checkout Stripe - Usuario: {user_name}, "
            f"Phone: {phone_number}, Items: {len(cart_items)}, Total: ${total_amount:.2f}"
        )
        
        # 5. Crear line_items para Stripe
        line_items = []
        for item in cart_items:
            unit_amount = int(item['price'] * 100)  # Convertir a centavos
            
            line_items.append({
                'price_data': {
                    'currency': 'mxn',
                    'product_data': {
                        'name': item['name'],
                        'description': f"{item['type']} - {item['color']}" if item.get('color') else item['type'],
                        'images': [item['image_url']] if item.get('image_url') else [],
                    },
                    'unit_amount': unit_amount,
                },
                'quantity': item['quantity'],
            })
        
        # 6. Crear sesión de Stripe
        frontend_url = os.getenv("FRONTEND_URL", "https://yourapp.com")
        
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=line_items,
            mode='payment',
            success_url=f"{frontend_url}/checkout/success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{frontend_url}/checkout/cancel",
            client_reference_id=conversation_id,
            metadata={
                'conversation_id': conversation_id,
                'user_name': user_name,
                'phone_number': phone_number,
                'cart_id': str(cart_id),
                'source': 'whatsapp'
            },
            billing_address_collection='required',
            shipping_address_collection={
                'allowed_countries': ['MX'],
            },
            phone_number_collection={'enabled': True}
        )
        
        logger.info(
            f"✅ Checkout creado - SessionID: {checkout_session.id}, "
            f"URL: {checkout_session.url[:50]}..."
        )
        
        return {
            "success": True,
            "checkout_url": checkout_session.url,
            "session_id": checkout_session.id,
            "total_amount": total_amount,
            "items_count": len(cart_items),
            "cart_items": cart_items
        }
        
    except stripe.error.StripeError as e:
        logger.error(f"❌ Error de Stripe: {type(e).__name__}: {str(e)}")
        return {
            "success": False,
            "error": f"Error al crear sesión de pago: {str(e)}"
        }
    
    except Exception as e:
        logger.error(f"❌ Error creando checkout: {type(e).__name__}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": "Error inesperado al crear sesión de pago. Intenta de nuevo."
        }


def format_checkout_message(cart_items: List[Dict], total: float, checkout_url: str) -> str:
    """
    Formatea un mensaje de WhatsApp con resumen del carrito y link de pago.
    
    Args:
        cart_items: Lista de items en el carrito
        total: Total a pagar
        checkout_url: URL del checkout de Stripe
        
    Returns:
        Mensaje formateado para WhatsApp
    """
    # Construir lista de items
    items_text = []
    for i, item in enumerate(cart_items, 1):
        subtotal = item['price'] * item['quantity']
        items_text.append(
            f"{i}. *{item['name']}*\n"
            f"   Color: {item.get('color', 'N/A')} | Cantidad: {item['quantity']}\n"
            f"   Precio: ${subtotal:.2f} MXN"
        )
    
    items_section = "\n\n".join(items_text)
    
    # Construir mensaje completo
    message = f"""🛒 *RESUMEN DE TU CARRITO*

{items_section}

━━━━━━━━━━━━━━━━━━
💰 *TOTAL: ${total:.2f} MXN*
━━━━━━━━━━━━━━━━━━

¡Perfecto! Tu orden está lista para procesar. 

👉 *HAZ CLIC AQUÍ PARA PAGAR:*
{checkout_url}

✅ Pago 100% seguro con Stripe
🚚 Envío a toda la República Mexicana  
📦 Recibirás confirmación de tu orden
⏰ Este link es válido por 24 horas

_Si deseas modificar tu carrito antes de pagar, solo dime "modificar carrito" o "seguir comprando"._"""
    
    return message


def format_cart_summary(cart_items: List[Dict], total: float) -> str:
    """
    Formatea un resumen del carrito sin link de pago.
    
    Args:
        cart_items: Lista de items en el carrito
        total: Total del carrito
        
    Returns:
        Mensaje formateado para WhatsApp
    """
    if not cart_items:
        return "Tu carrito está vacío. 🛒\n\nBusca productos y dime cuál quieres agregar."
    
    items_text = []
    for i, item in enumerate(cart_items, 1):
        subtotal = item['price'] * item['quantity']
        items_text.append(
            f"{i}. *{item['name']}* ({item.get('color', 'N/A')})\n"
            f"   Cantidad: {item['quantity']} | Subtotal: ${subtotal:.2f} MXN"
        )
    
    items_section = "\n\n".join(items_text)
    
    message = f"""🛒 *TU CARRITO ACTUAL*

{items_section}

━━━━━━━━━━━━━━━━━━
💰 *TOTAL: ${total:.2f} MXN*
━━━━━━━━━━━━━━━━━━

¿Qué deseas hacer?
• "Proceder al pago" - Para finalizar compra
• "Agregar más productos" - Para seguir comprando
• "Quitar producto X" - Para eliminar un item
• "Vaciar carrito" - Para empezar de nuevo"""
    
    return message