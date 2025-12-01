from typing import List, Dict, Any
from .db import get_connection


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


