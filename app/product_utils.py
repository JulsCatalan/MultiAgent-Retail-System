"""Utility functions for product operations."""
from typing import Optional, List
import numpy as np
from .db import get_connection
from .schemas import Product, SearchConstraints


def cosine_similarity(a, b):
    """Calcula similitud coseno entre dos vectores"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_products_simple(
    category: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    page: int = 1,
    page_size: int = 20
) -> tuple[List[Product], int]:
    """
    Obtiene productos con filtros SQL simples (sin búsqueda vectorial)
    """
    conn = get_connection()
    cur = conn.cursor()
    
    # Query base
    sql_query = """
        SELECT 
            article_id,
            prod_name,
            product_type_name,
            product_group_name,
            colour_group_name,
            detail_desc,
            price_mxn,
            image_url
        FROM products
        WHERE 1=1
    """
    params = []
    
    # Aplicar filtros
    if category:
        sql_query += " AND product_group_name = ?"
        params.append(category)
    
    if min_price is not None:
        sql_query += " AND price_mxn >= ?"
        params.append(min_price)
    
    if max_price is not None:
        sql_query += " AND price_mxn <= ?"
        params.append(max_price)
    
    # Contar total (antes de paginar)
    count_query = f"SELECT COUNT(*) FROM ({sql_query})"
    cur.execute(count_query, params)
    total = cur.fetchone()[0]
    
    # Agregar paginación
    offset = (page - 1) * page_size
    sql_query += " ORDER BY prod_name LIMIT ? OFFSET ?"
    params.extend([page_size, offset])
    
    # Ejecutar query
    cur.execute(sql_query, params)
    rows = cur.fetchall()
    
    # Formatear productos
    products = []
    for row in rows:
        # Convertir article_id a int
        try:
            product_id = int(row[0])
        except (ValueError, TypeError):
            product_id = abs(hash(row[0])) % (10 ** 9)
        
        products.append(Product(
            id=product_id,
            name=row[1],
            brand=row[2] or "Fashion Store",
            category=row[3] or "General",
            price=row[6],
            image=row[7] or "https://images.unsplash.com/photo-1523381210434-271e8be1f52b?w=400&h=600&fit=crop",
            description=row[5],
            color=row[4],
            type=row[2]
        ))
    
    conn.close()
    
    return products, total


# search_products_vector se queda para el endpoint /search y WhatsApp
def search_products_vector(query: str, constraints: SearchConstraints, k: int = 12) -> List[Product]:
    """
    Búsqueda vectorial de productos.
    TODO: Implementar búsqueda vectorial usando embeddings.
    """
    # TODO: Implementar búsqueda vectorial
    return []

