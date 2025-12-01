"""Utility functions for product operations."""
from typing import Optional, List
import numpy as np
import json
import logging
from .db import get_connection
from .schemas import Product, SearchConstraints
from .embeddings import embed_text

logger = logging.getLogger(__name__)


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


def search_products_vector(query: str, constraints: SearchConstraints, k: int = 12) -> List[Product]:
    """
    Búsqueda vectorial de productos usando embeddings.
    
    Args:
        query: Texto de búsqueda del usuario
        constraints: Restricciones adicionales (categoría, precio, color, etc.)
        k: Número máximo de productos a retornar
        
    Returns:
        Lista de Product ordenados por similitud
    """
    try:
        # Generar embedding de la consulta
        query_embedding = embed_text(query)
        
        # Buscar en la base de datos
        conn = get_connection()
        cur = conn.cursor()
        
        # Construir query SQL con filtros
        sql_query = """
            SELECT 
                article_id,
                prod_name,
                product_type_name,
                product_group_name,
                colour_group_name,
                detail_desc,
                price_mxn,
                image_url,
                embedding
            FROM products
            WHERE embedding IS NOT NULL AND embedding != ''
        """
        params = []
        
        # Aplicar filtros de constraints
        if constraints.category:
            sql_query += " AND product_group_name = ?"
            params.append(constraints.category)
        
        if constraints.color:
            sql_query += " AND colour_group_name LIKE ?"
            params.append(f"%{constraints.color}%")
        
        if constraints.brand:
            sql_query += " AND product_type_name LIKE ?"
            params.append(f"%{constraints.brand}%")
        
        if constraints.price_min is not None:
            sql_query += " AND price_mxn >= ?"
            params.append(constraints.price_min)
        
        if constraints.price_max is not None:
            sql_query += " AND price_mxn <= ?"
            params.append(constraints.price_max)
        
        cur.execute(sql_query, params)
        
        # Calcular similitud para cada producto
        products_with_similarity = []
        for row in cur.fetchall():
            try:
                product_embedding = json.loads(row[8])
                similarity = cosine_similarity(query_embedding, product_embedding)
                
                # Convertir article_id a int
                try:
                    product_id = int(row[0])
                except (ValueError, TypeError):
                    product_id = abs(hash(row[0])) % (10 ** 9)
                
                products_with_similarity.append({
                    "product": Product(
                        id=product_id,
                        name=row[1],
                        brand=row[2] or "Fashion Store",
                        category=row[3] or "General",
                        price=row[6],
                        image=row[7] or "https://images.unsplash.com/photo-1523381210434-271e8be1f52b?w=400&h=600&fit=crop",
                        description=row[5],
                        color=row[4],
                        type=row[2]
                    ),
                    "similarity": similarity
                })
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                print("Error procesando producto %s: %s", row[0], e)
                continue
        
        conn.close()
        
        # Ordenar por similitud (mayor a menor) y tomar top k
        products_with_similarity.sort(key=lambda x: x["similarity"], reverse=True)
        products = [item["product"] for item in products_with_similarity[:k]]
        
        print("Búsqueda vectorial: %d productos encontrados para query '%s'", len(products), query)
        
        return products
        
    except Exception as e:
        print("Error en búsqueda vectorial: %s", e)
        return []

