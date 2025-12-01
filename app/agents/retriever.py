# app/agents/retriever.py
from openai import OpenAI
import os
import json
from typing import Optional
from ..db import get_connection
from ..embeddings import embed_text
import numpy as np
import logging

logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def cosine_similarity(a, b):
    """Calcula similitud coseno entre dos vectores"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_products(
    search_query: str, 
    top_k: int = 5, 
    filters: Optional[dict] = None
) -> list:
    """
    Retriever agent: Búsqueda híbrida (vectorial + filtros)
    
    Args:
        search_query: Query optimizada para búsqueda (ya procesada por query_builder)
        top_k: Número de productos a retornar
        filters: Filtros SQL opcionales (categoría, precio, color)
    """
    
    print("🔍 Buscando productos con query: %s", search_query)
    
    # MEJORA 2: Generar embedding con mejor contexto
    # Agregar palabras clave de moda para mejorar la búsqueda
    enhanced_query = f"producto de moda ropa {search_query}"
    query_embedding = embed_text(enhanced_query)
    
    # MEJORA 3: Pre-filtrado SQL para reducir espacio de búsqueda
    conn = get_connection()
    cur = conn.cursor()
    
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
        WHERE embedding IS NOT NULL
    """
    params = []
    
    # Aplicar filtros SQL si existen
    if filters:
        if filters.get('category'):
            sql_query += " AND product_group_name = ?"
            params.append(filters['category'])
        
        if filters.get('min_price'):
            sql_query += " AND price_mxn >= ?"
            params.append(filters['min_price'])
        
        if filters.get('max_price'):
            sql_query += " AND price_mxn <= ?"
            params.append(filters['max_price'])
        
        if filters.get('color'):
            sql_query += " AND colour_group_name LIKE ?"
            params.append(f"%{filters['color']}%")
    
    cur.execute(sql_query, params)
    
    # MEJORA 4: Cálculo vectorizado más eficiente
    products = []
    rows = cur.fetchall()
    
    # Convertir todos los embeddings a numpy array de una vez
    embeddings_matrix = []
    product_data = []
    
    for row in rows:
        product_embedding = json.loads(row[8])
        embeddings_matrix.append(product_embedding)
        product_data.append({
            "article_id": row[0],
            "prod_name": row[1],
            "product_type_name": row[2],
            "product_group_name": row[3],
            "colour_group_name": row[4],
            "detail_desc": row[5],
            "price_mxn": row[6],
            "image_url": row[7],
        })
    
    conn.close()
    
    if not embeddings_matrix:
        print("⚠️ No se encontraron productos con embeddings")
        return []
    
    # Cálculo vectorizado (mucho más rápido)
    embeddings_matrix = np.array(embeddings_matrix)
    query_embedding_np = np.array(query_embedding)
    
    # Calcular similitudes para todos los productos a la vez
    similarities = np.dot(embeddings_matrix, query_embedding_np) / (
        np.linalg.norm(embeddings_matrix, axis=1) * np.linalg.norm(query_embedding_np)
    )
    
    # Combinar productos con sus similitudes
    for i, product in enumerate(product_data):
        product['similarity'] = float(similarities[i])
        products.append(product)
    
    # MEJORA 5: Re-ranking con boost por palabras clave
    keywords = search_query.lower().split()
    
    for product in products:
        # Boost si hay match exacto en nombre o descripción
        product_text = f"{product['prod_name']} {product['detail_desc']}".lower()
        
        keyword_matches = sum(1 for keyword in keywords if keyword in product_text)
        
        # Boost: 5% por cada palabra clave encontrada
        boost = 1 + (keyword_matches * 0.05)
        product['similarity'] *= boost
        product['keyword_matches'] = keyword_matches
    
    # Ordenar por similitud (ya con boost aplicado)
    products.sort(key=lambda x: x["similarity"], reverse=True)
    
    # MEJORA 6: Filtro de umbral mínimo de similitud
    min_similarity = 0.3  # Solo productos con al menos 30% de similitud
    products = [p for p in products if p['similarity'] >= min_similarity]
    
    if not products:
        print("⚠️ No se encontraron productos con suficiente similitud")
    else:
        print("✅ Top producto: %s (similitud: %s)", products[0]['prod_name'], products[0]['similarity'])
    
    return products[:top_k]