# app/agents/retriever.py
from openai import OpenAI
import os
import json
from ..db import get_connection
from ..embeddings import embed_text
import numpy as np

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def cosine_similarity(a, b):
    """Calcula similitud coseno entre dos vectores"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_products(user_message: str, top_k: int = 5, filters: dict = None) -> list:
    """
    Retriever agent mejorado: Búsqueda híbrida (vectorial + filtros)
    
    Args:
        user_message: Consulta del usuario
        top_k: Número de productos a retornar
        filters: Filtros SQL opcionales (categoría, precio, color)
    """
    
    # MEJORA 1: Mejor prompt de extracción - más específico
    extraction_prompt = f"""Eres un asistente experto en moda. Extrae SOLO los atributos más importantes de esta búsqueda:

Consulta del usuario: "{user_message}"

Extrae:
1. Tipo de producto (camisa, pantalón, zapatos, etc.)
2. Color principal (si se menciona)
3. Estilo/ocasión (casual, formal, deportivo, etc.)
4. Características especiales (manga larga, con bolsillos, etc.)

Responde en UNA frase corta y directa, enfocada en los atributos del producto.
Ejemplo: "camisa formal manga larga color blanco"
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": extraction_prompt}],
        max_tokens=100,  # Reducido para respuestas más concisas
        temperature=0.1   # Más determinista
    )
    
    search_query = response.choices[0].message.content.strip()
    print(f"🔍 Query optimizada: {search_query}")
    
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
        print(f"✅ Top producto: {products[0]['prod_name']} (similitud: {products[0]['similarity']:.3f})")
    
    return products[:top_k]