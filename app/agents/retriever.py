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

def search_products(user_message: str, top_k: int = 5) -> list:
    """
    Retriever agent: Busca productos relevantes en la base vectorial
    """
    
    # Extraemos la intención de búsqueda con GPT
    extraction_prompt = f"""Extrae los criterios de búsqueda de esta consulta:

"{user_message}"

Identifica:
- Tipo de prenda
- Color
- Estilo
- Rango de precio (si lo menciona)
- Cualquier otra característica relevante

Resume en una frase optimizada para búsqueda semántica."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": extraction_prompt}],
        max_tokens=200,
        temperature=0.3
    )
    
    search_query = response.choices[0].message.content.strip()
    print(f"🔍 Query optimizada: {search_query}")
    
    # Generamos embedding de la consulta
    query_embedding = embed_text(search_query)
    
    # Buscamos en la base de datos
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute("""
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
    """)
    
    products = []
    for row in cur.fetchall():
        product_embedding = json.loads(row[8])
        similarity = cosine_similarity(query_embedding, product_embedding)
        
        products.append({
            "article_id": row[0],
            "prod_name": row[1],
            "product_type_name": row[2],
            "product_group_name": row[3],
            "colour_group_name": row[4],
            "detail_desc": row[5],
            "price_mxn": row[6],
            "image_url": row[7],
            "similarity": similarity
        })
    
    conn.close()
    
    # Ordenamos por similitud y tomamos top_k
    products.sort(key=lambda x: x["similarity"], reverse=True)
    return products[:top_k]