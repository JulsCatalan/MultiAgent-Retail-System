# app/loader.py
import pandas as pd
import json
from .db import get_connection
from .embeddings import embed_text

def load_products_to_db(csv_path: str):
    df = pd.read_csv(csv_path)

    conn = get_connection()
    cur = conn.cursor()

    for _, row in df.iterrows():
        # MEJORA: Texto más estructurado y con contexto
        # Agregar "producto de moda" y estructurar mejor
        text_to_embed = f"""producto de moda ropa 
Nombre: {row['prod_name']}
Tipo: {row['product_type_name']} 
Categoría: {row['product_group_name']}
Color: {row['colour_group_name']}
Descripción: {row['detail_desc']}
Departamento: {row['department_name']}
Estilo: {row['graphical_appearance_name']}"""

        embedding = embed_text(text_to_embed)

        cur.execute("""
            INSERT OR REPLACE INTO products (
                article_id,
                product_code,
                prod_name,
                product_type_name,
                product_group_name,
                graphical_appearance_name,
                colour_group_name,
                perceived_colour_value_name,
                perceived_colour_master_name,
                department_name,
                index_group_name,
                section_no,
                section_name,
                detail_desc,
                price_mxn,
                image_url,
                embedding
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            row["article_id"],
            row["product_code"],
            row["prod_name"],
            row["product_type_name"],
            row["product_group_name"],
            row["graphical_appearance_name"],
            row["colour_group_name"],
            row["perceived_colour_value_name"],
            row["perceived_colour_master_name"],
            row["department_name"],
            row["index_group_name"],
            row["section_no"],
            row["section_name"],
            row["detail_desc"],
            row["price_mxn"],
            row["image_url"],
            json.dumps(embedding)
        ])

    conn.commit()
    conn.close()