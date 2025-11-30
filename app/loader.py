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

        # Creamos un texto completo del producto para el embedding
        text_to_embed = f"""
        {row['prod_name']} 
        {row['product_type_name']} 
        {row['product_group_name']} 
        {row['detail_desc']}
        """

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
