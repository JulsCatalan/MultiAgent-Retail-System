# app/db.py
import sqlite3

DB_PATH = "products.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS products (
            article_id TEXT PRIMARY KEY,
            product_code TEXT,
            prod_name TEXT,
            product_type_name TEXT,
            product_group_name TEXT,
            graphical_appearance_name TEXT,
            colour_group_name TEXT,
            perceived_colour_value_name TEXT,
            perceived_colour_master_name TEXT,
            department_name TEXT,
            index_group_name TEXT,
            section_no TEXT,
            section_name TEXT,
            detail_desc TEXT,
            price_mxn REAL,
            image_url TEXT,
            embedding TEXT  -- JSON del vector
        )
    """)

    conn.commit()
    conn.close()

def count_embeddings():
    """
    Cuenta cuántos productos tienen embeddings en la base de datos
    """
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute("""
        SELECT COUNT(*) 
        FROM products 
        WHERE embedding IS NOT NULL AND embedding != ''
    """)
    
    count = cur.fetchone()[0]
    conn.close()
    
    return count