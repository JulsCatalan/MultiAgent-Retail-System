# app/db.py
import sqlite3

DB_PATH = "products.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_connection()
    cur = conn.cursor()
    
    # Tabla principal de productos
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

    # Tabla de carritos por conversación/usuario
    cur.execute("""
        CREATE TABLE IF NOT EXISTS carts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT UNIQUE NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Items dentro del carrito
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cart_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cart_id INTEGER NOT NULL,
            article_id TEXT NOT NULL,
            quantity INTEGER NOT NULL DEFAULT 1,
            added_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(cart_id) REFERENCES carts(id),
            FOREIGN KEY(article_id) REFERENCES products(article_id)
        )
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_cart_items_cart_id
        ON cart_items(cart_id)
    """)

    # Productos más recientes mostrados al usuario (para mapping tipo "Producto 1")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS recent_products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            article_id TEXT NOT NULL,
            position INTEGER NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_recent_products_conversation
        ON recent_products(conversation_id)
    """)
    
    # Tabla de órdenes completadas
    cur.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            user_name TEXT,
            phone_number TEXT,
            stripe_session_id TEXT UNIQUE NOT NULL,
            stripe_payment_intent TEXT,
            total_amount REAL NOT NULL,
            currency TEXT DEFAULT 'MXN',
            status TEXT DEFAULT 'completed',
            shipping_address TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_orders_conversation
        ON orders(conversation_id)
    """)
    
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_orders_session
        ON orders(stripe_session_id)
    """)
    
    # Items de la orden (copia de cart_items al momento de la compra)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS order_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER NOT NULL,
            article_id TEXT NOT NULL,
            prod_name TEXT,
            price_mxn REAL NOT NULL,
            quantity INTEGER NOT NULL,
            subtotal REAL NOT NULL,
            FOREIGN KEY(order_id) REFERENCES orders(id),
            FOREIGN KEY(article_id) REFERENCES products(article_id)
        )
    """)
    
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_order_items_order
        ON order_items(order_id)
    """)
    
    # Índices para mejorar performance de búsqueda
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_products_category 
        ON products(product_group_name)
    """)
    
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_products_type 
        ON products(product_type_name)
    """)
    
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_products_colour 
        ON products(colour_group_name)
    """)
    
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_products_department 
        ON products(department_name)
    """)
    
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_products_price 
        ON products(price_mxn)
    """)
    
    # ⭐ IMPORTANTE: Hacer commit ANTES de cerrar
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