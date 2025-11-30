# app/main.py
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

from .db import init_db, count_embeddings, get_connection
from .loader import load_products_to_db
from .embeddings import embed_text
from kapso.use_kapso import use_kapso
import numpy as np

app = FastAPI(title="Fashion Store API", version="1.0.0")

# CORS para tu frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== MODELS ====================

class Product(BaseModel):
    id: int
    name: str
    brand: str
    category: str
    price: float
    image: str
    description: Optional[str] = None
    color: Optional[str] = None
    type: Optional[str] = None

class SearchConstraints(BaseModel):
    category: Optional[str] = None
    brand: Optional[str] = None
    color: Optional[str] = None
    size: Optional[str] = None
    price_min: Optional[float] = None
    price_max: Optional[float] = None

class SearchRequest(BaseModel):
    query: str
    constraints: SearchConstraints = SearchConstraints()
    k: int = 12

class SearchResponse(BaseModel):
    items: List[Product]
    total: int

class CartItem(BaseModel):
    id: int
    name: str
    brand: str
    category: str
    price: float
    image: str
    quantity: int
    color: Optional[str] = None
    size: Optional[str] = None

class CheckoutRequest(BaseModel):
    cart: List[CartItem]
    customer_name: str
    customer_phone: str

class CheckoutResponse(BaseModel):
    checkout_url: str
    session_id: str


# app/main.py - Reemplaza los endpoints de products y agrega uno nuevo

# ==================== HELPER FUNCTIONS ====================

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
    # ... (mantén esta función tal cual para /search)
    pass

# ==================== ENDPOINTS ====================

@app.get("/")
def root():
    return {
        "status": "API funcionando",
        "version": "1.0.0",
        "endpoints": {
            "products": "GET /products - Obtiene productos con filtros simples",
            "categories": "GET /categories - Lista de categorías",
            "search": "POST /search - Búsqueda vectorial (para WhatsApp)",
            "checkout": "POST /create-checkout-session",
            "regenerate": "POST /regenerate-embeddings",
            "whatsapp": "POST /whatsapp",
            "health": "GET /health"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "embeddings_count": count_embeddings()
    }

class ProductsResponse(BaseModel):
    items: List[Product]
    total: int
    page: int
    page_size: int
    total_pages: int

@app.get("/products", response_model=ProductsResponse)
def get_products(
    category: Optional[str] = Query(None, description="Filtrar por categoría"),
    min_price: Optional[float] = Query(None, description="Precio mínimo"),
    max_price: Optional[float] = Query(None, description="Precio máximo"),
    page: int = Query(1, ge=1, description="Número de página"),
    page_size: int = Query(20, ge=1, le=100, description="Productos por página")
):
    """
    Obtiene productos con filtros SQL simples (sin búsqueda vectorial)
    """
    try:
        products, total = get_products_simple(
            category=category,
            min_price=min_price,
            max_price=max_price,
            page=page,
            page_size=page_size
        )
        
        total_pages = (total + page_size - 1) // page_size  # Calcular total de páginas
        
        return ProductsResponse(
            items=products,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )
    except Exception as e:
        print(f"❌ Error obteniendo productos: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/categories")
def get_categories():
    """
    Obtiene todas las categorías disponibles con conteo de productos
    """
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute("""
        SELECT 
            product_group_name,
            COUNT(*) as count
        FROM products
        WHERE product_group_name IS NOT NULL
        GROUP BY product_group_name
        ORDER BY count DESC
    """)
    
    categories = []
    for row in cur.fetchall():
        categories.append({
            "name": row[0],
            "count": row[1]
        })
    
    conn.close()
    
    return {"categories": categories}

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Búsqueda vectorial de productos (usado por WhatsApp)
    """
    try:
        products = search_products_vector(
            request.query,
            request.constraints,
            request.k
        )
        
        return SearchResponse(
            items=products,
            total=len(products)
        )
    except Exception as e:
        print(f"❌ Error en búsqueda: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/whatsapp")
async def whatsapp_agent(request: Request):
    """
    Endpoint para webhooks de Kapso (WhatsApp)
    """
    try:
        webhook_data = await request.json()
        result = await use_kapso(webhook_data)
        
        if result.get("status") == "success":
            return result
        else:
            raise HTTPException(
                status_code=500, 
                detail=result.get("message", "Error procesando webhook")
            )
    except Exception as e:
        print(f"❌ Error en webhook: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    

class RegenerateRequest(BaseModel):
    password: str

@app.get("/regenerate-embeddings")
async def regenerate_embeddings(password: str = Query(..., alias="ADMIN_PASSWORD")):
    """
    Fuerza la regeneración de todos los embeddings
    Requiere contraseña de administrador como query parameter
    
    Uso: POST /regenerate-embeddings?ADMIN_PASSWORD=tu_contraseña
    """
    # Verificar contraseña
    admin_password = os.getenv("ADMIN_PASSWORD", "admin123")
    
    if password != admin_password:
        raise HTTPException(status_code=403, detail="Contraseña incorrecta")
    
    try:
        print("🔄 Iniciando regeneración de embeddings...")
        
        # Borrar embeddings existentes
        conn = get_connection()
        cur = conn.cursor()
        
        cur.execute("UPDATE products SET embedding = NULL")
        conn.commit()
        
        deleted_count = cur.rowcount
        print(f"🗑️  Embeddings borrados: {deleted_count}")
        
        conn.close()
        
        # Recargar productos y generar nuevos embeddings
        csv_path = "app/data/products.csv"
        load_products_to_db(csv_path)
        
        # Contar nuevos embeddings
        new_count = count_embeddings()
        
        print(f"✅ Regeneración completada: {new_count} embeddings creados")
        
        return {
            "status": "success",
            "message": "Embeddings regenerados exitosamente",
            "embeddings_deleted": deleted_count,
            "embeddings_created": new_count
        }
        
    except Exception as e:
        print(f"❌ Error regenerando embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))