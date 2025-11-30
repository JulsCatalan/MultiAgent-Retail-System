# app/main.py
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
from pydantic import BaseModel
from dotenv import load_dotenv
import json

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
    id: str
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
    id: str
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

# ==================== HELPER FUNCTIONS ====================

def cosine_similarity(a, b):
    """Calcula similitud coseno entre dos vectores"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_products_vector(query: str, constraints: SearchConstraints, k: int = 12) -> List[Product]:
    """
    Búsqueda vectorial de productos usando embeddings
    """
    # Generar embedding de la query
    query_embedding = embed_text(query)
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Construir query SQL base
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
    
    # Aplicar filtros de constraints
    if constraints.category:
        sql_query += " AND product_group_name = ?"
        params.append(constraints.category)
    
    if constraints.color:
        sql_query += " AND colour_group_name LIKE ?"
        params.append(f"%{constraints.color}%")
    
    if constraints.price_min is not None:
        sql_query += " AND price_mxn >= ?"
        params.append(constraints.price_min)
    
    if constraints.price_max is not None:
        sql_query += " AND price_mxn <= ?"
        params.append(constraints.price_max)
    
    # Ejecutar query
    cur.execute(sql_query, params)
    rows = cur.fetchall()
    
    # Calcular similitudes y rankear
    products_with_scores = []
    for row in rows:
        product_embedding = json.loads(row[8])
        similarity = cosine_similarity(query_embedding, product_embedding)
        
        products_with_scores.append({
            "product": Product(
                id=row[0],
                name=row[1],
                brand=row[2] or "Fashion Store",
                category=row[3] or "General",
                price=row[6],
                image=row[7] or "https://images.unsplash.com/photo-1523381210434-271e8be1f52b?w=400&h=600&fit=crop",
                description=row[5],
                color=row[4],
                type=row[2]
            ),
            "score": similarity
        })
    
    conn.close()
    
    # Ordenar por similitud y tomar top k
    products_with_scores.sort(key=lambda x: x["score"], reverse=True)
    return [item["product"] for item in products_with_scores[:k]]

# ==================== STARTUP ====================

@app.on_event("startup")
def startup_event():
    print("📦 Inicializando base de datos...")
    init_db()

    existing = count_embeddings()

    if existing == 0:
        print("📤 No hay embeddings. Cargando datos y generando embeddings...")
        csv_path = "app/data/products.csv"
        load_products_to_db(csv_path)
        print("✅ Datos cargados correctamente.")
    else:
        print(f"🔍 Embeddings existentes detectados: {existing}. No se recargarán datos.")

# ==================== ENDPOINTS ====================

@app.get("/")
def root():
    return {
        "status": "API funcionando",
        "version": "1.0.0",
        "endpoints": {
            "search": "POST /search",
            "products": "GET /products",
            "categories": "GET /categories",
            "checkout": "POST /create-checkout-session",
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

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Búsqueda vectorial de productos con filtros opcionales
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

@app.get("/products", response_model=SearchResponse)
def get_products(
    category: Optional[str] = Query(None),
    min_price: Optional[float] = Query(None),
    max_price: Optional[float] = Query(None),
    search: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100)
):
    """
    Obtiene productos con filtros opcionales (endpoint alternativo)
    """
    try:
        constraints = SearchConstraints(
            category=category,
            price_min=min_price,
            price_max=max_price
        )
        
        query = search if search else "productos de moda"
        products = search_products_vector(query, constraints, page_size)
        
        return SearchResponse(
            items=products,
            total=len(products)
        )
    except Exception as e:
        print(f"❌ Error obteniendo productos: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/categories")
def get_categories():
    """
    Obtiene todas las categorías disponibles
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

@app.post("/create-checkout-session", response_model=CheckoutResponse)
async def create_checkout_session(request: CheckoutRequest):
    """
    Crea una sesión de checkout (placeholder - integrar con Stripe)
    """
    try:
        # Calcular total
        total = sum(item.price * item.quantity for item in request.cart)
        
        # TODO: Integrar con Stripe
        # Por ahora, retornamos una URL de ejemplo
        # En producción, aquí crearías la sesión de Stripe
        
        # import stripe
        # stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
        # 
        # line_items = [
        #     {
        #         "price_data": {
        #             "currency": "mxn",
        #             "product_data": {
        #                 "name": item.name,
        #                 "images": [item.image],
        #             },
        #             "unit_amount": int(item.price * 100),
        #         },
        #         "quantity": item.quantity,
        #     }
        #     for item in request.cart
        # ]
        # 
        # session = stripe.checkout.Session.create(
        #     payment_method_types=["card"],
        #     line_items=line_items,
        #     mode="payment",
        #     success_url=f"{os.getenv('FRONTEND_URL')}/success?session_id={{CHECKOUT_SESSION_ID}}",
        #     cancel_url=f"{os.getenv('FRONTEND_URL')}/cancel",
        #     customer_email=request.customer_phone,  # O email si lo capturas
        #     metadata={
        #         "customer_name": request.customer_name,
        #         "customer_phone": request.customer_phone,
        #     }
        # )
        
        # Por ahora, URL de ejemplo
        session_id = f"cs_test_{request.customer_name.replace(' ', '_')}"
        checkout_url = f"https://checkout.stripe.com/pay/{session_id}"
        
        print(f"✅ Checkout creado para {request.customer_name}")
        print(f"   Total: ${total:.2f} MXN")
        print(f"   Items: {len(request.cart)}")
        
        return CheckoutResponse(
            checkout_url=checkout_url,
            session_id=session_id
        )
        
    except Exception as e:
        print(f"❌ Error creando checkout: {str(e)}")
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