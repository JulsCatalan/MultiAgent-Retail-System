# app/main.py
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from .db import init_db, count_embeddings
from typing import Optional, List
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import logging

logger = logging.getLogger(__name__)

load_dotenv()

from .db import count_embeddings, get_connection
from .loader import load_products_to_db
from kapso.use_kapso import use_kapso
from .schemas import (
    Product,
    SearchRequest,
    SearchResponse
)
from .product_utils import (
    get_products_simple,
    search_products_vector
)
from app.agents.process_user_query import process_user_query
from models import User, UserMetadata

app = FastAPI(title="Fashion Store API", version="1.0.0")

# CORS para tu frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
            "products": "GET /products - Obtiene productos con filtros simples",
            "categories": "GET /categories - Lista de categorías",
            "search": "POST /search - Búsqueda vectorial (para WhatsApp)",
            "checkout": "POST /create-checkout-session",
            "regenerate": "POST /regenerate-embeddings",
            "whatsapp": "POST /whatsapp",
            "test-agent": "POST /test-agent - Prueba el agente sin WhatsApp",
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
        logger.error(f"❌ Error obteniendo productos: {str(e)}")
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
        logger.error(f"❌ Error en búsqueda: {str(e)}")
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
        logger.error(f"❌ Error en webhook: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


class TestAgentRequest(BaseModel):
    """Request model para probar el agente"""
    message: str
    conversation_id: Optional[str] = "test-conversation-123"
    phone_number: Optional[str] = "+1234567890"
    user_name: Optional[str] = "Usuario de Prueba"


@app.post("/test-agent")
async def test_agent(request: TestAgentRequest):
    """
    Endpoint para probar el agente sin necesidad de WhatsApp.
    
    Permite enviar un mensaje y recibir la respuesta del agente con productos
    y decisiones de routing, simulando una conversación.
    
    Ejemplo de uso:
    ```json
    {
        "message": "Busco camisetas rojas",
        "conversation_id": "test-123",
        "phone_number": "+1234567890",
        "user_name": "Juan"
    }
    ```
    """
    try:
        # Crear usuario de prueba
        user = User(
            name=request.user_name,
            phone_number=request.phone_number,
            conversation_id=request.conversation_id,
            metadata=UserMetadata(
                whatsapp_config_id="test-config",
                reached_from_phone_number=request.phone_number
            )
        )
        
        logger.info("🧪 Probando agente con mensaje: %s", request.message)
        
        # Procesar query con el agente
        result = await process_user_query(user, request.message)
        
        return {
            "status": "success",
            "user_message": request.message,
            "agent_response": result["response"],
            "products_found": len(result.get("products", [])),
            "products": result.get("products", []),
            "routing_decision": result.get("routing_decision"),
            "conversation_history_count": len(result.get("conversation_history", [])),
            "conversation_history": result.get("conversation_history", [])
        }
        
    except Exception as e:
        logger.error("❌ Error probando agente: %s", str(e))
        import traceback
        logger.error("Traceback: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error probando agente: {str(e)}")


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
        logger.info("🔄 Iniciando regeneración de embeddings...")
        
        # Borrar embeddings existentes
        conn = get_connection()
        cur = conn.cursor()
        
        cur.execute("UPDATE products SET embedding = NULL")
        conn.commit()
        
        deleted_count = cur.rowcount
        logger.info(f"🗑️  Embeddings borrados: {deleted_count}")
        
        conn.close()
        
        # Recargar productos y generar nuevos embeddings
        csv_path = "app/data/products.csv"
        load_products_to_db(csv_path)
        
        # Contar nuevos embeddings
        new_count = count_embeddings()
        
        logger.info(f"✅ Regeneración completada: {new_count} embeddings creados")
        
        return {
            "status": "success",
            "message": "Embeddings regenerados exitosamente",
            "embeddings_deleted": deleted_count,
            "embeddings_created": new_count
        }
        
    except Exception as e:
        logger.error(f"❌ Error regenerando embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))