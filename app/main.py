# app/main.py
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from .db import init_db, count_embeddings
from typing import Optional, List
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import logging
import time

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
from models import User, UserMetadata, ConversationMessage
from app.models import Constraints

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
        print("🔍 Embeddings existentes detectados: %s. No se recargarán datos.", existing)

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
            "whatsapp": "POST /whatsapp - Webhook de WhatsApp",
            "test-agent": "POST /test-agent - Prueba el agente sin WhatsApp",
            "test-agent-web": "POST /test-agent-web - Agente para frontend web con contexto",
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
    start_time = time.time()
    
    logger.info(
        "🔍 [SEARCH] Nueva búsqueda - "
        f"Query: '{request.query[:100]}{'...' if len(request.query) > 100 else ''}', "
        f"K: {request.k}, "
        f"Filtros: {dict((k, v) for k, v in request.constraints.dict().items() if v is not None)}"
    )
    
    try:
        products = search_products_vector(
            request.query,
            request.constraints,
            request.k
        )
        
        total_time = time.time() - start_time
        
        logger.info(
            f"✅ [SEARCH] Búsqueda completada - "
            f"Productos encontrados: {len(products)}, "
            f"Tiempo: {total_time:.2f}s"
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
        print(f"❌ Error en webhook: {str(e)}")
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
    start_time = time.time()
    
    logger.info(
        "🧪 [TEST-AGENT] Nueva solicitud - "
        f"Usuario: '{request.user_name}', "
        f"Mensaje: '{request.message[:100]}{'...' if len(request.message) > 100 else ''}', "
        f"ConvID: {request.conversation_id}"
    )
    
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
        agent_start = time.time()
        result = await process_user_query(user, request.message)
        agent_time = time.time() - agent_start
        
        logger.info(f"🤖 Agente procesado en {agent_time:.2f}s")
        
        total_time = time.time() - start_time
        
        logger.info(
            "✅ [TEST-AGENT] Solicitud completada - "
            f"Usuario: '{request.user_name}', "
            f"Productos: {len(result.get('products', []))}, "
            f"Decisión: '{result.get('routing_decision', 'unknown')}', "
            f"Tiempo total: {total_time:.2f}s"
        )
        
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


class WebContext(BaseModel):
    """Contexto de conversación para el frontend web"""
    history: List[ConversationMessage] = Field(default_factory=list)
    current_constraints: Optional[Constraints] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class TestAgentWebRequest(BaseModel):
    """Request model para el agente web con contexto"""
    message: str
    user_name: Optional[str] = "Usuario Web"
    conversation_id: Optional[str] = None
    context: Optional[WebContext] = None

class TestAgentWebResponse(BaseModel):
    """Response model para el agente web"""
    status: str
    message: str
    agent_response: str
    products: List[Product]  # Reusing existing Product model from schemas
    products_found: int
    routing_decision: str
    conversation_history: List[ConversationMessage]

@app.post("/test-agent-web", response_model=TestAgentWebResponse)
async def test_agent_web(request: TestAgentWebRequest):
    start_time = time.time()
    
    # Log incoming request
    logger.info(
        "🌐 [WEB-AGENT] Nueva solicitud - "
        f"Usuario: '{request.user_name}', "
        f"Mensaje: '{request.message[:100]}{'...' if len(request.message) > 100 else ''}', "
        f"ConvID: {request.conversation_id or 'nuevo'}, "
        f"Historial: {len(request.context.history) if request.context and request.context.history else 0} mensajes"
    )
    
    try:
        # Generar conversation_id único si no se proporciona
        if not request.conversation_id:
            import uuid
            request.conversation_id = f"web-{uuid.uuid4().hex[:12]}"
            logger.debug(f"🆔 Conversation ID generado: {request.conversation_id}")
        
        # Crear usuario web
        user = User(
            name=request.user_name,
            phone_number=None,
            conversation_id=request.conversation_id,
            metadata=UserMetadata(
                whatsapp_config_id="web-app",
                reached_from_phone_number="web"
            )
        )
        
        logger.debug(f"👤 Usuario creado: {user.name} (ConvID: {user.conversation_id})")
        
        # Procesar query con el agente (sin cliente Kapso)
        agent_start = time.time()
        result = await process_user_query(user, request.message, kapso_client=None)
        agent_time = time.time() - agent_start
        
        logger.info(
            f"🤖 Agente procesado en {agent_time:.2f}s - "
            f"Decisión: {result.get('routing_decision', 'unknown')}"
        )
        
        # Convertir productos del retriever al modelo Product existente
        products_web = []
        conversion_errors = 0
        
        for idx, product in enumerate(result.get("products", [])):
            try:
                # Convert article_id to int for Product model
                try:
                    product_id = int(product.get("article_id", 0))
                except (ValueError, TypeError):
                    # If conversion fails, use hash as fallback
                    product_id = abs(hash(product.get("article_id", ""))) % (10 ** 9)
                    logger.debug(f"⚠️ ArticleID no numérico convertido a hash: {product.get('article_id')} → {product_id}")
                
                products_web.append(Product(
                    id=product_id,
                    name=product.get("prod_name", ""),
                    brand=product.get("product_type_name", "Fashion Store"),
                    category=product.get("product_group_name", "General"),
                    price=product.get("price_mxn", 0.0),
                    image=product.get("image_url", ""),
                    description=product.get("detail_desc"),
                    color=product.get("colour_group_name"),
                    type=product.get("product_type_name")
                ))
            except Exception as e:
                conversion_errors += 1
                logger.warning(f"❌ Error convirtiendo producto {idx}: {str(e)}")
        
        if conversion_errors > 0:
            logger.warning(f"⚠️ {conversion_errors} producto(s) no pudieron ser convertidos")
        
        logger.info(f"📦 Productos convertidos: {len(products_web)}/{len(result.get('products', []))}")
        
        # Construir historial actualizado
        conversation_history = []
        
        # Agregar historial anterior si existe
        if request.context and request.context.history:
            conversation_history.extend(request.context.history)
            logger.debug(f"📚 Historial anterior cargado: {len(request.context.history)} mensajes")
        
        # Agregar el mensaje del usuario actual
        from datetime import datetime
        current_timestamp = datetime.utcnow().isoformat() + "Z"
        
        conversation_history.append(ConversationMessage(
            timestamp=current_timestamp,
            sender="client",
            message=request.message,
            message_id=f"{request.conversation_id}-{len(conversation_history)}"
        ))
        
        # Agregar la respuesta del agente
        conversation_history.append(ConversationMessage(
            timestamp=current_timestamp,
            sender="cedamoney",
            message=result["response"],
            message_id=f"{request.conversation_id}-{len(conversation_history)}"
        ))
        
        logger.debug(f"💬 Historial actualizado: {len(conversation_history)} mensajes totales")
        
        # Calculate total request time
        total_time = time.time() - start_time
        
        logger.info(
            "✅ [WEB-AGENT] Solicitud completada exitosamente - "
            f"Usuario: '{request.user_name}', "
            f"ConvID: {request.conversation_id}, "
            f"Productos: {len(products_web)}, "
            f"Decisión: '{result.get('routing_decision', 'unknown')}', "
            f"Respuesta: {len(result['response'])} caracteres, "
            f"Tiempo total: {total_time:.2f}s (Agente: {agent_time:.2f}s)"
        )
        
        return TestAgentWebResponse(
            status="success",
            message="Respuesta generada exitosamente",
            agent_response=result["response"],
            products=products_web,
            products_found=len(products_web),
            routing_decision=result.get("routing_decision", "general"),
            conversation_history=conversation_history
        )
        
    except Exception as e:
        total_time = time.time() - start_time
        
        logger.error(
            "❌ [WEB-AGENT] Error procesando solicitud - "
            f"Usuario: '{request.user_name}', "
            f"ConvID: {request.conversation_id or 'nuevo'}, "
            f"Mensaje: '{request.message[:50]}...', "
            f"Error: {type(e).__name__}: {str(e)}, "
            f"Tiempo hasta error: {total_time:.2f}s"
        )
        
        import traceback
        logger.error("📍 Traceback completo:\n%s", traceback.format_exc())
        
        raise HTTPException(
            status_code=500, 
            detail=f"Error procesando solicitud del agente web: {str(e)}"
        )

