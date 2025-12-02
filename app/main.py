# app/main.py
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from .db import init_db, count_embeddings
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import logging
import time

logger = logging.getLogger(__name__)

load_dotenv()

from .db import count_embeddings, get_connection
from .loader import load_products_to_db
from kapso.use_kapso import use_kapso
from kapso.client import KapsoClient
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

import stripe
from .cart import (
    get_cart_by_conversation,
    get_cart_items,
    calculate_cart_total,
    clear_cart_by_id
)

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
            "checkout": "POST /create-checkout-session - Crear sesión de pago con Stripe",
            "checkout-success": "GET /checkout/success?session_id=xxx - Confirmar pago y guardar orden",
            "checkout-cancel": "GET /checkout/cancel - Pago cancelado",
            "orders": "GET /orders/{conversation_id} - Historial de órdenes del usuario",
            "order-details": "GET /orders/{conversation_id}/{order_id} - Detalles de una orden específica",
            "regenerate": "GET /regenerate-embeddings?ADMIN_PASSWORD=xxx - Regenerar embeddings",
            "whatsapp": "POST /whatsapp - Webhook de WhatsApp",
            "test-agent": "POST /test-agent - Prueba el agente sin WhatsApp",
            "test-agent-web": "POST /test-agent-web - Agente para frontend web con contexto",
            "health": "GET /health - Estado de la API"
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
        
        # Preparar contexto de conversación desde el request si existe
        conversation_context = None
        if request.context and request.context.history:
            conversation_context = request.context.history
            logger.debug(f"📚 Contexto de conversación proporcionado: {len(conversation_context)} mensajes")
        
        # Procesar query con el agente (sin cliente Kapso, pero con contexto si está disponible)
        agent_start = time.time()
        result = await process_user_query(
            user, 
            request.message, 
            kapso_client=None,
            conversation_context=conversation_context
        )
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
        
        # Construir historial actualizado usando el historial que devolvió el agente
        conversation_history = result.get("conversation_history", [])
        
        # Agregar el mensaje del usuario actual y la respuesta del agente
        from datetime import datetime
        current_timestamp = datetime.utcnow().isoformat() + "Z"
        
        conversation_history.append(ConversationMessage(
            timestamp=current_timestamp,
            sender="client",
            message=request.message,
            message_id=f"{request.conversation_id}-{len(conversation_history)}"
        ))
        
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



stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

class CheckoutRequest(BaseModel):
    """Request para crear una sesión de checkout de Stripe"""
    conversation_id: str = Field(..., description="ID de la conversación (usuario)")
    user_name: str = Field(..., description="Nombre del usuario")
    phone_number: str = Field(..., description="Número de teléfono del usuario")

class CheckoutResponse(BaseModel):
    """Response con la URL de checkout de Stripe"""
    status: str
    checkout_url: Optional[str] = None
    session_id: Optional[str] = None
    error: Optional[str] = None
    cart_items_count: int = 0
    total_amount: float = 0.0

# app/main.py - Endpoint simplificado

@app.post("/create-checkout-session", response_model=CheckoutResponse)
async def create_checkout_session(request: CheckoutRequest):
    """
    Crea una sesión de checkout de Stripe para el carrito del usuario.
    Solo requiere: conversation_id, user_name, phone_number
    
    Ejemplo:
```json
    {
        "conversation_id": "whatsapp-521234567890",
        "user_name": "Juan Pérez",
        "phone_number": "+521234567890"
    }
```
    """
    start_time = time.time()
    
    logger.info(
        "💳 [CHECKOUT] Nueva solicitud - "
        f"Usuario: '{request.user_name}', "
        f"Phone: {request.phone_number}"
    )
    
    try:
        # 1. Buscar carrito del usuario
        cart_id = get_cart_by_conversation(request.conversation_id)
        
        if not cart_id:
            logger.warning(f"⚠️ No existe carrito para: {request.conversation_id}")
            return CheckoutResponse(
                status="error",
                error="No se encontró un carrito para este usuario",
                cart_items_count=0,
                total_amount=0.0
            )
        
        # 2. Obtener items del carrito
        cart_items = get_cart_items(cart_id)
        
        if not cart_items:
            logger.warning(f"⚠️ Carrito vacío")
            return CheckoutResponse(
                status="error",
                error="El carrito está vacío",
                cart_items_count=0,
                total_amount=0.0
            )
        
        # 3. Calcular total
        total_amount = calculate_cart_total(cart_id)
        logger.info(f"💰 Total: ${total_amount:.2f} MXN con {len(cart_items)} items")
        
        # 4. Crear line_items para Stripe
        line_items = []
        for item in cart_items:
            unit_amount = int(item['price'] * 100)  # Convertir a centavos
            
            line_items.append({
                'price_data': {
                    'currency': 'mxn',
                    'product_data': {
                        'name': item['name'],
                        'description': f"{item['type']} - {item['color']}" if item['color'] else item['type'],
                        'images': [item['image_url']] if item['image_url'] else [],
                    },
                    'unit_amount': unit_amount,
                },
                'quantity': item['quantity'],
            })
        
        # 5. Crear sesión de Stripe (SIN email)
        frontend_url = os.getenv("FRONTEND_URL")
        
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=line_items,
            mode='payment',
            success_url=f"{frontend_url}/checkout/success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{frontend_url}/checkout/cancel",
            client_reference_id=request.conversation_id,
            metadata={
                'conversation_id': request.conversation_id,
                'user_name': request.user_name,
                'phone_number': request.phone_number,
                'cart_id': str(cart_id)
            },
            billing_address_collection='required',
            shipping_address_collection={
                'allowed_countries': ['MX'],
            },
            phone_number_collection={
                'enabled': True
            }
        )
        
        total_time = time.time() - start_time
        
        logger.info(
            f"✅ [CHECKOUT] Sesión creada - "
            f"ID: {checkout_session.id}, "
            f"Items: {len(cart_items)}, "
            f"Total: ${total_amount:.2f} MXN, "
            f"Tiempo: {total_time:.2f}s"
        )
        
        return CheckoutResponse(
            status="success",
            checkout_url=checkout_session.url,
            session_id=checkout_session.id,
            cart_items_count=len(cart_items),
            total_amount=total_amount
        )
        
    except stripe.error.StripeError as e:
        logger.error(f"❌ Error de Stripe: {str(e)}")
        return CheckoutResponse(
            status="error",
            error=f"Error al crear sesión de pago: {str(e)}",
            cart_items_count=0,
            total_amount=0.0
        )
    
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# app/main.py - Agregar después del checkout

class CheckoutSuccessResponse(BaseModel):
    """Response después de completar el pago"""
    status: str
    message: str
    order_id: Optional[int] = None
    total_amount: float = 0.0
    items_count: int = 0
    error: Optional[str] = None

@app.get("/checkout/success", response_model=CheckoutSuccessResponse)
async def checkout_success(session_id: str = Query(..., description="Stripe session ID")):
    """
    Endpoint de éxito después del pago.
    Guarda la orden en la BD y vacía el carrito.
    
    Se llama automáticamente cuando Stripe redirige después del pago exitoso.
    URL: /checkout/success?session_id=cs_test_xxx
    """
    start_time = time.time()
    
    logger.info(f"🎉 [SUCCESS] Procesando pago exitoso - SessionID: {session_id}")
    
    try:
        # 1. Obtener información de la sesión de Stripe
        session = stripe.checkout.Session.retrieve(
            session_id,
            expand=['line_items', 'payment_intent']
        )
        
        if session.payment_status != 'paid':
            logger.warning(f"⚠️ Sesión no pagada: {session.payment_status}")
            return CheckoutSuccessResponse(
                status="error",
                message="El pago aún no se ha completado",
                error=f"Payment status: {session.payment_status}"
            )
        
        # 2. Extraer metadata
        conversation_id = session.client_reference_id
        user_name = session.metadata.get('user_name', 'Cliente')
        phone_number = session.metadata.get('phone_number', '')
        cart_id = int(session.metadata.get('cart_id'))
        
        logger.info(f"📋 Datos: Usuario={user_name}, Phone={phone_number}, CartID={cart_id}")
        
        # 3. Verificar si ya existe la orden (evitar duplicados)
        conn = get_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT id FROM orders 
            WHERE stripe_session_id = ?
        """, [session_id])
        
        existing_order = cur.fetchone()
        
        if existing_order:
            logger.warning(f"⚠️ Orden ya existe: ID {existing_order[0]}")
            
            # Contar items de la orden existente
            cur.execute("""
                SELECT COUNT(*), SUM(subtotal) 
                FROM order_items 
                WHERE order_id = ?
            """, [existing_order[0]])
            
            count_result = cur.fetchone()
            items_count = count_result[0] if count_result else 0
            total = count_result[1] if count_result else 0.0
            
            conn.close()
            
            return CheckoutSuccessResponse(
                status="success",
                message="Orden ya fue procesada anteriormente",
                order_id=existing_order[0],
                total_amount=total,
                items_count=items_count
            )
        
        # 4. Obtener items del carrito
        cart_items = get_cart_items(cart_id)
        total_amount = calculate_cart_total(cart_id)
        
        logger.info(f"🛒 Items del carrito: {len(cart_items)}, Total: ${total_amount:.2f}")
        
        # 5. Obtener dirección de envío (si existe)
        shipping_address = None
        shipping_details = getattr(session, 'shipping_details', None)
        if shipping_details and hasattr(shipping_details, 'address'):
            address = shipping_details.address
            if address:
                shipping_address = f"{address.line1}, {address.city}, {address.state} {address.postal_code}, {address.country}"
        
        # 6. Crear orden en la BD
        cur.execute("""
            INSERT INTO orders (
                conversation_id,
                user_name,
                phone_number,
                stripe_session_id,
                stripe_payment_intent,
                total_amount,
                currency,
                status,
                shipping_address
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            conversation_id,
            user_name,
            phone_number,
            session_id,
            session.payment_intent if hasattr(session, 'payment_intent') else None,
            total_amount,
            'MXN',
            'completed',
            shipping_address
        ])
        
        order_id = cur.lastrowid
        logger.info(f"✅ Orden creada: ID {order_id}")
        
        # 7. Copiar items del carrito a order_items
        for item in cart_items:
            cur.execute("""
                INSERT INTO order_items (
                    order_id,
                    article_id,
                    prod_name,
                    price_mxn,
                    quantity,
                    subtotal
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, [
                order_id,
                item['article_id'],
                item['name'],
                item['price'],
                item['quantity'],
                item['subtotal']
            ])
        
        logger.info(f"📦 {len(cart_items)} items agregados a la orden")
        
        conn.commit()
        conn.close()
        
        # 8. Enviar mensaje de confirmación de compra por WhatsApp
        if conversation_id:
            try:
                with KapsoClient() as kapso:
                    # Construir mensaje de confirmación
                    items_summary = []
                    for i, item in enumerate(cart_items[:10], start=1):  # Limitar a 10 items para el mensaje
                        items_summary.append(
                            f"{i}. {item['name']} x{item['quantity']} - ${item['subtotal']:.2f} MXN"
                        )
                    
                    if len(cart_items) > 10:
                        items_summary.append(f"... y {len(cart_items) - 10} producto(s) más")
                    
                    confirmation_message = (
                        f"🎉 *¡COMPRA CONFIRMADA!*\n\n"
                        f"✅ Tu orden #{order_id} ha sido procesada exitosamente.\n\n"
                        f"*Resumen de tu compra:*\n"
                        f"{chr(10).join(items_summary)}\n\n"
                        f"━━━━━━━━━━━━━━━━━━\n"
                        f"💰 *Total pagado: ${total_amount:.2f} MXN*\n"
                        f"━━━━━━━━━━━━━━━━━━\n\n"
                    )
                    
                    if shipping_address:
                        confirmation_message += f"📍 *Dirección de envío:*\n{shipping_address}\n\n"
                    
                    confirmation_message += (
                        f"📦 Recibirás un correo de confirmación con los detalles de tu pedido.\n"
                        f"🚚 Te notificaremos cuando tu orden sea enviada.\n\n"
                        f"¡Gracias por tu compra! 🛍️"
                    )
                    
                    kapso.send_message(conversation_id, confirmation_message)
                    logger.info(f"✅ Mensaje de confirmación enviado a conversación {conversation_id}")
            except Exception as e:
                logger.error(f"❌ Error enviando mensaje de confirmación: {e}")
        
        # 9. Vaciar el carrito
        clear_cart_by_id(cart_id)
        logger.info(f"🗑️ Carrito {cart_id} vaciado")
        
        total_time = time.time() - start_time
        
        logger.info(
            f"🎊 [SUCCESS] Orden completada exitosamente - "
            f"OrderID: {order_id}, "
            f"Usuario: {user_name}, "
            f"Items: {len(cart_items)}, "
            f"Total: ${total_amount:.2f} MXN, "
            f"Tiempo: {total_time:.2f}s"
        )
        
        return CheckoutSuccessResponse(
            status="success",
            message="¡Pago completado exitosamente! Tu orden ha sido registrada.",
            order_id=order_id,
            total_amount=total_amount,
            items_count=len(cart_items)
        )
        
    except stripe.error.StripeError as e:
        logger.error(f"❌ Error de Stripe en success: {str(e)}")
        return CheckoutSuccessResponse(
            status="error",
            message="Error verificando el pago",
            error=str(e)
        )
    
    except Exception as e:
        logger.error(f"❌ Error procesando orden: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/checkout/cancel")
async def checkout_cancel():
    """
    Endpoint cuando el usuario cancela el pago
    """
    logger.info("❌ [CANCEL] Usuario canceló el pago")
    
    return {
        "status": "cancelled",
        "message": "Pago cancelado. Tu carrito sigue disponible."
    }

# Historial de órdenes

@app.get("/orders/{conversation_id}")
async def get_user_orders(conversation_id: str):
    """
    Obtiene todas las órdenes de un usuario
    """
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute("""
        SELECT 
            o.id,
            o.stripe_session_id,
            o.total_amount,
            o.status,
            o.created_at,
            COUNT(oi.id) as items_count
        FROM orders o
        LEFT JOIN order_items oi ON o.id = oi.order_id
        WHERE o.conversation_id = ?
        GROUP BY o.id
        ORDER BY o.created_at DESC
    """, [conversation_id])
    
    orders = []
    for row in cur.fetchall():
        orders.append({
            "order_id": row[0],
            "session_id": row[1],
            "total": row[2],
            "status": row[3],
            "date": row[4],
            "items_count": row[5]
        })
    
    conn.close()
    
    return {
        "conversation_id": conversation_id,
        "orders_count": len(orders),
        "orders": orders
    }


@app.get("/orders/{conversation_id}/{order_id}")
async def get_order_details(conversation_id: str, order_id: int):
    """
    Obtiene detalles completos de una orden específica
    """
    conn = get_connection()
    cur = conn.cursor()
    
    # Verificar que la orden pertenece al usuario
    cur.execute("""
        SELECT 
            id, user_name, phone_number, total_amount, 
            status, shipping_address, created_at
        FROM orders
        WHERE id = ? AND conversation_id = ?
    """, [order_id, conversation_id])
    
    order = cur.fetchone()
    
    if not order:
        conn.close()
        raise HTTPException(status_code=404, detail="Orden no encontrada")
    
    # Obtener items de la orden
    cur.execute("""
        SELECT 
            article_id, prod_name, price_mxn, quantity, subtotal
        FROM order_items
        WHERE order_id = ?
    """, [order_id])
    
    items = []
    for row in cur.fetchall():
        items.append({
            "article_id": row[0],
            "name": row[1],
            "price": row[2],
            "quantity": row[3],
            "subtotal": row[4]
        })
    
    conn.close()
    
    return {
        "order_id": order[0],
        "user_name": order[1],
        "phone_number": order[2],
        "total_amount": order[3],
        "status": order[4],
        "shipping_address": order[5],
        "created_at": order[6],
        "items": items
    }