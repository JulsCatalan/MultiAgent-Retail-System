# app/agents/orchestrator.py
from .router import route_query
from .retriever import search_products
from .generator import generate_response
import logging  

logger = logging.getLogger(__name__)

async def process_user_query(user_message: str) -> dict:
    """
    Orquestador principal: coordina los tres agentes
    """
    
    logger.info("💬 Consulta del usuario: %s", user_message)
    
    # 1. Router decide qué hacer
    routing = route_query(user_message)
    logger.info("🧭 Router decidió: %s", routing['decision'])
    
    products = None
    
    # 2. Si necesita búsqueda, activamos retriever
    if routing["decision"] == "search":
        products = search_products(user_message)
        logger.info("📦 Productos encontrados: %s", len(products))
    
    # 3. Generator crea la respuesta final
    response = generate_response(user_message, products)
    
    return {
        "response": response,
        "products": products,
        "routing_decision": routing["decision"]
    }