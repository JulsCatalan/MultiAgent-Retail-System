# app/main.py
from fastapi import FastAPI, Request, HTTPException
from dotenv import load_dotenv
import os

load_dotenv()

from .db import init_db, count_embeddings
from .loader import load_products_to_db
from kapso.use_kapso import use_kapso

app = FastAPI(title="Fashion Store API", version="1.0.0")

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

@app.get("/")
def root():
    return {
        "status": "API funcionando",
        "version": "1.0.0",
        "endpoints": {
            "whatsapp": "/whatsapp",
            "health": "/health"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint para Render"""
    return {
        "status": "healthy",
        "embeddings_count": count_embeddings()
    }

@app.post("/whatsapp")
async def whatsapp_agent(request: Request):
    """
    Endpoint que recibe webhooks de Kapso (WhatsApp)
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