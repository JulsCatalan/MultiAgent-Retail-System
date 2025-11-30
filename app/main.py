# app/main.py
from fastapi import FastAPI, Request, HTTPException
from dotenv import load_dotenv
import uvicorn
load_dotenv()

from .db import init_db, count_embeddings
from .loader import load_products_to_db
from app.models import ConversationalResponse
from kapso.use_kapso import use_kapso


app = FastAPI()

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
    return {"status": "API funcionando"}

@app.post("/whatsapp" )
async def whatsapp_agent(request: Request):
    """
    Endpoint that receives a query, processes it using the agent logic,
    sends the response via Kapso (WhatsApp), and returns the response.
    """
    webhook_data = await request.json()
    
    result = await use_kapso(webhook_data)
    
    # Verificar el resultado
    if result.get("status") == "success":
        return result
    else:
        return HTTPException(status_code=500, detail=result.get("message", "Error procesando webhook"))

@app.get("/ping")
def ping():
    return {"message": "pong"}
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
