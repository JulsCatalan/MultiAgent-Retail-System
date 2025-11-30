# app/main.py
from fastapi import FastAPI
from dotenv import load_dotenv
load_dotenv()

from .db import init_db, count_embeddings
from .loader import load_products_to_db

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
