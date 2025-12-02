# app/embeddings.py
from openai import OpenAI
import os
from datetime import datetime

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Contador global
embedding_counter = {"total": 0, "session_start": datetime.now()}

def embed_text(text: str) -> list:
    """
    Usa 1536 dimensiones - sweet spot de rendimiento/precisión
    
    Beneficios:
    - 2x más rápido que 3072
    - Solo ~1-2% menos preciso
    - Usa la mitad de almacenamiento
    - Perfecto para e-commerce con miles de productos
    """
    
    # Registrar inicio
    start_time = datetime.now()
    print(f"\n{'='*60}")
    print(f"🔄 Generando embedding #{embedding_counter['total'] + 1}")
    print(f"📅 Timestamp: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📝 Texto (primeros 100 chars): {text[:100]}...")
    print(f"{'='*60}")
    
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text,
        dimensions=1536
    )
    
    # Registrar fin y estadísticas
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    embedding_counter["total"] += 1
    
    print(f"✅ Embedding generado exitosamente")
    print(f"⏱️  Tiempo de generación: {duration:.3f} segundos")
    print(f"📊 Total embeddings en esta sesión: {embedding_counter['total']}")
    print(f"🕐 Tiempo desde inicio: {(end_time - embedding_counter['session_start']).total_seconds():.1f}s")
    print(f"{'='*60}\n")
    
    return response.data[0].embedding

def get_embedding_stats():
    """Obtener estadísticas de embeddings"""
    return {
        "total": embedding_counter["total"],
        "session_start": embedding_counter["session_start"],
        "uptime": (datetime.now() - embedding_counter["session_start"]).total_seconds()
    }
