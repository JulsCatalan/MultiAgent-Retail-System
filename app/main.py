"""Main module for the FastAPI application."""
from fastapi import FastAPI, Request, HTTPException
import uvicorn

from app.models import ConversationalResponse
from kapso.use_kapso import use_kapso

app = FastAPI()


@app.post("/whatsapp" )
async def whatsapp_agent(request: Request) -> ConversationalResponse:
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