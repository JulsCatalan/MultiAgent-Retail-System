"""Main module for the FastAPI application."""
from fastapi import FastAPI
import uvicorn



app = FastAPI()

@app.get("/ping")
def ping():
    return {"message": "pong"}
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)