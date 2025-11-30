# app/agents/generator.py
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_response(user_message: str, products: list = None) -> str:
    """
    Generator agent: Genera respuesta natural basada en productos encontrados
    """
    
    if products:
        # Formateamos los productos para el prompt
        products_text = "\n\n".join([
            f"""Producto {i+1}:
- Nombre: {p['prod_name']}
- Tipo: {p['product_type_name']}
- Grupo: {p['product_group_name']}
- Color: {p['colour_group_name']}
- Descripción: {p['detail_desc']}
- Precio: ${p['price_mxn']:.2f} MXN
- ID: {p['article_id']}"""
            for i, p in enumerate(products)
        ])
        
        prompt = f"""Eres un asistente de ventas amigable de una tienda de ropa.

El cliente preguntó: "{user_message}"

He encontrado estos productos relevantes:

{products_text}

Genera una respuesta natural y útil que:
1. Responda directamente a la pregunta del cliente
2. Recomiende los productos más adecuados
3. Destaque características clave
4. Sea conversacional y amigable
5. Menciona los precios en pesos mexicanos

NO uses formato de lista numerada, sé conversacional."""

    else:
        # Respuesta general sin productos
        prompt = f"""Eres un asistente amigable de una tienda de ropa.

El cliente dijo: "{user_message}"

Responde de manera natural, amigable y profesional. Si es un saludo, saluda de vuelta. Si pregunta algo general sobre la tienda, responde apropiadamente."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.7
    )
    
    return response.choices[0].message.content