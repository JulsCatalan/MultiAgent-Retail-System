# app/stripe_integration.py
import stripe
import os
from typing import List
from .main import CartItem

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

def create_stripe_checkout_session(
    cart_items: List[CartItem],
    customer_name: str,
    customer_phone: str,
    success_url: str,
    cancel_url: str
):
    """
    Crea una sesión real de Stripe Checkout
    """
    
    line_items = []
    for item in cart_items:
        line_items.append({
            "price_data": {
                "currency": "mxn",
                "product_data": {
                    "name": item.name,
                    "description": f"{item.brand} - {item.category}",
                    "images": [item.image] if item.image else [],
                },
                "unit_amount": int(item.price * 100),  # Stripe usa centavos
            },
            "quantity": item.quantity,
        })
    
    session = stripe.checkout.Session.create(
        payment_method_types=["card"],
        line_items=line_items,
        mode="payment",
        success_url=success_url,
        cancel_url=cancel_url,
        metadata={
            "customer_name": customer_name,
            "customer_phone": customer_phone,
        },
        customer_email=None,  # Puedes agregar email si lo capturas
        shipping_address_collection={
            "allowed_countries": ["MX"],
        },
    )
    
    return {
        "checkout_url": session.url,
        "session_id": session.id
    }