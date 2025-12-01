"""Schemas/models for the FastAPI application."""
from typing import Optional, List
from pydantic import BaseModel


class Product(BaseModel):
    id: int
    name: str
    brand: str
    category: str
    price: float
    image: str
    description: Optional[str] = None
    color: Optional[str] = None
    type: Optional[str] = None


class SearchConstraints(BaseModel):
    category: Optional[str] = None
    brand: Optional[str] = None
    color: Optional[str] = None
    size: Optional[str] = None
    price_min: Optional[float] = None
    price_max: Optional[float] = None


class SearchRequest(BaseModel):
    query: str
    constraints: SearchConstraints = SearchConstraints()
    k: int = 12


class SearchResponse(BaseModel):
    items: List[Product]
    total: int


class CartItem(BaseModel):
    id: int
    name: str
    brand: str
    category: str
    price: float
    image: str
    quantity: int
    color: Optional[str] = None
    size: Optional[str] = None


class CheckoutRequest(BaseModel):
    cart: List[CartItem]
    customer_name: str
    customer_phone: str


class CheckoutResponse(BaseModel):
    checkout_url: str
    session_id: str

