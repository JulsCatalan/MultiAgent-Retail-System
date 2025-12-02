import os
from typing import Any, Dict, List, Optional, Union

import httpx
import logging

logger = logging.getLogger(__name__)

class KapsoClient:
    """Minimal Kapso API client focused on WhatsApp templates.

    Environment variables:
    - KAPSO_BASE_URL: base URL, e.g., https://app.kapso.ai/api/v1
    - KAPSO_API_KEY: API key for authentication
    - KAPSO_SEND_TEMPLATE_PATH: optional override for the send-template endpoint path
    """

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None, timeout_seconds: int = 15) -> None:
        self.base_url = (base_url or os.getenv("KAPSO_BASE_URL") or "").rstrip("/")
        self.api_key = api_key or os.getenv("KAPSO_API_KEY") or ""
        self.timeout_seconds = timeout_seconds
        if not self.base_url:
            raise ValueError("KAPSO_BASE_URL is required")
        if not self.api_key:
            raise ValueError("KAPSO_API_KEY is required")

        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"X-API-Key": self.api_key, "Content-Type": "application/json"},
            timeout=self.timeout_seconds,
        )
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close the client."""
        if hasattr(self, '_client'):
            self._client.close()
    
    def close(self):
        """Explicitly close the HTTP client."""
        if hasattr(self, '_client'):
            self._client.close()


    def list_templates(
        self,
        page: int = 1,
        per_page: int = 20,
        name_contains: Optional[str] = None,
        language_code: Optional[str] = None,
        category: Optional[str] = None,
        status: Optional[str] = None,
        customer_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List available WhatsApp templates.

        Mirrors documentation in documentation/kapso/templates/list_templates.md
        """
        params: Dict[str, Any] = {"page": page, "per_page": per_page}
        if customer_id:
            params["customer_id"] = customer_id
            params["q[customer_id_eq]"] = customer_id
        if name_contains:
            params["q[name_cont]"] = name_contains
        if language_code:
            params["q[language_code_eq]"] = language_code
        if category:
            params["q[category_eq]"] = category
        if status:
            params["q[status_eq]"] = status

        response = self._client.get("/whatsapp_templates", params=params)
        response.raise_for_status()
        return response.json()

    def get_template_info(self, template_id: str) -> Dict[str, Any]:
        """Get template details including parameter requirements."""
        url = f"/whatsapp_templates/{template_id}"
        response = self._client.get(url)
        response.raise_for_status()
        result = response.json()
        
        return result

    def mark_as_read(self, message_id: str, typing_indicator: bool = False) -> Dict[str, Any]:
        """Mark a WhatsApp message as read.
        
        Args:
            message_id: The ID of the message to mark as read
            typing_indicator: Whether to show typing indicator after marking as read
            
        Returns:
            Response from the API
        """
        url = f"/whatsapp_messages/{message_id}/mark_as_read"
        params = {}
        if typing_indicator:
            params['typing_indicator'] = str(typing_indicator).lower()
            
        response = self._client.patch(url, params=params)
        # We don't raise for status here to match existing logic of handling errors gracefully
        # but ideally we should. For now let's return json or empty dict on error.
        if response.status_code == 200:
            return response.json()
        return {"error": f"Failed with status {response.status_code}", "status_code": response.status_code}

    def send_template_by_id(
        self,
        template_id: str,
        phone_number_e164: str,
        template_parameters: Optional[Union[List[str], Dict[str, str]]] = None,
        header_type: Optional[str] = None,
        header_params: Optional[str] = None,
        header_filename: Optional[str] = None,
        button_url_params: Optional[Dict[str, str]] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send a WhatsApp template message using template ID.
        
        Based on documentation/kapso/templates/send_template.md spec.
        """
        body: Dict[str, Any] = {
            "template": {
                "phone_number": phone_number_e164,
            }
        }
        
        if template_parameters is not None:
            body["template"]["template_parameters"] = template_parameters
        if header_type:
            body["template"]["header_type"] = header_type
        if header_params:
            body["template"]["header_params"] = header_params
        if header_filename:
            body["template"]["header_filename"] = header_filename
        if button_url_params:
            body["template"]["button_url_params"] = button_url_params
        if extra_payload:
            body["template"].update(extra_payload)

        url = f"/whatsapp_templates/{template_id}/send_template"
        
        response = self._client.post(url, json=body)
        
        
        response.raise_for_status()
        return response.json()

    def get_conversation_messages(
        self,
        conversation_id: str,
        page: int = 1,
        per_page: int = 100,
        
    ) -> Dict[str, Any]:
        """Get messages from a specific conversation.
        
        Args:
            conversation_id: ID of the conversation
            page: Page number (default 1)
            per_page: Messages per page (default 50)
            order: Sort order (default created_at_desc)
            
        Returns:
            Dict with conversation messages data
        """
        url = f"/whatsapp_conversations/{conversation_id}/whatsapp_messages"
        params = {
            "page": page,
            "per_page": per_page,
        }
        

        response = self._client.get(url, params=params)
        
        
        response.raise_for_status()
        result = response.json()
        
        return result

    def disable_typing_indicator(self, conversation_id: str) -> Dict[str, Any]:
        """Disable typing indicator for a conversation."""
        url = f"/whatsapp_conversations/{conversation_id}/typing"
        data = {"typing": False}
        response = self._client.patch(url, json=data)
        
        if response.status_code in [200, 204]:
            return {"success": True}
        return {"success": False, "status_code": response.status_code}

    def send_message(self, conversation_id: str, message: str) -> Dict[str, Any]:
        """Send a text message to a specific conversation.
        
        Args:
            conversation_id: ID of the conversation
            message: Text content to send
            
        Returns:
            Dict with the response from Kapso API
        """
        url = f"/whatsapp_conversations/{conversation_id}/whatsapp_messages"
        body = {
            "message": {
                "content": message,
                "message_type": "text"
            }
        }
        
        response = self._client.post(url, json=body)
        
        # Log response for debugging
        try:
            response_json = response.json()
            logger.debug(f"Send message response: {response_json}")
        except Exception as e:
            logger.debug(f"Could not parse response JSON: {e}")
        
        response.raise_for_status()
        return response_json

    def send_image_message(
        self, 
        conversation_id: str, 
        image_url: str, 
        caption: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send an image message to a specific conversation.
        
        Args:
            conversation_id: ID of the conversation
            image_url: Public URL of the image to send
            caption: Optional caption/description for the image
            
        Returns:
            Dict with the response from Kapso API
        """
        url = f"/whatsapp_conversations/{conversation_id}/whatsapp_messages"
        body = {
            "message": {
                "message_type": "image",
                "media_url": image_url,
            }
        }
        
        if caption:
            body["message"]["caption"] = caption
            body["message"]["content"] = caption
        
        logger.debug(f"📷 Sending image to {conversation_id}: {image_url[:50] if image_url else 'no-url'}...")
        
        try:
            response = self._client.post(url, json=body)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"❌ Error sending image: {e}")
            raise

    def send_product_with_image(
        self,
        conversation_id: str,
        product: Dict[str, Any],
        position: int = 1
    ) -> Dict[str, Any]:
        """Send a product card with image and details.
        
        Args:
            conversation_id: ID of the conversation
            product: Product dict with name, price, color, image_url, etc.
            position: Product position number (1, 2, 3...)
            
        Returns:
            Dict with the response from Kapso API
        """
        # Build caption with product details
        name = product.get("prod_name") or product.get("name", "Producto")
        price = product.get("price_mxn") or product.get("price", 0)
        color = product.get("colour_group_name") or product.get("color", "")
        category = product.get("product_group_name") or product.get("category", "")
        product_type = product.get("product_type_name") or product.get("type", "")
        image_url = product.get("image_url", "")
        
        # Build formatted caption
        caption = f"*Producto {position}:* {name}\n"
        if color:
            caption += f"🎨 Color: {color}\n"
        if product_type:
            caption += f"👕 Tipo: {product_type}\n"
        if category:
            caption += f"📦 Categoría: {category}\n"
        caption += f"💰 *Precio: ${price:.2f} MXN*\n"
        caption += f"\n_Responde \"Agrega el producto {position}\" para añadirlo al carrito_"
        
        # If no valid image URL, fall back to text-only
        if not image_url or not image_url.startswith("http"):
            logger.warning(f"⚠️ Product {position} has no valid image URL, sending text only")
            return self.send_message(conversation_id, caption)
        
        return self.send_image_message(conversation_id, image_url, caption)

    def send_cart_item_with_image(
        self,
        conversation_id: str,
        item: Dict[str, Any],
        position: int = 1
    ) -> Dict[str, Any]:
        """Send a cart item with image and details.
        
        Args:
            conversation_id: ID of the conversation
            item: Cart item dict with name, price, color, image_url, quantity, etc.
            position: Item position in cart (1, 2, 3...)
            
        Returns:
            Dict with the response from Kapso API
        """
        name = item.get("prod_name") or item.get("name", "Producto")
        price = item.get("price_mxn") or item.get("price", 0)
        color = item.get("colour_group_name") or item.get("color", "")
        quantity = item.get("quantity", 1)
        image_url = item.get("image_url", "")
        subtotal = price * quantity
        
        # Build formatted caption for cart item
        caption = f"*{position}. {name}*\n"
        if color:
            caption += f"🎨 Color: {color}\n"
        caption += f"📦 Cantidad: {quantity}\n"
        caption += f"💰 Subtotal: *${subtotal:.2f} MXN*"
        
        # If no valid image URL, fall back to text-only
        if not image_url or not image_url.startswith("http"):
            logger.warning(f"⚠️ Cart item {position} has no valid image URL, sending text only")
            return self.send_message(conversation_id, caption)
        
        return self.send_image_message(conversation_id, image_url, caption)

    def send_products_with_images(
        self,
        conversation_id: str,
        products: List[Dict[str, Any]],
        intro_message: Optional[str] = None,
        max_images: int = 5
    ) -> List[Dict[str, Any]]:
        """Send multiple products with images.
        
        Args:
            conversation_id: ID of the conversation
            products: List of product dicts
            intro_message: Optional intro text before products
            max_images: Max number of product images to send (WhatsApp best practice)
            
        Returns:
            List of responses from each message sent
        """
        responses = []
        
        # Send intro message first if provided
        if intro_message:
            try:
                resp = self.send_message(conversation_id, intro_message)
                responses.append(resp)
            except Exception as e:
                logger.error(f"Error sending intro message: {e}")
        
        # Send product images (limit to avoid spam)
        for i, product in enumerate(products[:max_images], start=1):
            try:
                resp = self.send_product_with_image(conversation_id, product, position=i)
                responses.append(resp)
            except Exception as e:
                logger.error(f"Error sending product {i}: {e}")
                # Fall back to text if image fails
                name = product.get("prod_name") or product.get("name", "N/A")
                price = product.get("price_mxn") or product.get("price", 0)
                fallback_text = f"Producto {i}: {name} - ${price:.2f} MXN"
                try:
                    resp = self.send_message(conversation_id, fallback_text)
                    responses.append(resp)
                except Exception as e2:
                    logger.error(f"Error sending fallback text for product {i}: {e2}")
        
        # If more products available, send a note
        if len(products) > max_images:
            remaining = len(products) - max_images
            note = f"_...y {remaining} producto(s) más. Pídeme ver más opciones si te interesa._"
            try:
                resp = self.send_message(conversation_id, note)
                responses.append(resp)
            except Exception as e:
                logger.error(f"Error sending 'more products' note: {e}")
        
        return responses

    def send_cart_with_images(
        self,
        conversation_id: str,
        cart_items: List[Dict[str, Any]],
        total: float,
        header_message: Optional[str] = None,
        footer_message: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Send cart contents with images for each item.
        
        Args:
            conversation_id: ID of the conversation
            cart_items: List of cart item dicts
            total: Cart total
            header_message: Optional header text before items
            footer_message: Optional footer text after items (e.g., total + actions)
            
        Returns:
            List of responses from each message sent
        """
        responses = []
        
        # Send header if provided
        if header_message:
            try:
                resp = self.send_message(conversation_id, header_message)
                responses.append(resp)
            except Exception as e:
                logger.error(f"Error sending cart header: {e}")
        
        # Send each cart item with image
        for i, item in enumerate(cart_items, start=1):
            try:
                resp = self.send_cart_item_with_image(conversation_id, item, position=i)
                responses.append(resp)
            except Exception as e:
                logger.error(f"Error sending cart item {i}: {e}")
                # Fallback to text
                name = item.get("prod_name") or item.get("name", "N/A")
                price = item.get("price_mxn") or item.get("price", 0)
                qty = item.get("quantity", 1)
                fallback = f"{i}. {name} x{qty} - ${price * qty:.2f} MXN"
                try:
                    resp = self.send_message(conversation_id, fallback)
                    responses.append(resp)
                except Exception as e2:
                    logger.error(f"Error sending fallback for cart item {i}: {e2}")
        
        # Send footer with total and actions
        if footer_message:
            try:
                resp = self.send_message(conversation_id, footer_message)
                responses.append(resp)
            except Exception as e:
                logger.error(f"Error sending cart footer: {e}")
        else:
            # Default footer with total
            default_footer = (
                f"━━━━━━━━━━━━━━━━━━\n"
                f"💰 *TOTAL: ${total:.2f} MXN*\n"
                f"━━━━━━━━━━━━━━━━━━\n\n"
                f"¿Qué deseas hacer?\n"
                f"• \"Proceder al pago\" - Finalizar compra\n"
                f"• \"Quitar producto X\" - Eliminar un item\n"
                f"• \"Seguir comprando\" - Buscar más productos"
            )
            try:
                resp = self.send_message(conversation_id, default_footer)
                responses.append(resp)
            except Exception as e:
                logger.error(f"Error sending default cart footer: {e}")
        
        return responses

    def send_checkout_with_images(
        self,
        conversation_id: str,
        cart_items: List[Dict[str, Any]],
        total: float,
        checkout_url: str
    ) -> List[Dict[str, Any]]:
        """Send checkout summary with images for each item and payment link.
        
        Args:
            conversation_id: ID of the conversation
            cart_items: List of cart item dicts
            total: Cart total
            checkout_url: Stripe checkout URL
            
        Returns:
            List of responses from each message sent
        """
        responses = []
        
        # Header
        header = "🛒 *RESUMEN DE TU ORDEN*\n\nEstos son los productos que vas a comprar:"
        try:
            resp = self.send_message(conversation_id, header)
            responses.append(resp)
        except Exception as e:
            logger.error(f"Error sending checkout header: {e}")
        
        # Send each item with image
        for i, item in enumerate(cart_items, start=1):
            try:
                resp = self.send_cart_item_with_image(conversation_id, item, position=i)
                responses.append(resp)
            except Exception as e:
                logger.error(f"Error sending checkout item {i}: {e}")
        
        # Footer with total and payment link
        footer = (
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💰 *TOTAL A PAGAR: ${total:.2f} MXN*\n"
            f"━━━━━━━━━━━━━━━━━━\n\n"
            f"👉 *HAZ CLIC AQUÍ PARA PAGAR:*\n"
            f"{checkout_url}\n\n"
            f"✅ Pago 100% seguro con Stripe\n"
            f"🚚 Envío a toda la República Mexicana\n"
            f"📦 Recibirás confirmación de tu orden\n"
            f"⏰ Este link es válido por 24 horas\n\n"
            f"_Si deseas modificar tu carrito, dime \"seguir comprando\" o \"modificar carrito\"._"
        )
        try:
            resp = self.send_message(conversation_id, footer)
            responses.append(resp)
        except Exception as e:
            logger.error(f"Error sending checkout footer: {e}")
        
        return responses


__all__ = ["KapsoClient"]
