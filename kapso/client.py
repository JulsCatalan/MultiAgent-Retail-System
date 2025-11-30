import os
from typing import Any, Dict, List, Optional, Union

import httpx


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
            print(f"DEBUG: Send message response: {response_json}")
        except Exception as e:
            print(f"DEBUG: Could not parse response JSON: {e}")
        
        response.raise_for_status()
        return response_json


__all__ = ["KapsoClient"]
