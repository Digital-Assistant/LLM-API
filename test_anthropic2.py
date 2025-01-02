import os
import requests
import json
from typing import Optional, Dict, Any

class AnthropicClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

    def create_message(
        self,
        messages: list,
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system: Optional[str] = None
    ) -> Dict[Any, Any]:
        """
        Send a message to Claude and get a response.
        
        Args:
            messages: List of message objects with 'role' and 'content'
            model: Model identifier to use
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-1)
            system: System message to set context
        
        Returns:
            Dict containing the API response
        """
        payload = {
            "messages": messages,
            "model": model
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if system is not None:
            payload["system"] = system

        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")

# Example usage
def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    client = AnthropicClient(api_key)
    
    messages = [
        {
            "role": "user",
            "content": "Hello, what is the weather in Tokyo?"
        }
    ]
    
    try:
        response = client.create_message(
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        print("Response:", response)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()